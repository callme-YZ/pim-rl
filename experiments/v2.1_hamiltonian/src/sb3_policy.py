"""
SB3-Compatible Hamiltonian Policy

Integrates HamiltonianPolicy components into Stable-Baselines3's ActorCriticPolicy
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import gymnasium as gym
import numpy as np

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor

try:
    from .encoder import LatentEncoder
    from .pseudo_hamiltonian import PseudoHamiltonianNetwork
except ImportError:
    # Direct import when run as script
    from encoder import LatentEncoder
    from pseudo_hamiltonian import PseudoHamiltonianNetwork


class HamiltonianFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor that encodes observations to latent space
    
    Replaces standard MLP feature extractor with our LatentEncoder
    """
    
    def __init__(self, 
                 observation_space: gym.spaces.Box,
                 latent_dim: int = 8):
        """
        Args:
            observation_space: Env observation space (113-dim for v2.0)
            latent_dim: Latent encoding dimension
        """
        # BaseFeaturesExtractor expects features_dim in __init__
        super().__init__(observation_space, features_dim=latent_dim)
        
        obs_dim = observation_space.shape[0]
        self.encoder = LatentEncoder(obs_dim=obs_dim, latent_dim=latent_dim)
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Encode observations to latent space
        
        Args:
            observations: (batch_size, obs_dim) tensor
            
        Returns:
            latent: (batch_size, latent_dim) tensor
        """
        return self.encoder(observations)


class HamiltonianActorCriticPolicy(ActorCriticPolicy):
    """
    Hamiltonian-Informed PPO Policy
    
    Extends SB3's ActorCriticPolicy with:
    1. Latent encoder (observation → latent space)
    2. Pseudo-Hamiltonian network H(z, a)
    3. Hamiltonian-guided action selection
    
    Hyperparameters:
        lambda_h: Hamiltonian guidance strength (0=baseline PPO, 1=strong guidance)
        latent_dim: Latent space dimension
        h_hidden_dim: Hamiltonian network hidden layer size
    """
    
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        lambda_h: float = 0.5,
        latent_dim: int = 8,
        h_hidden_dim: int = 64,
        *args,
        **kwargs,
    ):
        """
        Initialize Hamiltonian policy
        
        Args:
            observation_space: Environment observation space
            action_space: Environment action space
            lr_schedule: Learning rate schedule (from SB3)
            lambda_h: Hamiltonian guidance weight (0-1)
            latent_dim: Latent encoding dimension
            h_hidden_dim: Hamiltonian network hidden size
        """
        # Store Hamiltonian-specific hyperparameters
        self.lambda_h = lambda_h
        self.latent_dim = latent_dim
        self.h_hidden_dim = h_hidden_dim
        
        # Force custom features extractor
        kwargs["features_extractor_class"] = HamiltonianFeaturesExtractor
        kwargs["features_extractor_kwargs"] = dict(latent_dim=latent_dim)
        
        # Initialize parent ActorCriticPolicy
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )
        
        # Add Hamiltonian network (after parent init)
        # Get action dimension (handle both Box and Discrete)
        if isinstance(action_space, gym.spaces.Box):
            action_dim = action_space.shape[0]
        elif isinstance(action_space, gym.spaces.Discrete):
            action_dim = action_space.n
        else:
            raise ValueError(f"Unsupported action space: {type(action_space)}")
            
        self.h_network = PseudoHamiltonianNetwork(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dims=(h_hidden_dim, h_hidden_dim)  # Tuple of hidden dims
        )
        
        # Enable gradient computation for Hamiltonian guidance
        self.h_network.train()
        
    def forward(
        self, 
        obs: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with Hamiltonian guidance
        
        Args:
            obs: Observations (batch_size, obs_dim)
            deterministic: If True, return mean action (no sampling)
            
        Returns:
            actions: Selected actions (batch_size, action_dim)
            values: State values (batch_size, 1)
            log_prob: Action log probabilities (batch_size, 1)
        """
        # Extract latent features
        latent = self.extract_features(obs)  # Uses HamiltonianFeaturesExtractor
        
        # Get base policy distribution (standard PPO)
        latent_pi = self.mlp_extractor.forward_actor(latent)
        latent_vf = self.mlp_extractor.forward_critic(latent)
        
        # Actor: action distribution
        mean_actions = self.action_net(latent_pi)
        
        # Apply Hamiltonian guidance if lambda_h > 0
        if self.lambda_h > 0 and not deterministic:
            # Compute ∂H/∂a gradient (for guidance)
            with torch.enable_grad():
                # Create action tensor that requires grad
                actions_sample = mean_actions.detach().clone().requires_grad_(True)
                
                # Compute H(z, a)
                h_values = self.h_network(latent.detach(), actions_sample)
                
                # Compute ∂H/∂a
                h_grad = torch.autograd.grad(
                    h_values.sum(), 
                    actions_sample,
                    create_graph=False
                )[0]
            
            # Adjust action mean: a' = a - λ_H * ∂H/∂a
            # (gradient descent on H to find lower-energy actions)
            mean_actions = mean_actions - self.lambda_h * h_grad.detach()
        
        # Create distribution
        distribution = self._get_action_dist_from_latent(latent_pi)
        distribution.distribution.loc = mean_actions  # Override mean
        
        # Sample or deterministic
        if deterministic:
            actions = mean_actions
        else:
            actions = distribution.get_actions(deterministic=False)
        
        # Value prediction
        values = self.value_net(latent_vf)
        
        # Log probability
        log_prob = distribution.log_prob(actions)
        
        return actions, values, log_prob
    
    def evaluate_actions(
        self, 
        obs: torch.Tensor, 
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions (for PPO loss computation)
        
        Args:
            obs: Observations (batch_size, obs_dim)
            actions: Actions to evaluate (batch_size, action_dim)
            
        Returns:
            values: State values (batch_size, 1)
            log_prob: Action log probabilities (batch_size, 1)
            entropy: Distribution entropy (batch_size, 1)
        """
        # Extract features
        latent = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(latent)
        latent_vf = self.mlp_extractor.forward_critic(latent)
        
        # Get distribution
        distribution = self._get_action_dist_from_latent(latent_pi)
        
        # Log prob and entropy
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        # Values
        values = self.value_net(latent_vf)
        
        return values, log_prob, entropy
    
    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Predict state values
        
        Args:
            obs: Observations (batch_size, obs_dim)
            
        Returns:
            values: State values (batch_size, 1)
        """
        latent = self.extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(latent)
        return self.value_net(latent_vf)
