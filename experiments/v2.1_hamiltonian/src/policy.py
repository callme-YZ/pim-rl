"""
Hamiltonian-Informed Policy

PPO policy with pseudo-Hamiltonian guidance
Integrates with Stable-Baselines3
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple, Optional
import numpy as np

from .encoder import LatentEncoder
from .pseudo_hamiltonian import PseudoHamiltonianNetwork


class HamiltonianActor(nn.Module):
    """
    Actor network with Hamiltonian guidance
    
    Output: action mean and log_std
    Incorporates ∂H/∂a gradient when lambda_H > 0
    """
    
    def __init__(self,
                 latent_dim: int = 8,
                 action_dim: int = 4,
                 hidden_dim: int = 128,
                 lambda_H: float = 0.5):
        """
        Args:
            latent_dim: Latent state dimension
            action_dim: Action space dimension
            hidden_dim: Hidden layer size
            lambda_H: Hamiltonian guidance weight (0=pure RL, 1=strong guidance)
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.lambda_H = lambda_H
        
        # Base actor (standard PPO)
        self.base_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Mean and log_std heads
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, 
                z: torch.Tensor,
                dH_da: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute action distribution parameters
        
        Args:
            z: (batch, latent_dim) latent state
            dH_da: (batch, action_dim) Hamiltonian gradient (optional)
        
        Returns:
            mean: (batch, action_dim) action mean
            log_std: (batch, action_dim) action log std
        """
        # Base features
        features = self.base_net(z)
        
        # Action mean (base)
        mean_base = self.mean_head(features)
        
        # Add Hamiltonian guidance if provided
        if dH_da is not None and self.lambda_H > 0:
            # Gradient descent on H: move in -∂H/∂a direction
            mean_final = mean_base - self.lambda_H * dH_da
        else:
            mean_final = mean_base
        
        # Action log_std (exploration)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, -20, 2)  # Stability
        
        return mean_final, log_std


class HamiltonianCritic(nn.Module):
    """
    Critic network (value function)
    Standard architecture, no H-specific modification
    """
    
    def __init__(self,
                 latent_dim: int = 8,
                 hidden_dim: int = 128):
        """
        Args:
            latent_dim: Latent state dimension
            hidden_dim: Hidden layer size
        """
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute state value
        
        Args:
            z: (batch, latent_dim) latent state
        
        Returns:
            value: (batch, 1) state value
        """
        return self.net(z)


class HamiltonianPolicy(nn.Module):
    """
    Complete Hamiltonian-informed policy
    
    Components:
    - Encoder: obs → z
    - Hamiltonian network: H(z, a)
    - Actor: z → action (with H guidance)
    - Critic: z → value
    """
    
    def __init__(self,
                 obs_dim: int = 113,
                 action_dim: int = 4,
                 latent_dim: int = 8,
                 lambda_H: float = 0.5):
        """
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            latent_dim: Latent encoding dimension
            lambda_H: Hamiltonian guidance weight
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.lambda_H = lambda_H
        
        # Components
        self.encoder = LatentEncoder(obs_dim, latent_dim)
        self.hamiltonian = PseudoHamiltonianNetwork(latent_dim, action_dim)
        self.actor = HamiltonianActor(latent_dim, action_dim, lambda_H=lambda_H)
        self.critic = HamiltonianCritic(latent_dim)
    
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation to latent"""
        return self.encoder(obs)
    
    def get_action(self, 
                   obs: torch.Tensor,
                   deterministic: bool = False) -> Tuple[torch.Tensor, dict]:
        """
        Sample action from policy
        
        Args:
            obs: (batch, obs_dim) observations
            deterministic: If True, return mean action
        
        Returns:
            action: (batch, action_dim)
            info: dict with diagnostics
        """
        # Encode
        z = self.encode(obs)
        
        # Get H gradient (if lambda_H > 0)
        if self.lambda_H > 0:
            # Sample initial action for gradient computation
            with torch.no_grad():
                mean_init, log_std_init = self.actor(z, dH_da=None)
                action_init = mean_init  # Use mean for gradient
            
            # Compute ∂H/∂a
            action_init_grad = action_init.requires_grad_(True)
            dH_da = self.hamiltonian.compute_gradient_wrt_action(z, action_init_grad)
        else:
            dH_da = None
        
        # Get action distribution
        mean, log_std = self.actor(z, dH_da)
        std = torch.exp(log_std)
        
        # Sample or take mean
        if deterministic:
            action = mean
        else:
            dist = Normal(mean, std)
            action = dist.sample()
        
        # Diagnostics
        info = {
            'mean': mean.detach(),
            'std': std.detach(),
            'z': z.detach()
        }
        
        if dH_da is not None:
            info['dH_da'] = dH_da.detach()
        
        return action, info
    
    def evaluate_actions(self,
                        obs: torch.Tensor,
                        actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions (for PPO training)
        
        Args:
            obs: (batch, obs_dim)
            actions: (batch, action_dim)
        
        Returns:
            values: (batch, 1) state values
            log_probs: (batch,) action log probabilities
            entropy: (batch,) action entropy
        """
        # Encode
        z = self.encode(obs)
        
        # Compute H gradient
        if self.lambda_H > 0:
            actions_grad = actions.requires_grad_(True)
            dH_da = self.hamiltonian.compute_gradient_wrt_action(z, actions_grad)
        else:
            dH_da = None
        
        # Action distribution
        mean, log_std = self.actor(z, dH_da)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        
        # Log prob and entropy
        log_probs = dist.log_prob(actions).sum(dim=1)
        entropy = dist.entropy().sum(dim=1)
        
        # Value
        values = self.critic(z)
        
        return values, log_probs, entropy
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get state value
        
        Args:
            obs: (batch, obs_dim)
        
        Returns:
            value: (batch, 1)
        """
        z = self.encode(obs)
        return self.critic(z)


# Parameter counting utility

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Smoke test
    print("=" * 60)
    print("Hamiltonian Policy Smoke Test")
    print("=" * 60)
    
    # Create policy
    policy = HamiltonianPolicy(
        obs_dim=113,
        action_dim=4,
        latent_dim=8,
        lambda_H=0.5
    )
    
    print(f"\nTotal parameters: {count_parameters(policy)}")
    print(f"  Encoder: {count_parameters(policy.encoder)}")
    print(f"  Hamiltonian: {count_parameters(policy.hamiltonian)}")
    print(f"  Actor: {count_parameters(policy.actor)}")
    print(f"  Critic: {count_parameters(policy.critic)}")
    
    # Test forward pass
    batch_size = 32
    obs = torch.randn(batch_size, 113)
    
    # Get action
    action, info = policy.get_action(obs, deterministic=False)
    print(f"\nAction shape: {action.shape}")
    print(f"Action mean shape: {info['mean'].shape}")
    print(f"Action std shape: {info['std'].shape}")
    
    # Evaluate actions
    values, log_probs, entropy = policy.evaluate_actions(obs, action)
    print(f"\nValues shape: {values.shape}")
    print(f"Log probs shape: {log_probs.shape}")
    print(f"Entropy shape: {entropy.shape}")
    
    # Get value only
    value = policy.get_value(obs)
    print(f"\nValue (standalone) shape: {value.shape}")
    
    # Test lambda_H=0 (pure RL)
    policy_baseline = HamiltonianPolicy(lambda_H=0.0)
    action_baseline, _ = policy_baseline.get_action(obs)
    print(f"\nBaseline action shape: {action_baseline.shape}")
    
    print("\n✅ Hamiltonian Policy smoke test passed!")
