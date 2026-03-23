"""
Pseudo-Hamiltonian Network

Learns H(z, a) - NOT physical conserved Hamiltonian
Control-oriented differentiable function for ∂H/∂a gradient
"""

import torch
import torch.nn as nn
from typing import Tuple


class PseudoHamiltonianNetwork(nn.Module):
    """
    Pseudo-Hamiltonian: H(z, a) ∈ ℝ
    
    Physics Clarification (小P review):
    - H is NOT physical energy (MHD is dissipative)
    - H is NOT conserved (resistivity causes loss)
    - H is learned function for control guidance
    - ∂H/∂a provides useful gradient
    - Can interpret as Lyapunov-like (decreases with good control)
    
    Architecture:
    - Input: [z, a] concatenated
    - Output: scalar H
    - Fully connected with Tanh activation
    """
    
    def __init__(self,
                 latent_dim: int = 8,
                 action_dim: int = 4,
                 hidden_dims: Tuple[int, ...] = (256, 256, 128)):
        """
        Args:
            latent_dim: Latent space dimension
            action_dim: Action space dimension (4 for v2.0 RMP)
            hidden_dims: Hidden layer sizes
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.input_dim = latent_dim + action_dim
        
        # Build network
        layers = []
        in_dim = self.input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.Tanh()  # Tanh for bounded activations
            ])
            in_dim = hidden_dim
        
        # Final projection to scalar
        layers.append(nn.Linear(in_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Compute H(z, a)
        
        Args:
            z: (batch, latent_dim) latent state
            a: (batch, action_dim) action
        
        Returns:
            H: (batch, 1) pseudo-Hamiltonian value
        """
        x = torch.cat([z, a], dim=1)
        return self.network(x)
    
    def compute_gradient_wrt_action(self, 
                                    z: torch.Tensor,
                                    a: torch.Tensor) -> torch.Tensor:
        """
        Compute ∂H/∂a for control gradient
        
        This is the key quantity for Hamiltonian-informed control
        
        Args:
            z: (batch, latent_dim) latent state
            a: (batch, action_dim) action (requires_grad=True)
        
        Returns:
            dH_da: (batch, action_dim) gradient of H w.r.t. action
        """
        # Ensure action requires grad
        if not a.requires_grad:
            a = a.requires_grad_(True)
        
        # Compute H
        H = self.forward(z.detach(), a)  # Detach z, gradient only w.r.t. a
        
        # Compute gradient
        dH_da = torch.autograd.grad(
            outputs=H.sum(),
            inputs=a,
            create_graph=True,
            retain_graph=True
        )[0]
        
        return dH_da
    
    def lyapunov_trend(self,
                      z_trajectory: torch.Tensor,
                      a_trajectory: torch.Tensor) -> torch.Tensor:
        """
        Compute H trend over trajectory (Lyapunov interpretation)
        
        Good control → H should decrease (island suppression)
        
        Args:
            z_trajectory: (T, latent_dim) latent states over time
            a_trajectory: (T, action_dim) actions over time
        
        Returns:
            H_trend: (T,) H values over trajectory
        """
        T = z_trajectory.shape[0]
        H_values = []
        
        for t in range(T):
            z_t = z_trajectory[t:t+1]
            a_t = a_trajectory[t:t+1]
            H_t = self.forward(z_t, a_t)
            H_values.append(H_t.item())
        
        return torch.tensor(H_values)


class HamiltonianLoss(nn.Module):
    """
    Loss functions for training pseudo-Hamiltonian network
    
    Components:
    1. Value matching: H(z,a) ≈ V(z) (critic value)
    2. Gradient alignment: ∂H/∂a aligns with advantage
    """
    
    def __init__(self, 
                 value_weight: float = 1.0,
                 alignment_weight: float = 0.5):
        """
        Args:
            value_weight: Weight for value matching loss
            alignment_weight: Weight for gradient alignment loss
        """
        super().__init__()
        self.value_weight = value_weight
        self.alignment_weight = alignment_weight
    
    def forward(self,
                H_pred: torch.Tensor,
                V_target: torch.Tensor,
                dH_da: torch.Tensor,
                advantage: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute total Hamiltonian loss
        
        Args:
            H_pred: (batch, 1) predicted H values
            V_target: (batch, 1) target values (from critic)
            dH_da: (batch, action_dim) H gradient w.r.t. action
            advantage: (batch, 1) advantage estimates
        
        Returns:
            total_loss: scalar
            info: dict with loss components
        """
        # Loss 1: Value matching
        loss_value = torch.mean((H_pred - V_target) ** 2)
        
        # Loss 2: Gradient alignment
        # ∂H/∂a should point in direction of advantage
        # (negative dot product = alignment)
        # Handle advantage broadcasting
        if advantage.shape[1] == 1:
            advantage_expanded = advantage.expand_as(dH_da)
        else:
            advantage_expanded = advantage
        
        # Cosine similarity (normalized dot product)
        dH_norm = torch.nn.functional.normalize(dH_da, dim=1)
        adv_norm = torch.nn.functional.normalize(advantage_expanded, dim=1)
        
        # We want alignment (positive cosine sim)
        # So minimize negative cosine sim
        loss_alignment = -torch.mean(torch.sum(dH_norm * adv_norm, dim=1))
        
        # Total loss
        total_loss = (self.value_weight * loss_value + 
                     self.alignment_weight * loss_alignment)
        
        info = {
            'loss_value': loss_value.item(),
            'loss_alignment': loss_alignment.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, info


if __name__ == "__main__":
    # Smoke test
    print("=" * 60)
    print("Pseudo-Hamiltonian Network Smoke Test")
    print("=" * 60)
    
    # Create network
    phn = PseudoHamiltonianNetwork(latent_dim=8, action_dim=4)
    print(f"\nPHN parameters: {sum(p.numel() for p in phn.parameters())}")
    
    # Test forward pass
    batch_size = 32
    z = torch.randn(batch_size, 8)
    a = torch.randn(batch_size, 4)
    
    H = phn(z, a)
    print(f"\nInput z shape: {z.shape}")
    print(f"Input a shape: {a.shape}")
    print(f"Output H shape: {H.shape}")
    assert H.shape == (batch_size, 1), "Shape mismatch!"
    
    # Test gradient computation
    a_grad = a.requires_grad_(True)
    dH_da = phn.compute_gradient_wrt_action(z, a_grad)
    print(f"\n∂H/∂a shape: {dH_da.shape}")
    assert dH_da.shape == (batch_size, 4), "Gradient shape mismatch!"
    
    # Test loss
    loss_fn = HamiltonianLoss()
    V_target = torch.randn(batch_size, 1)
    advantage = torch.randn(batch_size, 1)
    
    total_loss, info = loss_fn(H, V_target, dH_da, advantage)
    print(f"\nLoss components:")
    for key, val in info.items():
        print(f"  {key}: {val:.4f}")
    
    print("\n✅ Pseudo-Hamiltonian smoke test passed!")
