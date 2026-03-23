"""
Latent Encoder for MHD Observations

Maps high-dim MHD state (113-dim) to low-dim latent space (8-dim)
No canonical (q,p) structure - generic encoding
"""

import torch
import torch.nn as nn
from typing import Tuple


class LatentEncoder(nn.Module):
    """
    Encoder: obs ∈ ℝ¹¹³ → z ∈ ℝᴰ
    
    Architecture:
    - FC layers with ReLU
    - Progressive dimension reduction
    - No physics constraints (generic encoding)
    
    Note: D (latent_dim) to be determined empirically (ablate 4,8,16)
    """
    
    def __init__(self, 
                 obs_dim: int = 113,
                 latent_dim: int = 8,
                 hidden_dims: Tuple[int, ...] = (256, 128, 64)):
        """
        Args:
            obs_dim: Observation dimension (113 for v2.0 MHD)
            latent_dim: Latent space dimension (default 8, to be validated)
            hidden_dims: Hidden layer sizes
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        
        # Build encoder
        layers = []
        in_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU()
            ])
            in_dim = hidden_dim
        
        # Final projection to latent
        layers.append(nn.Linear(in_dim, latent_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Encode observation to latent
        
        Args:
            obs: (batch, 113) MHD observations
        
        Returns:
            z: (batch, latent_dim) latent encoding
        """
        return self.encoder(obs)


class LatentDecoder(nn.Module):
    """
    Decoder: z ∈ ℝᴰ → obs ∈ ℝ¹¹³
    
    For pre-training encoder (optional)
    Validates that latent captures sufficient information
    """
    
    def __init__(self,
                 latent_dim: int = 8,
                 obs_dim: int = 113,
                 hidden_dims: Tuple[int, ...] = (64, 128, 256)):
        """
        Args:
            latent_dim: Latent space dimension
            obs_dim: Observation dimension
            hidden_dims: Hidden layer sizes (reverse of encoder)
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        
        # Build decoder
        layers = []
        in_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU()
            ])
            in_dim = hidden_dim
        
        # Final projection to observation space
        layers.append(nn.Linear(in_dim, obs_dim))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to observation
        
        Args:
            z: (batch, latent_dim) latent encoding
        
        Returns:
            obs_recon: (batch, 113) reconstructed observation
        """
        return self.decoder(z)


class Autoencoder(nn.Module):
    """
    Complete autoencoder (encoder + decoder)
    
    For optional pre-training phase
    Validates latent dimension choice via reconstruction loss
    """
    
    def __init__(self,
                 obs_dim: int = 113,
                 latent_dim: int = 8):
        """
        Args:
            obs_dim: Observation dimension
            latent_dim: Latent dimension to validate
        """
        super().__init__()
        
        self.encoder = LatentEncoder(obs_dim, latent_dim)
        self.decoder = LatentDecoder(latent_dim, obs_dim)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full autoencoder pass
        
        Args:
            obs: (batch, 113) observations
        
        Returns:
            z: (batch, latent_dim) latent encoding
            obs_recon: (batch, 113) reconstruction
        """
        z = self.encoder(obs)
        obs_recon = self.decoder(z)
        return z, obs_recon
    
    def reconstruction_loss(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE reconstruction loss
        
        Args:
            obs: (batch, 113) observations
        
        Returns:
            loss: scalar MSE between obs and reconstruction
        """
        _, obs_recon = self.forward(obs)
        return torch.mean((obs - obs_recon) ** 2)


# Utility functions

def validate_latent_dim(encoder: LatentEncoder,
                       decoder: LatentDecoder,
                       obs_samples: torch.Tensor) -> float:
    """
    Validate latent dimension by reconstruction error
    
    Args:
        encoder: Trained encoder
        decoder: Trained decoder
        obs_samples: (N, 113) observation samples
    
    Returns:
        recon_error: Mean reconstruction error
    """
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        z = encoder(obs_samples)
        obs_recon = decoder(z)
        error = torch.mean((obs_samples - obs_recon) ** 2).item()
    
    return error


if __name__ == "__main__":
    # Smoke test
    print("=" * 60)
    print("Latent Encoder Smoke Test")
    print("=" * 60)
    
    # Create encoder
    encoder = LatentEncoder(obs_dim=113, latent_dim=8)
    print(f"\nEncoder parameters: {sum(p.numel() for p in encoder.parameters())}")
    
    # Test forward pass
    batch_size = 32
    obs = torch.randn(batch_size, 113)
    z = encoder(obs)
    
    print(f"Input shape: {obs.shape}")
    print(f"Output shape: {z.shape}")
    assert z.shape == (batch_size, 8), "Shape mismatch!"
    
    # Test autoencoder
    autoencoder = Autoencoder(obs_dim=113, latent_dim=8)
    z, obs_recon = autoencoder(obs)
    
    print(f"\nAutoencoder reconstruction shape: {obs_recon.shape}")
    print(f"Reconstruction loss (untrained): {autoencoder.reconstruction_loss(obs).item():.4f}")
    
    print("\n✅ Encoder smoke test passed!")
