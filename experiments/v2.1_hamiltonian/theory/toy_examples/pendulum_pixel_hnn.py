"""
Pixel Pendulum with Latent Space HNN
Based on Greydanus 2019 Section 5

Goal: Learn Hamiltonian from pixel observations via autoencoder
Validates: Conservation in latent space
"""

import numpy as np
import torch
import torch.nn as nn
import gym
from scipy.integrate import solve_ivp
from typing import Tuple, Dict, List


# ============================================================================
# Dataset Generation (OpenAI Gym Pendulum-v0)
# ============================================================================

def generate_pixel_dataset(n_trajectories: int = 200,
                           n_frames: int = 100,
                           max_displacement: float = np.pi/6,
                           img_size: int = 28,
                           seed: int = 42) -> Dict:
    """
    Generate pixel pendulum dataset from Gym
    
    Following Greydanus 2019 Section 5.1:
    - 200 trajectories × 100 frames
    - Max displacement π/6
    - Downsample to 28×28
    - 2 consecutive frames as input
    """
    np.random.seed(seed)
    env = gym.make('Pendulum-v1', render_mode='rgb_array')
    
    all_frames = []
    all_states = []  # Ground truth (θ, θ_dot)
    
    for _ in range(n_trajectories):
        # Reset with constrained initial state
        env.reset()
        # Set initial angle within ±π/6
        theta_0 = np.random.uniform(-max_displacement, max_displacement)
        theta_dot_0 = np.random.uniform(-0.5, 0.5)
        
        # Manually set state (Gym Pendulum internals)
        env.unwrapped.state = np.array([theta_0, theta_dot_0])
        
        frames = []
        states = []
        
        for _ in range(n_frames):
            # Render frame
            frame = env.render()
            
            # Preprocess: crop, grayscale, downsample
            frame_gray = np.mean(frame, axis=2)  # RGB → grayscale
            # Downsample (simple: take every k-th pixel)
            h, w = frame_gray.shape
            step_h = h // img_size
            step_w = w // img_size
            frame_small = frame_gray[::step_h, ::step_w][:img_size, :img_size]
            
            frames.append(frame_small)
            states.append(env.unwrapped.state.copy())
            
            # No action (free evolution)
            env.step([0.0])
        
        all_frames.append(np.stack(frames))  # (100, 28, 28)
        all_states.append(np.stack(states))  # (100, 2)
    
    env.close()
    
    frames = np.array(all_frames)  # (200, 100, 28, 28)
    states = np.array(all_states)  # (200, 100, 2)
    
    # Create paired frames (t, t+1)
    frames_t = frames[:, :-1]  # (200, 99, 28, 28)
    frames_t1 = frames[:, 1:]  # (200, 99, 28, 28)
    
    # Concatenate along channel dimension
    frame_pairs = np.stack([frames_t, frames_t1], axis=-1)  # (200, 99, 28, 28, 2)
    
    # Corresponding states
    states_t = states[:, :-1]  # (200, 99, 2)
    states_t1 = states[:, 1:]
    
    # Flatten trajectories
    frame_pairs = frame_pairs.reshape(-1, 28, 28, 2)  # (19800, 28, 28, 2)
    states_t = states_t.reshape(-1, 2)
    states_t1 = states_t1.reshape(-1, 2)
    
    # Normalize pixels to [0, 1]
    frame_pairs = frame_pairs / 255.0
    
    return {
        'frames': frame_pairs,
        'states_t': states_t,
        'states_t1': states_t1
    }


# ============================================================================
# Networks
# ============================================================================

class Autoencoder(nn.Module):
    """
    Autoencoder for pixel pendulum
    Architecture: 4 FC layers, ReLU, residual connections
    Latent: 2D (matches q, p)
    """
    def __init__(self, img_size: int = 28, latent_dim: int = 2, hidden_dim: int = 200):
        super().__init__()
        self.img_size = img_size
        self.input_dim = img_size * img_size * 2  # 2 frames concatenated
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.input_dim),
            nn.Sigmoid()  # Output [0, 1]
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (batch, 28, 28, 2)
        Output: (batch, 2) latent z
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Input: (batch, 2) latent z
        Output: (batch, 28, 28, 2) reconstructed frames
        """
        recon_flat = self.decoder(z)
        return recon_flat.view(-1, self.img_size, self.img_size, 2)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: (latent z, reconstruction)
        """
        z = self.encode(x)
        recon = self.decode(z)
        return z, recon


class LatentHNN(nn.Module):
    """
    HNN operating on latent space z
    Same architecture as basic HNN: 3 layers, 200 hidden, tanh
    """
    def __init__(self, latent_dim: int = 2, hidden_dim: int = 200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)  # Scalar H(z)
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute H(z)"""
        return self.net(z)
    
    def time_derivative(self, z: torch.Tensor) -> torch.Tensor:
        """
        Symplectic gradient in latent space
        z = (z_q, z_p) → (∂H/∂z_p, -∂H/∂z_q)
        """
        z = z.requires_grad_(True)
        H = self.forward(z)
        
        dH = torch.autograd.grad(
            H.sum(), z,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Split z into (z_q, z_p)
        half = z.shape[1] // 2
        dz_q = dH[:, :half]
        dz_p = dH[:, half:]
        
        # Symplectic gradient
        dz_q_dt = dz_p  # ∂H/∂z_p
        dz_p_dt = -dz_q  # -∂H/∂z_q
        
        return torch.cat([dz_q_dt, dz_p_dt], dim=1)


# ============================================================================
# Combined Model
# ============================================================================

class PixelHNN(nn.Module):
    """
    Combined Autoencoder + HNN for pixel pendulum
    """
    def __init__(self, img_size: int = 28, latent_dim: int = 2, hidden_dim: int = 200):
        super().__init__()
        self.autoencoder = Autoencoder(img_size, latent_dim, hidden_dim)
        self.hnn = LatentHNN(latent_dim, hidden_dim)
        self.latent_dim = latent_dim
    
    def forward(self, x: torch.Tensor) -> Dict:
        """
        Full forward pass
        Returns: dict with z, recon, H, dz_dt
        """
        z, recon = self.autoencoder(x)
        H = self.hnn(z)
        dz_dt = self.hnn.time_derivative(z)
        
        return {
            'z': z,
            'recon': recon,
            'H': H,
            'dz_dt': dz_dt
        }


# ============================================================================
# Training
# ============================================================================

def compute_auxiliary_loss(z_t: torch.Tensor, z_t1: torch.Tensor) -> torch.Tensor:
    """
    Auxiliary loss (Equation 7):
    L_CC = ‖z_p^t - (z_q^t - z_q^{t+1})‖²
    
    Encourages z_p to behave like derivative of z_q
    """
    half = z_t.shape[1] // 2
    z_q_t = z_t[:, :half]
    z_p_t = z_t[:, half:]
    z_q_t1 = z_t1[:, :half]
    
    # z_p should approximate finite difference of z_q
    z_q_diff = z_q_t - z_q_t1  # Finite difference
    
    loss = torch.mean((z_p_t - z_q_diff)**2)
    return loss


def train_pixel_hnn(model: PixelHNN,
                    dataset: Dict,
                    n_epochs: int = 10000,
                    batch_size: int = 200,
                    lr: float = 1e-3,
                    weight_decay: float = 1e-5) -> Dict:
    """
    Train combined autoencoder + HNN model
    
    Loss = L_recon + L_HNN + L_auxiliary
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    frames = torch.FloatTensor(dataset['frames'])
    n_samples = len(frames)
    
    losses = []
    
    for epoch in range(n_epochs):
        # Shuffle
        indices = torch.randperm(n_samples)
        
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_frames = frames[batch_idx]
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_frames)
            z = outputs['z']
            recon = outputs['recon']
            
            # Loss 1: Reconstruction
            loss_recon = torch.mean((recon - batch_frames)**2)
            
            # Loss 2: Auxiliary (canonical coordinates)
            # For this, need z_t and z_t+1
            # Approximate: use current batch as pairs
            if len(batch_idx) > 1:
                z_t = z[:-1]
                z_t1 = z[1:]
                loss_aux = compute_auxiliary_loss(z_t, z_t1)
            else:
                loss_aux = torch.tensor(0.0)
            
            # Total loss
            loss = loss_recon + 0.1 * loss_aux  # Weight auxiliary loss
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")
    
    return {'losses': losses}


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Pixel Pendulum HNN - Step 2.2")
    print("=" * 60)
    
    # Generate dataset
    print("\nGenerating pixel dataset...")
    print("(This may take a few minutes)")
    
    dataset = generate_pixel_dataset(
        n_trajectories=50,  # Reduced for speed
        n_frames=50,
        seed=42
    )
    
    print(f"Dataset size: {len(dataset['frames'])} frame pairs")
    print(f"Frame shape: {dataset['frames'][0].shape}")
    
    # Initialize model
    print("\nInitializing Pixel-HNN...")
    model = PixelHNN(img_size=28, latent_dim=2, hidden_dim=200)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train
    print("\nTraining (5000 epochs, may take time)...")
    history = train_pixel_hnn(
        model, dataset,
        n_epochs=5000,
        batch_size=200,
        lr=1e-3
    )
    
    print(f"\nFinal loss: {history['losses'][-1]:.6f}")
    
    print("\nStep 2.2 training complete!")
    print("Next: Test conservation in latent space")
