"""
Latent Space HNN with Synthetic Data
Based on Greydanus 2019 Section 5 concept

Goal: Verify HNN conservation in learned latent space
Uses synthetic "pseudo-pixels" instead of real Gym renders
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from typing import Tuple, Dict


# ============================================================================
# Synthetic Dataset (Pendulum physics → pseudo-pixels)
# ============================================================================

class PendulumPhysics:
    """Ground truth pendulum physics for synthetic data"""
    def __init__(self, m=1.0, l=1.0, g=3.0):
        self.m = m
        self.l = l
        self.g = g
    
    def hamiltonian(self, q: float, p: float) -> float:
        PE = 2 * self.m * self.g * self.l * (1 - np.cos(q))
        KE = (self.l**2 * p**2) / (2 * self.m)
        return PE + KE
    
    def dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        q, p = state
        dq_dt = (self.l**2 * p) / self.m
        dp_dt = -2 * self.m * self.g * self.l * np.sin(q)
        return np.array([dq_dt, dp_dt])


def state_to_image(q: float, p: float, img_size: int = 16) -> np.ndarray:
    """
    Convert (q, p) to pseudo-pixel representation
    
    Simple encoding:
    - Image is img_size x img_size
    - Value = function of distance to (q, p) in normalized space
    - This creates a unique pattern for each (q, p)
    """
    img = np.zeros((img_size, img_size))
    
    # Normalize q, p to [0, 1]
    q_norm = (np.sin(q) + 1) / 2  # q ∈ [-π, π] → [0, 1]
    p_norm = (np.tanh(p) + 1) / 2  # p unbounded → [0, 1]
    
    # Create pattern centered at (q_norm, p_norm) * img_size
    center_i = int(q_norm * (img_size - 1))
    center_j = int(p_norm * (img_size - 1))
    
    for i in range(img_size):
        for j in range(img_size):
            dist = np.sqrt((i - center_i)**2 + (j - center_j)**2)
            img[i, j] = np.exp(-dist**2 / 8)
    
    return img


def generate_synthetic_dataset(n_trajectories: int = 100,
                               n_points: int = 50,
                               img_size: int = 16,
                               seed: int = 42) -> Dict:
    """
    Generate synthetic "pixel" dataset
    
    Physics: (q, p) → evolve → (q', p')
    Pseudo-pixels: state_to_image(q, p)
    
    Returns pairs: (img_t, img_t+1) with ground truth (q,p)
    """
    np.random.seed(seed)
    pendulum = PendulumPhysics()
    
    all_imgs_t = []
    all_imgs_t1 = []
    all_states_t = []
    all_states_t1 = []
    
    for _ in range(n_trajectories):
        # Random initial condition
        q0 = np.random.uniform(-np.pi/2, np.pi/2)
        p0 = np.random.uniform(-1, 1)
        
        # Generate trajectory
        sol = solve_ivp(
            pendulum.dynamics,
            t_span=(0, 3),
            y0=[q0, p0],
            t_eval=np.linspace(0, 3, n_points),
            method='RK45',
            rtol=1e-9,
            atol=1e-9
        )
        
        q_traj = sol.y[0]
        p_traj = sol.y[1]
        
        # Convert to images
        for i in range(n_points - 1):
            img_t = state_to_image(q_traj[i], p_traj[i], img_size)
            img_t1 = state_to_image(q_traj[i+1], p_traj[i+1], img_size)
            
            all_imgs_t.append(img_t)
            all_imgs_t1.append(img_t1)
            all_states_t.append([q_traj[i], p_traj[i]])
            all_states_t1.append([q_traj[i+1], p_traj[i+1]])
    
    # Stack and add noise
    imgs_t = np.stack(all_imgs_t)
    imgs_t1 = np.stack(all_imgs_t1)
    
    # Add small noise to images
    imgs_t += np.random.normal(0, 0.01, imgs_t.shape)
    imgs_t1 += np.random.normal(0, 0.01, imgs_t1.shape)
    
    # Clip to [0, 1]
    imgs_t = np.clip(imgs_t, 0, 1)
    imgs_t1 = np.clip(imgs_t1, 0, 1)
    
    # Concatenate images (like paper's 2-frame input)
    img_pairs = np.stack([imgs_t, imgs_t1], axis=-1)  # (N, 16, 16, 2)
    
    states_t = np.array(all_states_t)
    states_t1 = np.array(all_states_t1)
    
    return {
        'images': img_pairs,
        'states_t': states_t,
        'states_t1': states_t1
    }


# ============================================================================
# Networks (same as pixel version)
# ============================================================================

class Autoencoder(nn.Module):
    """Autoencoder: images → latent z (2D)"""
    def __init__(self, img_size: int = 16, latent_dim: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.img_size = img_size
        self.input_dim = img_size * img_size * 2
        
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        recon = self.decoder(z)
        return recon.view(-1, self.img_size, self.img_size, 2)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        return z, recon


class LatentHNN(nn.Module):
    """HNN in latent space"""
    def __init__(self, latent_dim: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
    
    def time_derivative(self, z: torch.Tensor) -> torch.Tensor:
        z = z.requires_grad_(True)
        H = self.forward(z)
        
        dH = torch.autograd.grad(H.sum(), z, create_graph=True, retain_graph=True)[0]
        
        # Symplectic gradient: (∂H/∂z_p, -∂H/∂z_q)
        half = z.shape[1] // 2
        dz_q_dt = dH[:, half:]  # ∂H/∂z_p
        dz_p_dt = -dH[:, :half]  # -∂H/∂z_q
        
        return torch.cat([dz_q_dt, dz_p_dt], dim=1)


# ============================================================================
# Training
# ============================================================================

def compute_auxiliary_loss(z_t: torch.Tensor, z_t1: torch.Tensor) -> torch.Tensor:
    """Auxiliary loss: encourage z_p ≈ ∂z_q/∂t"""
    half = z_t.shape[1] // 2
    z_q_t = z_t[:, :half]
    z_p_t = z_t[:, half:]
    z_q_t1 = z_t1[:, :half]
    
    z_q_diff = z_q_t - z_q_t1
    return torch.mean((z_p_t - z_q_diff)**2)


def train_latent_hnn(autoencoder: Autoencoder,
                     hnn: LatentHNN,
                     dataset: Dict,
                     n_epochs: int = 3000,
                     batch_size: int = 100,
                     lr: float = 1e-3) -> Dict:
    """Train autoencoder + HNN jointly"""
    
    params = list(autoencoder.parameters()) + list(hnn.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    
    images = torch.FloatTensor(dataset['images'])
    n_samples = len(images)
    
    losses = []
    
    for epoch in range(n_epochs):
        indices = torch.randperm(n_samples)
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_imgs = images[batch_idx]
            
            optimizer.zero_grad()
            
            # Encode
            z, recon = autoencoder(batch_imgs)
            
            # Loss 1: Reconstruction
            loss_recon = torch.mean((recon - batch_imgs)**2)
            
            # Loss 2: Auxiliary (canonical coordinates)
            if len(z) > 1:
                z_pairs_t = z[:-1]
                z_pairs_t1 = z[1:]
                loss_aux = compute_auxiliary_loss(z_pairs_t, z_pairs_t1)
            else:
                loss_aux = torch.tensor(0.0)
            
            # Loss 3: HNN dynamics (optional, for better training)
            # Predict dz/dt and compare with finite difference
            dz_dt_pred = hnn.time_derivative(z)
            if len(z) > 1:
                dz_actual = z[1:] - z[:-1]
                loss_hnn = torch.mean((dz_dt_pred[:-1] - dz_actual)**2)
            else:
                loss_hnn = torch.tensor(0.0)
            
            # Total loss
            loss = loss_recon + 0.1 * loss_aux + 0.05 * loss_hnn
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")
    
    return {'losses': losses}


# ============================================================================
# Evaluation: Conservation in latent space
# ============================================================================

def test_latent_conservation(autoencoder: Autoencoder,
                             hnn: LatentHNN,
                             q0: float = 0.5,
                             p0: float = 1.0,
                             n_steps: int = 100,
                             img_size: int = 16) -> Dict:
    """
    Test if HNN conserves H in latent space
    
    Process:
    1. Start with (q0, p0)
    2. Convert to image
    3. Encode to latent z
    4. Integrate HNN dynamics in latent space (RK4)
    5. Check H(z) conservation
    """
    autoencoder.eval()
    hnn.eval()
    
    # Initial image
    img0 = state_to_image(q0, p0, img_size)
    img_pair = np.stack([img0, img0], axis=-1)  # Dummy pair
    img_tensor = torch.FloatTensor(img_pair).unsqueeze(0)
    
    # Encode to latent
    with torch.no_grad():
        z0, _ = autoencoder(img_tensor)
        z0 = z0[0].numpy()
    
    # Define HNN dynamics for scipy
    def hnn_dynamics(t, z_np):
        z_t = torch.FloatTensor([z_np])
        dz = hnn.time_derivative(z_t).detach().numpy()[0]
        return dz
    
    # Integrate in latent space using RK4
    sol = solve_ivp(
        hnn_dynamics,
        t_span=(0, 10),
        y0=z0,
        method='RK45',
        rtol=1e-9,
        atol=1e-9,
        dense_output=True
    )
    
    t_eval = np.linspace(0, 10, n_steps + 1)
    z_traj = sol.sol(t_eval).T
    
    # Compute H(z) along trajectory
    H_values = []
    for z_np in z_traj:
        z_t = torch.FloatTensor([z_np])
        with torch.no_grad():
            H = hnn(z_t).item()
        H_values.append(H)
    
    H_values = np.array(H_values)
    drift = np.max(np.abs(H_values - H_values[0])) / np.abs(H_values[0]) * 100
    
    return {
        'z_trajectory': z_traj,
        'H_values': H_values,
        'drift_percent': drift
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Latent Space HNN (Synthetic Data) - Step 2.2")
    print("=" * 60)
    
    # Generate synthetic dataset
    print("\nGenerating synthetic dataset...")
    dataset = generate_synthetic_dataset(
        n_trajectories=100,
        n_points=50,
        img_size=16,
        seed=42
    )
    print(f"Dataset: {len(dataset['images'])} image pairs")
    print(f"Image shape: {dataset['images'][0].shape}")
    
    # Initialize models
    print("\nInitializing models...")
    autoencoder = Autoencoder(img_size=16, latent_dim=2, hidden_dim=128)
    hnn = LatentHNN(latent_dim=2, hidden_dim=128)
    
    total_params = sum(p.numel() for p in autoencoder.parameters()) + \
                   sum(p.numel() for p in hnn.parameters())
    print(f"Total parameters: {total_params}")
    
    # Train
    print("\nTraining (3000 epochs)...")
    history = train_latent_hnn(
        autoencoder, hnn, dataset,
        n_epochs=3000,
        batch_size=100,
        lr=1e-3
    )
    print(f"Final loss: {history['losses'][-1]:.6f}")
    
    # Test conservation
    print("\n" + "=" * 60)
    print("Testing Conservation in Latent Space")
    print("=" * 60)
    
    result = test_latent_conservation(
        autoencoder, hnn,
        q0=0.5, p0=1.0,
        n_steps=100,
        img_size=16
    )
    
    print(f"\nLatent space conservation:")
    print(f"H(t=0) = {result['H_values'][0]:.8f}")
    print(f"H(t=10) = {result['H_values'][-1]:.8f}")
    print(f"H diff = {result['H_values'][-1] - result['H_values'][0]:.8f}")
    print(f"Drift = {result['drift_percent']:.6f}%")
    
    # Verdict
    print("\n" + "=" * 60)
    print("Step 2.2 Result")
    print("=" * 60)
    
    if result['drift_percent'] < 0.1:
        print("✅ EXCELLENT: <0.1% drift in latent space")
    elif result['drift_percent'] < 1.0:
        print("✅ GOOD: <1% drift in latent space")
    elif result['drift_percent'] < 10:
        print("⚠️  OK: <10% drift (better than baseline)")
    else:
        print(f"❌ HIGH DRIFT: {result['drift_percent']:.2f}%")
    
    print("\nStep 2.2 complete!")
