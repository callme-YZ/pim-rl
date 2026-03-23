"""
1D Pendulum Basic HNN Implementation
Based on Greydanus 2019 Section 3 (Task 2: Ideal Pendulum)

Goal: Verify HNN can learn and conserve Hamiltonian for simple pendulum
Validates: Energy conservation <0.01% over 100 steps
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from typing import Tuple, Dict


# ============================================================================
# Physics: Ideal Pendulum
# ============================================================================

class IdealPendulum:
    """
    Hamiltonian: H = 2mgl(1-cos(q)) + (l^2 * p^2)/(2m)
    
    Parameters (Greydanus 2019):
    - m = 1 (mass)
    - l = 1 (length)
    - g = 3 (gravity, chosen to show nonlinear transition)
    """
    def __init__(self, m=1.0, l=1.0, g=3.0):
        self.m = m
        self.l = l
        self.g = g
    
    def hamiltonian(self, q: float, p: float) -> float:
        """Total energy: H = PE + KE"""
        PE = 2 * self.m * self.g * self.l * (1 - np.cos(q))
        KE = (self.l**2 * p**2) / (2 * self.m)
        return PE + KE
    
    def dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Hamilton's equations:
        dq/dt = ∂H/∂p = (l^2 * p) / m
        dp/dt = -∂H/∂q = -2mgl * sin(q)
        """
        q, p = state
        dq_dt = (self.l**2 * p) / self.m
        dp_dt = -2 * self.m * self.g * self.l * np.sin(q)
        return np.array([dq_dt, dp_dt])
    
    def generate_trajectory(self, q0: float, p0: float, 
                           t_span: Tuple[float, float],
                           n_points: int = 30) -> Dict:
        """Generate ground truth trajectory using RK4 integration"""
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        
        sol = solve_ivp(
            self.dynamics,
            t_span,
            [q0, p0],
            t_eval=t_eval,
            method='RK45',
            rtol=1e-9,
            atol=1e-9
        )
        
        q = sol.y[0]
        p = sol.y[1]
        
        # Compute dq/dt, dp/dt (targets for HNN)
        dq_dt = np.zeros_like(q)
        dp_dt = np.zeros_like(p)
        for i in range(len(q)):
            derivs = self.dynamics(t_eval[i], [q[i], p[i]])
            dq_dt[i] = derivs[0]
            dp_dt[i] = derivs[1]
        
        # Compute energy
        energy = np.array([self.hamiltonian(q[i], p[i]) for i in range(len(q))])
        
        return {
            't': t_eval,
            'q': q,
            'p': p,
            'dq_dt': dq_dt,
            'dp_dt': dp_dt,
            'energy': energy
        }


# ============================================================================
# Dataset Generation (Greydanus 2019 protocol)
# ============================================================================

def generate_dataset(n_trajectories: int = 25,
                    energy_range: Tuple[float, float] = (1.3, 2.3),
                    noise_std: float = 0.1,
                    seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate dataset following Greydanus 2019 Section 3:
    - 25 trajectories
    - Energy in [1.3, 2.3] (linear to nonlinear transition)
    - Gaussian noise σ^2 = 0.1
    - 30 observations per trajectory
    """
    np.random.seed(seed)
    pendulum = IdealPendulum(m=1.0, l=1.0, g=3.0)
    
    all_states = []
    all_derivs = []
    
    for _ in range(n_trajectories):
        # Sample initial energy
        target_energy = np.random.uniform(*energy_range)
        
        # Sample initial q uniformly in [-π/2, π/2]
        q0 = np.random.uniform(-np.pi/2, np.pi/2)
        
        # Compute p0 to match target energy
        PE0 = 2 * pendulum.m * pendulum.g * pendulum.l * (1 - np.cos(q0))
        KE0 = target_energy - PE0
        if KE0 < 0:
            continue
        p0 = np.sqrt(2 * pendulum.m * KE0) / pendulum.l
        p0 *= np.random.choice([-1, 1])
        
        # Generate trajectory
        traj = pendulum.generate_trajectory(q0, p0, t_span=(0, 3), n_points=30)
        
        # Add noise
        q_noisy = traj['q'] + np.random.normal(0, noise_std, size=traj['q'].shape)
        p_noisy = traj['p'] + np.random.normal(0, noise_std, size=traj['p'].shape)
        
        # States and targets
        states = np.stack([q_noisy, p_noisy], axis=1)
        derivs = np.stack([traj['dq_dt'], traj['dp_dt']], axis=1)
        
        all_states.append(states)
        all_derivs.append(derivs)
    
    states = np.concatenate(all_states, axis=0)
    derivs = np.concatenate(all_derivs, axis=0)
    
    return states, derivs


# ============================================================================
# Neural Networks
# ============================================================================

class BaselineNN(nn.Module):
    """Baseline: Direct prediction of (dq/dt, dp/dt)"""
    def __init__(self, input_dim: int = 2, hidden_dim: int = 200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class HamiltonianNN(nn.Module):
    """HNN: Learn scalar Hamiltonian H(q,p)"""
    def __init__(self, input_dim: int = 2, hidden_dim: int = 200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)
    
    def time_derivative(self, state: torch.Tensor) -> torch.Tensor:
        """Compute symplectic gradient: (∂H/∂p, -∂H/∂q)"""
        state = state.requires_grad_(True)
        H = self.forward(state)
        
        dH = torch.autograd.grad(
            H.sum(), state,
            create_graph=True,
            retain_graph=True
        )[0]
        
        dq_dt = dH[:, 1:2]  # ∂H/∂p
        dp_dt = -dH[:, 0:1]  # -∂H/∂q
        
        return torch.cat([dq_dt, dp_dt], dim=1)


# ============================================================================
# Training
# ============================================================================

def train_model(model: nn.Module,
                states: np.ndarray,
                derivs: np.ndarray,
                is_hnn: bool = False,
                n_epochs: int = 2000,
                lr: float = 1e-3) -> Dict:
    """Train model following Greydanus 2019 protocol"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    states_t = torch.FloatTensor(states)
    derivs_t = torch.FloatTensor(derivs)
    
    losses = []
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        if is_hnn:
            pred_derivs = model.time_derivative(states_t)
        else:
            pred_derivs = model(states_t)
        
        loss = torch.mean((pred_derivs - derivs_t)**2)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.6f}")
    
    return {'losses': losses}


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_energy_conservation(model: nn.Module,
                                 is_hnn: bool,
                                 q0: float,
                                 p0: float,
                                 n_steps: int = 100,
                                 dt: float = 0.1) -> Dict:
    """Integrate model dynamics and check energy conservation"""
    model.eval()
    pendulum = IdealPendulum()
    
    q, p = q0, p0
    trajectory = [(q, p)]
    energies = [pendulum.hamiltonian(q, p)]
    
    for _ in range(n_steps):
        state = torch.FloatTensor([[q, p]])
        
        if is_hnn:
            # HNN needs gradients, no no_grad context
            k1 = model.time_derivative(state).detach().numpy()[0]
        else:
            with torch.no_grad():
                k1 = model(state).numpy()[0]
        
        dq, dp = k1
        q = q + dt * dq
        p = p + dt * dp
        
        trajectory.append((q, p))
        energies.append(pendulum.hamiltonian(q, p))
    
    energies = np.array(energies)
    energy_drift = np.max(np.abs(energies - energies[0])) / energies[0]
    
    return {
        'trajectory': np.array(trajectory),
        'energy': energies,
        'energy_drift_percent': energy_drift * 100
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("1D Pendulum Basic HNN - Step 2.1")
    print("=" * 60)
    
    # Generate dataset
    print("\nGenerating dataset...")
    train_states, train_derivs = generate_dataset(
        n_trajectories=25,
        energy_range=(1.3, 2.3),
        noise_std=0.1,
        seed=42
    )
    print(f"Dataset: {train_states.shape[0]} samples\n")
    
    # Initialize models
    print("Initializing models...")
    baseline = BaselineNN(input_dim=2, hidden_dim=200)
    hnn = HamiltonianNN(input_dim=2, hidden_dim=200)
    print(f"Baseline params: {sum(p.numel() for p in baseline.parameters())}")
    print(f"HNN params: {sum(p.numel() for p in hnn.parameters())}\n")
    
    # Train baseline
    print("Training Baseline...")
    baseline_history = train_model(
        baseline, train_states, train_derivs,
        is_hnn=False, n_epochs=2000, lr=1e-3
    )
    print(f"Final loss: {baseline_history['losses'][-1]:.6f}\n")
    
    # Train HNN
    print("Training HNN...")
    hnn_history = train_model(
        hnn, train_states, train_derivs,
        is_hnn=True, n_epochs=2000, lr=1e-3
    )
    print(f"Final loss: {hnn_history['losses'][-1]:.6f}\n")
    
    # Evaluate energy conservation
    print("=" * 60)
    print("Energy Conservation Test")
    print("=" * 60)
    
    q_test, p_test = 0.5, 1.0
    print(f"\nTest IC: q0={q_test}, p0={p_test}")
    print(f"True energy: {IdealPendulum().hamiltonian(q_test, p_test):.4f}\n")
    
    baseline_eval = evaluate_energy_conservation(
        baseline, is_hnn=False, q0=q_test, p0=p_test,
        n_steps=100, dt=0.1
    )
    print(f"Baseline energy drift: {baseline_eval['energy_drift_percent']:.4f}%")
    
    hnn_eval = evaluate_energy_conservation(
        hnn, is_hnn=True, q0=q_test, p0=p_test,
        n_steps=100, dt=0.1
    )
    print(f"HNN energy drift: {hnn_eval['energy_drift_percent']:.4f}%")
    
    # Verification
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)
    target_drift = 0.01
    if hnn_eval['energy_drift_percent'] < target_drift:
        print(f"✅ PASS: Drift {hnn_eval['energy_drift_percent']:.4f}% < {target_drift}%")
    else:
        print(f"❌ FAIL: Drift {hnn_eval['energy_drift_percent']:.4f}% > {target_drift}%")
    
    print("\nStep 2.1 complete!")
