"""
Controlled Hamiltonian System - Step 2.3
Coupled oscillators with control input

Goal: Verify HNN can learn H(state, control) and compute ∂H/∂u
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from typing import Dict, Tuple


# ============================================================================
# Physics: Coupled Pendulums with Control
# ============================================================================

class ControlledOscillators:
    """
    Two coupled pendulums with external control
    
    H(q₁, p₁, q₂, p₂, u) = H₀ + H_control
    
    H₀ = (p₁² + p₂²)/2 + (1-cos(q₁)) + (1-cos(q₂)) + k·q₁·q₂
    H_control = u·sin(q₁)
    
    This models:
    - Natural oscillator dynamics (H₀)
    - Control coupling (H_control)
    - Analogous to MHD + RMP (simplified)
    """
    def __init__(self, k: float = 0.1):
        self.k = k  # coupling strength
    
    def hamiltonian(self, q1: float, p1: float, 
                   q2: float, p2: float, u: float) -> float:
        """Total energy with control"""
        # Kinetic energy
        KE = (p1**2 + p2**2) / 2
        
        # Potential energy (pendulums)
        PE = (1 - np.cos(q1)) + (1 - np.cos(q2))
        
        # Coupling
        coupling = self.k * q1 * q2
        
        # Control term
        control = u * np.sin(q1)
        
        return KE + PE + coupling + control
    
    def dynamics(self, t: float, state: np.ndarray, u: float) -> np.ndarray:
        """
        Hamilton's equations with control
        
        state = [q1, p1, q2, p2]
        """
        q1, p1, q2, p2 = state
        
        # dq/dt = ∂H/∂p
        dq1_dt = p1
        dq2_dt = p2
        
        # dp/dt = -∂H/∂q
        dp1_dt = -np.sin(q1) - self.k * q2 - u * np.cos(q1)
        dp2_dt = -np.sin(q2) - self.k * q1
        
        return np.array([dq1_dt, dp1_dt, dq2_dt, dp2_dt])
    
    def control_gradient(self, q1: float) -> float:
        """
        ∂H/∂u = sin(q₁)
        
        This is the control influence on Hamiltonian
        """
        return np.sin(q1)


# ============================================================================
# Dataset Generation
# ============================================================================

def generate_controlled_dataset(n_trajectories: int = 50,
                                n_points: int = 30,
                                u_range: Tuple[float, float] = (-0.5, 0.5),
                                seed: int = 42) -> Dict:
    """
    Generate dataset with varying control inputs
    
    For each trajectory:
    - Random initial state
    - Random constant control u
    - Evolve system
    - Record (state, u, dstate/dt)
    """
    np.random.seed(seed)
    system = ControlledOscillators(k=0.1)
    
    all_states = []
    all_controls = []
    all_derivs = []
    all_energies = []
    
    for _ in range(n_trajectories):
        # Random initial condition
        q1_0 = np.random.uniform(-np.pi/2, np.pi/2)
        p1_0 = np.random.uniform(-0.5, 0.5)
        q2_0 = np.random.uniform(-np.pi/2, np.pi/2)
        p2_0 = np.random.uniform(-0.5, 0.5)
        
        state0 = [q1_0, p1_0, q2_0, p2_0]
        
        # Random constant control
        u = np.random.uniform(*u_range)
        
        # Generate trajectory
        sol = solve_ivp(
            lambda t, y: system.dynamics(t, y, u),
            t_span=(0, 3),
            y0=state0,
            t_eval=np.linspace(0, 3, n_points),
            method='RK45',
            rtol=1e-9,
            atol=1e-9
        )
        
        states = sol.y.T  # (n_points, 4)
        
        # Compute derivatives
        derivs = np.array([system.dynamics(0, s, u) for s in states])
        
        # Compute energies
        energies = np.array([
            system.hamiltonian(s[0], s[1], s[2], s[3], u)
            for s in states
        ])
        
        # Store
        all_states.append(states)
        all_controls.append(np.full(n_points, u))
        all_derivs.append(derivs)
        all_energies.append(energies)
    
    # Flatten
    states = np.concatenate(all_states)  # (N, 4)
    controls = np.concatenate(all_controls)  # (N,)
    derivs = np.concatenate(all_derivs)  # (N, 4)
    energies = np.concatenate(all_energies)  # (N,)
    
    # Add noise
    states += np.random.normal(0, 0.01, states.shape)
    
    return {
        'states': states,
        'controls': controls,
        'derivs': derivs,
        'energies': energies
    }


# ============================================================================
# Controlled HNN
# ============================================================================

class ControlledHNN(nn.Module):
    """
    HNN with control input
    
    Input: (q₁, p₁, q₂, p₂, u) - state + control
    Output: H(state, u) - scalar Hamiltonian
    
    Dynamics: dstate/dt = symplectic_gradient(H)
    Control gradient: ∂H/∂u
    """
    def __init__(self, state_dim: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = state_dim + 1  # state + control
        
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        """
        Compute H(state, u)
        
        state: (batch, 4)
        control: (batch, 1)
        output: (batch, 1)
        """
        x = torch.cat([state, control], dim=1)
        return self.net(x)
    
    def time_derivative(self, state: torch.Tensor, 
                       control: torch.Tensor) -> torch.Tensor:
        """
        Symplectic gradient: (∂H/∂p, -∂H/∂q)
        
        Returns: dstate/dt (batch, 4)
        """
        state = state.requires_grad_(True)
        control_detached = control.detach()  # Control is parameter, not variable
        
        H = self.forward(state, control_detached)
        
        dH_dstate = torch.autograd.grad(
            H.sum(), state,
            create_graph=True,
            retain_graph=True
        )[0]  # (batch, 4) = (∂H/∂q₁, ∂H/∂p₁, ∂H/∂q₂, ∂H/∂p₂)
        
        # Symplectic structure
        dq1_dt = dH_dstate[:, 1:2]  # ∂H/∂p₁
        dp1_dt = -dH_dstate[:, 0:1]  # -∂H/∂q₁
        dq2_dt = dH_dstate[:, 3:4]  # ∂H/∂p₂
        dp2_dt = -dH_dstate[:, 2:3]  # -∂H/∂q₂
        
        return torch.cat([dq1_dt, dp1_dt, dq2_dt, dp2_dt], dim=1)
    
    def control_gradient(self, state: torch.Tensor,
                        control: torch.Tensor) -> torch.Tensor:
        """
        Compute ∂H/∂u
        
        This shows how control influences Hamiltonian
        """
        control_grad = control.requires_grad_(True)
        state_detached = state.detach()
        
        H = self.forward(state_detached, control_grad)
        
        dH_du = torch.autograd.grad(H.sum(), control_grad)[0]
        
        return dH_du


# ============================================================================
# Training
# ============================================================================

def train_controlled_hnn(model: ControlledHNN,
                        dataset: Dict,
                        n_epochs: int = 2000,
                        batch_size: int = 100,
                        lr: float = 1e-3) -> Dict:
    """Train HNN on controlled system"""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    states = torch.FloatTensor(dataset['states'])
    controls = torch.FloatTensor(dataset['controls']).unsqueeze(1)
    derivs = torch.FloatTensor(dataset['derivs'])
    
    n_samples = len(states)
    losses = []
    
    for epoch in range(n_epochs):
        indices = torch.randperm(n_samples)
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_states = states[batch_idx]
            batch_controls = controls[batch_idx]
            batch_derivs = derivs[batch_idx]
            
            optimizer.zero_grad()
            
            # Predict derivatives via HNN
            pred_derivs = model.time_derivative(batch_states, batch_controls)
            
            # Loss: match derivatives
            loss = torch.mean((pred_derivs - batch_derivs)**2)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        
        if (epoch + 1) % 400 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")
    
    return {'losses': losses}


# ============================================================================
# Evaluation
# ============================================================================

def test_conservation_with_control(model: ControlledHNN,
                                   u_fixed: float = 0.2,
                                   n_steps: int = 100) -> Dict:
    """
    Test: With constant control, HNN should conserve H
    """
    model.eval()
    system = ControlledOscillators(k=0.1)
    
    # Initial state
    state0 = np.array([0.5, 0.5, -0.3, 0.4])
    
    # Integrate HNN dynamics
    def hnn_dynamics(t, state_np):
        state_t = torch.FloatTensor([state_np]).requires_grad_(True)
        u_t = torch.FloatTensor([[u_fixed]])
        dstate = model.time_derivative(state_t, u_t).detach().numpy()[0]
        return dstate
    
    sol = solve_ivp(
        hnn_dynamics,
        t_span=(0, 10),
        y0=state0,
        method='RK45',
        rtol=1e-9,
        atol=1e-9,
        dense_output=True
    )
    
    t_eval = np.linspace(0, 10, n_steps + 1)
    states_traj = sol.sol(t_eval).T
    
    # Compute H along trajectory
    H_values = []
    for state_np in states_traj:
        state_t = torch.FloatTensor([state_np])
        u_t = torch.FloatTensor([[u_fixed]])
        with torch.no_grad():
            H = model(state_t, u_t).item()
        H_values.append(H)
    
    H_values = np.array(H_values)
    drift = np.max(np.abs(H_values - H_values[0])) / np.abs(H_values[0]) * 100
    
    return {
        'trajectory': states_traj,
        'H_values': H_values,
        'drift_percent': drift
    }


def test_control_gradient(model: ControlledHNN,
                         system: ControlledOscillators) -> Dict:
    """
    Test: ∂H/∂u matches theory
    
    Theory: ∂H/∂u = sin(q₁)
    """
    model.eval()
    
    # Test points
    q1_test = np.linspace(-np.pi/2, np.pi/2, 20)
    
    errors = []
    
    for q1 in q1_test:
        state = torch.FloatTensor([[q1, 0.5, 0.0, 0.3]])
        u = torch.FloatTensor([[0.0]]).requires_grad_(True)
        
        # HNN gradient
        dH_du_hnn = model.control_gradient(state, u).item()
        
        # True gradient
        dH_du_true = system.control_gradient(q1)
        
        error = abs(dH_du_hnn - dH_du_true)
        errors.append(error)
    
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    
    return {
        'mean_error': mean_error,
        'max_error': max_error,
        'errors': errors
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Controlled Hamiltonian System (Step 2.3)")
    print("=" * 60)
    
    # Generate dataset
    print("\nGenerating dataset...")
    dataset = generate_controlled_dataset(
        n_trajectories=50,
        n_points=30,
        u_range=(-0.5, 0.5),
        seed=42
    )
    print(f"Dataset: {len(dataset['states'])} samples")
    print(f"Control range: [{dataset['controls'].min():.2f}, {dataset['controls'].max():.2f}]")
    
    # Initialize model
    print("\nInitializing Controlled HNN...")
    model = ControlledHNN(state_dim=4, hidden_dim=128)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train
    print("\nTraining (2000 epochs)...")
    history = train_controlled_hnn(
        model, dataset,
        n_epochs=2000,
        batch_size=100,
        lr=1e-3
    )
    print(f"Final loss: {history['losses'][-1]:.6f}")
    
    # Test conservation
    print("\n" + "=" * 60)
    print("Test 1: Conservation with Constant Control")
    print("=" * 60)
    
    cons_result = test_conservation_with_control(model, u_fixed=0.2, n_steps=100)
    print(f"H(t=0) = {cons_result['H_values'][0]:.6f}")
    print(f"H(t=10) = {cons_result['H_values'][-1]:.6f}")
    print(f"Drift = {cons_result['drift_percent']:.4f}%")
    
    # Test control gradient
    print("\n" + "=" * 60)
    print("Test 2: Control Gradient ∂H/∂u")
    print("=" * 60)
    
    system = ControlledOscillators(k=0.1)
    grad_result = test_control_gradient(model, system)
    print(f"Mean error: {grad_result['mean_error']:.6f}")
    print(f"Max error: {grad_result['max_error']:.6f}")
    
    # Verdict
    print("\n" + "=" * 60)
    print("Step 2.3 Result")
    print("=" * 60)
    
    conservation_pass = cons_result['drift_percent'] < 1.0
    gradient_pass = grad_result['mean_error'] < 0.1
    
    if conservation_pass and gradient_pass:
        print("✅ PASS: Conservation + Control Gradient")
    elif conservation_pass:
        print("⚠️  PARTIAL: Conservation OK, gradient needs work")
    elif gradient_pass:
        print("⚠️  PARTIAL: Gradient OK, conservation needs work")
    else:
        print("❌ FAIL: Both tests need improvement")
    
    print("\nStep 2.3 complete!")
