"""Test with Leapfrog (Verlet) integrator - 2nd order symplectic"""
import torch
import numpy as np
from pendulum_basic_hnn import HamiltonianNN, generate_dataset, train_model

# Train HNN
print("Training HNN (100 traj, 5000 epochs)...")
train_states, train_derivs = generate_dataset(n_trajectories=100, seed=42)
hnn = HamiltonianNN()
train_model(hnn, train_states, train_derivs, is_hnn=True, n_epochs=5000, lr=1e-3)
print("Training complete\n")

# Test with LEAPFROG (Verlet)
q, p = 0.5, 1.0
dt = 0.1
n_steps = 100

H_values = []

print("Integrating with Leapfrog (Verlet)...")
for step in range(n_steps + 1):
    state = torch.FloatTensor([[q, p]])
    
    with torch.no_grad():
        H = hnn(state).item()
    H_values.append(H)
    
    if step < n_steps:
        # Leapfrog (Verlet) scheme:
        # 1. p_{n+1/2} = p_n + (dt/2) * dp_dt(q_n, p_n)
        # 2. q_{n+1} = q_n + dt * dq_dt(q_n, p_{n+1/2})
        # 3. p_{n+1} = p_{n+1/2} + (dt/2) * dp_dt(q_{n+1}, p_{n+1/2})
        
        # Half-step for p
        state_0 = torch.FloatTensor([[q, p]])
        derivs_0 = hnn.time_derivative(state_0).detach().numpy()[0]
        dq_dt_0, dp_dt_0 = derivs_0
        
        p_half = p + (dt / 2) * dp_dt_0
        
        # Full step for q using p_half
        state_half = torch.FloatTensor([[q, p_half]])
        derivs_half = hnn.time_derivative(state_half).detach().numpy()[0]
        dq_dt_half = derivs_half[0]
        
        q_new = q + dt * dq_dt_half
        
        # Another half-step for p using new q
        state_new = torch.FloatTensor([[q_new, p_half]])
        derivs_new = hnn.time_derivative(state_new).detach().numpy()[0]
        dp_dt_new = derivs_new[1]
        
        p_new = p_half + (dt / 2) * dp_dt_new
        
        q, p = q_new, p_new

H_values = np.array(H_values)
drift = np.max(np.abs(H_values - H_values[0])) / np.abs(H_values[0]) * 100

print(f"\nLeapfrog (Verlet) results:")
print(f"H(0) = {H_values[0]:.4f}")
print(f"H(100) = {H_values[-1]:.4f}")
print(f"Drift = {drift:.4f}%\n")

print("Comparison:")
print("- Standard Euler: 49%")
print("- Symplectic Euler: 15%")
print(f"- Leapfrog: {drift:.2f}%\n")

if drift < 0.01:
    print("✅ PASS: <0.01%")
elif drift < 1.0:
    print(f"✅ GOOD: <1% (acceptable for HNN demo)")
elif drift < 10:
    print(f"⚠️  OK: <10% (better than baseline)")
else:
    print(f"❌ FAIL: Still >10%")
