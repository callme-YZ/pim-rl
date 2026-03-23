"""Test with Symplectic Euler instead of standard Euler"""
import torch
import numpy as np
from pendulum_basic_hnn import HamiltonianNN, generate_dataset, train_model

# Train HNN
print("Training HNN (100 traj, 5000 epochs)...")
train_states, train_derivs = generate_dataset(n_trajectories=100, seed=42)
hnn = HamiltonianNN()
train_model(hnn, train_states, train_derivs, is_hnn=True, n_epochs=5000, lr=1e-3)
print("Training complete\n")

# Test with SYMPLECTIC Euler
q, p = 0.5, 1.0
dt = 0.1
n_steps = 100

H_values = []

print("Integrating with Symplectic Euler...")
for step in range(n_steps + 1):
    state = torch.FloatTensor([[q, p]])
    
    with torch.no_grad():
        H = hnn(state).item()
    H_values.append(H)
    
    if step < n_steps:
        # Symplectic Euler (semi-implicit):
        # p_{n+1} = p_n + dt * f(q_n)
        # q_{n+1} = q_n + dt * g(p_{n+1})  <- use NEW p!
        
        # Get dp/dt using OLD q
        state_q = torch.FloatTensor([[q, p]])
        derivs = hnn.time_derivative(state_q).detach().numpy()[0]
        dq_dt, dp_dt = derivs
        
        # Update p first
        p_new = p + dt * dp_dt
        
        # Update q using NEW p
        state_p = torch.FloatTensor([[q, p_new]])
        derivs_new = hnn.time_derivative(state_p).detach().numpy()[0]
        dq_dt_new = derivs_new[0]
        
        q_new = q + dt * dq_dt_new
        
        q, p = q_new, p_new

H_values = np.array(H_values)
drift = np.max(np.abs(H_values - H_values[0])) / np.abs(H_values[0]) * 100

print(f"\nSymplectic Euler results:")
print(f"H(0) = {H_values[0]:.4f}")
print(f"H(100) = {H_values[-1]:.4f}")
print(f"Drift = {drift:.4f}%\n")

if drift < 0.01:
    print("✅ PASS")
elif drift < 1.0:
    print(f"⚠️  Better but not <0.01%")
else:
    print(f"❌ FAIL")
