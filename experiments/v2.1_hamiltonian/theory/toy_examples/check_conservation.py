"""Check if HNN conserves its learned H (not necessarily true H)"""
import torch
import numpy as np
from pendulum_basic_hnn import HamiltonianNN, generate_dataset, train_model

# Train
train_states, train_derivs = generate_dataset(n_trajectories=25, seed=42)
hnn = HamiltonianNN()
train_model(hnn, train_states, train_derivs, is_hnn=True, n_epochs=2000, lr=1e-3)

# Integrate and track HNN's own H
q, p = 0.5, 1.0
dt = 0.1
n_steps = 100

H_values = []

for step in range(n_steps + 1):
    state = torch.FloatTensor([[q, p]])
    
    # Get HNN's H value
    with torch.no_grad():
        H = hnn(state).item()
    H_values.append(H)
    
    if step < n_steps:
        # Integrate
        derivs = hnn.time_derivative(state).detach().numpy()[0]
        dq, dp = derivs
        q += dt * dq
        p += dt * dp

H_values = np.array(H_values)
H_drift = np.max(np.abs(H_values - H_values[0])) / np.abs(H_values[0])

print(f"HNN's H conservation:")
print(f"H(t=0) = {H_values[0]:.4f}")
print(f"H(t=10) = {H_values[-1]:.4f}")
print(f"Drift = {H_drift*100:.4f}%")

if H_drift * 100 < 0.01:
    print("✅ HNN conserves its own H!")
else:
    print("❌ HNN does not conserve H")
