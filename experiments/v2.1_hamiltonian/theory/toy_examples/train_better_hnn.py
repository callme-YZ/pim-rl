"""Train HNN with more data and epochs"""
import torch
import numpy as np
from pendulum_basic_hnn import HamiltonianNN, generate_dataset, train_model

print("Training with MORE data and epochs...")
print("Dataset: 100 trajectories (was 25)")
print("Epochs: 5000 (was 2000)\n")

# More data
train_states, train_derivs = generate_dataset(n_trajectories=100, seed=42)
print(f"Dataset size: {train_states.shape[0]} samples\n")

# Train longer
hnn = HamiltonianNN()
history = train_model(hnn, train_states, train_derivs, is_hnn=True, n_epochs=5000, lr=1e-3)
print(f"\nFinal loss: {history['losses'][-1]:.6f}\n")

# Test conservation
q, p = 0.5, 1.0
dt = 0.1
n_steps = 100

H_values = []
for step in range(n_steps + 1):
    state = torch.FloatTensor([[q, p]])
    with torch.no_grad():
        H = hnn(state).item()
    H_values.append(H)
    
    if step < n_steps:
        derivs = hnn.time_derivative(state).detach().numpy()[0]
        q += dt * derivs[0]
        p += dt * derivs[1]

H_values = np.array(H_values)
drift = np.max(np.abs(H_values - H_values[0])) / np.abs(H_values[0]) * 100

print(f"Conservation test:")
print(f"H(0) = {H_values[0]:.4f}")
print(f"H(100) = {H_values[-1]:.4f}")
print(f"Drift = {drift:.4f}%\n")

if drift < 0.01:
    print("✅ PASS")
else:
    print(f"❌ FAIL (need < 0.01%, got {drift:.4f}%)")
