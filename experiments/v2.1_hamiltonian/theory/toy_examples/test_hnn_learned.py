"""Quick test: Check if HNN learned correct Hamiltonian"""
import torch
import numpy as np
from pendulum_basic_hnn import IdealPendulum, HamiltonianNN, generate_dataset, train_model

# Train HNN
print("Training HNN...")
train_states, train_derivs = generate_dataset(n_trajectories=25, seed=42)
hnn = HamiltonianNN()
train_model(hnn, train_states, train_derivs, is_hnn=True, n_epochs=2000, lr=1e-3)

# Test on a point
pendulum = IdealPendulum()
q, p = 0.5, 1.0

# True Hamiltonian
H_true = pendulum.hamiltonian(q, p)

# Learned Hamiltonian
state = torch.FloatTensor([[q, p]])
with torch.no_grad():
    H_learned = hnn(state).item()

print(f"\nH_true = {H_true:.4f}")
print(f"H_learned = {H_learned:.4f}")
print(f"Ratio = {H_learned/H_true:.4f}")

# Test symplectic gradient
state_grad = torch.FloatTensor([[q, p]]).requires_grad_(True)
H = hnn(state_grad)
dH = torch.autograd.grad(H, state_grad, create_graph=True)[0]

# True derivatives
dq_dt_true, dp_dt_true = pendulum.dynamics(0, [q, p])

# Learned derivatives
dq_dt_learned = dH[0, 1].item()  # ∂H/∂p
dp_dt_learned = -dH[0, 0].item()  # -∂H/∂q

print(f"\ndq/dt: true={dq_dt_true:.4f}, learned={dq_dt_learned:.4f}")
print(f"dp/dt: true={dp_dt_true:.4f}, learned={dp_dt_learned:.4f}")
