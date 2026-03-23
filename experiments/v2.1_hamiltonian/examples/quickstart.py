"""
Quickstart Example: Training Hamiltonian PPO

This example demonstrates how to use HamiltonianActorCriticPolicy
with Stable-Baselines3 PPO on the v2.0 MHD environment.
"""

import sys
import os

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../v2.0'))

from sb3_policy import HamiltonianActorCriticPolicy
from mhd_elsasser_env import MHDElsasserEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

print("=" * 70)
print("Quickstart: Hamiltonian PPO Training")
print("=" * 70)

# Step 1: Create environment
print("\n1. Creating MHD Elsasser environment...")
env = DummyVecEnv([lambda: MHDElsasserEnv()])
print("   ✅ Environment created")
print(f"   Observation space: {env.observation_space}")
print(f"   Action space: {env.action_space}")

# Step 2: Create PPO model with Hamiltonian policy
print("\n2. Creating PPO with HamiltonianPolicy...")

# Configuration
lambda_h = 0.5  # Hamiltonian guidance strength (0 = baseline PPO, 1.0 = strong guidance)
latent_dim = 8  # Latent space dimension
h_hidden_dim = 64  # H-network hidden layer size

model = PPO(
    HamiltonianActorCriticPolicy,
    env,
    policy_kwargs=dict(
        lambda_h=lambda_h,
        latent_dim=latent_dim,
        h_hidden_dim=h_hidden_dim
    ),
    learning_rate=3e-4,
    n_steps=128,
    batch_size=64,
    n_epochs=10,
    verbose=1
)

print("   ✅ Model created")
print(f"   Lambda_H: {lambda_h} (guidance strength)")
print(f"   Latent dim: {latent_dim}")
print(f"   H hidden dims: ({h_hidden_dim}, {h_hidden_dim})")

# Step 3: Train
print("\n3. Training for 1000 steps (quick demo)...")
model.learn(total_timesteps=1000, progress_bar=True)
print("   ✅ Training complete")

# Step 4: Evaluate
print("\n4. Evaluating trained policy...")
obs = env.reset()
episode_reward = 0
done = False
steps = 0

while not done and steps < 200:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    episode_reward += reward[0]
    steps += 1
    done = done[0]

print(f"   ✅ Evaluation complete")
print(f"   Episode reward: {episode_reward:.2f}")
print(f"   Episode length: {steps} steps")

# Step 5: Save model (optional)
print("\n5. Saving model...")
model.save("hamiltonian_ppo_demo")
print("   ✅ Model saved to hamiltonian_ppo_demo.zip")

print("\n" + "=" * 70)
print("Quickstart complete!")
print("=" * 70)

print("\nNext steps:")
print("  - Adjust lambda_h (try 0.0, 0.1, 0.5, 1.0) to see guidance effect")
print("  - Train longer (100k steps) for better performance")
print("  - Compare with baseline (lambda_h=0.0)")
print("  - See scripts/train_hamiltonian_variants.py for production training")
