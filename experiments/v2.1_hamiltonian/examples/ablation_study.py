"""
Ablation Study Example: Testing Different λ_H Values

This example shows how to run a systematic ablation study
comparing different Hamiltonian guidance strengths.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../v2.0'))

from sb3_policy import HamiltonianActorCriticPolicy
from mhd_elsasser_env import MHDElsasserEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

print("=" * 70)
print("Ablation Study: λ_H Sensitivity Analysis")
print("=" * 70)

# Configurations to test
lambda_h_values = [0.0, 0.1, 0.5, 1.0]
results = {}

for lambda_h in lambda_h_values:
    print(f"\n{'=' * 70}")
    print(f"Testing λ_H = {lambda_h}")
    print('=' * 70)
    
    # Create environment
    env = DummyVecEnv([lambda: MHDElsasserEnv()])
    
    # Create model
    model = PPO(
        HamiltonianActorCriticPolicy,
        env,
        policy_kwargs=dict(lambda_h=lambda_h, latent_dim=8, h_hidden_dim=64),
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        verbose=0
    )
    
    # Train
    print(f"Training for 5000 steps...")
    model.learn(total_timesteps=5000, progress_bar=False)
    
    # Evaluate (5 episodes)
    episode_rewards = []
    for _ in range(5):
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
        
        episode_rewards.append(episode_reward)
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    results[lambda_h] = {
        'mean': mean_reward,
        'std': std_reward
    }
    
    print(f"✅ λ_H={lambda_h}: {mean_reward:.2f} ± {std_reward:.2f}")
    
    env.close()

# Summary
print("\n" + "=" * 70)
print("Ablation Study Results")
print("=" * 70)

print(f"\n{'λ_H':<10} {'Mean Reward':<15} {'Std':<10} {'vs Baseline':<15}")
print("-" * 70)

baseline_reward = results[0.0]['mean']

for lambda_h in lambda_h_values:
    mean = results[lambda_h]['mean']
    std = results[lambda_h]['std']
    diff = mean - baseline_reward
    pct = (diff / abs(baseline_reward)) * 100 if baseline_reward != 0 else 0
    
    print(f"{lambda_h:<10.1f} {mean:<15.2f} {std:<10.2f} {diff:+.2f} ({pct:+.1f}%)")

print("\n" + "=" * 70)
print("Interpretation:")
print("  - λ_H=0.0: Baseline PPO (no Hamiltonian guidance)")
print("  - λ_H>0.0: Physics-guided PPO")
print("  - Positive % = Improvement over baseline")
print("=" * 70)
