#!/usr/bin/env python3
"""
M4 Baseline Training - Toroidal MHD Environment

Uses ToroidalMHDEnv with real physics (ToroidalMHDSolver).

**Training Configuration:**
- Algorithm: PPO
- Parallel envs: 8 (multi-core)
- Total timesteps: 10k (smoke test)
- Physics constraints: ∇·B < 1e-6

**交付要求:**
- ✅ 能训练 (不卡死)
- ✅ Physics constraints 验证
- ⚠️ 控制效果不要求好

Usage:
    python examples/m4_baseline_training.py [--timesteps N] [--n-envs N]

Author: 小A 🤖
Date: 2026-03-17
Phase: M4.3
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from pytokmhd.rl import ToroidalMHDEnv


def make_env(rank: int = 0):
    """
    Environment factory for multiprocessing.
    
    Parameters
    ----------
    rank : int
        Process rank (for seed)
    
    Returns
    -------
    callable
        Environment factory
    """
    def _init():
        env = ToroidalMHDEnv(
            grid_size=32,
            dt=0.01,
            eta_base=1e-5,
            nu_base=1e-4,
            max_steps=200
        )
        env.reset(seed=42 + rank)
        return env
    return _init


def train_m4_baseline(
    total_timesteps: int = 10000,
    n_envs: int = 8,
    learning_rate: float = 3e-4,
    batch_size: int = 256,
    gamma: float = 0.99,
    save_path: str = 'models/m4_baseline_10k.zip'
):
    """
    Train PPO on ToroidalMHDEnv.
    
    Parameters
    ----------
    total_timesteps : int
        Total training steps
    n_envs : int
        Number of parallel environments
    learning_rate : float
        PPO learning rate
    batch_size : int
        Minibatch size
    gamma : float
        Discount factor
    save_path : str
        Model save path
    """
    
    print("=" * 70)
    print("M4 Phase 3 - Baseline Training (ToroidalMHDEnv)")
    print("=" * 70)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel envs: {n_envs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Gamma: {gamma}")
    print("=" * 70)
    
    # Create vectorized environment
    if n_envs == 1:
        env = DummyVecEnv([make_env(0)])
        print("Using single-process environment")
    else:
        env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
        print(f"Using {n_envs}-core parallel training")
    
    # Create PPO model
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gamma=gamma,
        verbose=1,
        tensorboard_log='./logs/m4_ppo/'
    )
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=5000 // n_envs,  # Adjust for parallel envs
        save_path='./checkpoints/m4_baseline/',
        name_prefix='m4_ppo'
    )
    
    # Train
    print("\n[Training] Starting PPO...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save
    model.save(save_path)
    print(f"\n✅ Model saved: {save_path}")
    
    # Evaluate
    print("\n[Evaluation] Testing policy...")
    obs = env.reset()
    episode_rewards = []
    episode_div_B = []
    
    for ep in range(5):
        obs = env.reset()
        episode_reward = 0
        max_div_B = 0
        
        for _ in range(200):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            
            # Track div_B
            if 'div_B' in info[0]:
                max_div_B = max(max_div_B, info[0]['div_B'])
            
            if done[0]:
                break
        
        episode_rewards.append(episode_reward)
        episode_div_B.append(max_div_B)
        print(f"  Episode {ep+1}: reward={episode_reward:.2f}, max_div_B={max_div_B:.2e}")
    
    env.close()
    
    # Verify physics constraints
    mean_reward = np.mean(episode_rewards)
    max_div_B_overall = np.max(episode_div_B)
    
    print("\n" + "=" * 70)
    print("Results:")
    print(f"  Mean reward: {mean_reward:.2f}")
    print(f"  Max ∇·B: {max_div_B_overall:.2e}")
    
    if max_div_B_overall < 1e-6:
        print("  ✅ Physics constraints satisfied (∇·B < 1e-6)")
    else:
        print("  ⚠️ Physics constraints violated (∇·B > 1e-6)")
    
    print("=" * 70)
    print("M4 Phase 3 - RL Integration COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=10000,
                        help='Total timesteps (default: 10000)')
    parser.add_argument('--n-envs', type=int, default=8,
                        help='Parallel environments (default: 8)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')
    
    args = parser.parse_args()
    
    train_m4_baseline(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        learning_rate=args.lr
    )
