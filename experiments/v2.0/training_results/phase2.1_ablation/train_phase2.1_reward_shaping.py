"""
Phase 2.1: Physics-Informed Reward Shaping

Multi-objective reward to improve sample efficiency.

Reward components:
- Primary: -|m=2 amplitude| (mode suppression)
- Energy: -λ_E × |ΔE/E₀| (conservation)
- Helicity: -λ_H × |ΔH/H₀| (conservation)
- Control effort: -λ_RMP × |I_RMP|² (minimize actuation)

Gate: Compare vs baseline (single-objective)

Author: 小A 🤖
Date: 2026-03-21
"""

import numpy as np
import os
import multiprocessing as mp

mp.set_start_method('spawn', force=True)

from mhd_elsasser_env import MHDElsasserEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import gymnasium as gym
import time


def make_env(rank, seed=0, reward_weights=None):
    """
    Factory for parallel envs with configurable reward weights.
    
    reward_weights: dict with keys 'energy', 'helicity', 'rmp' (optional)
    """
    def _init():
        env = MHDElsasserEnv(
            grid_shape=(16, 32, 16),
            max_episode_steps=100
        )
        
        # Wrap with multi-objective reward if weights provided
        if reward_weights is not None:
            env = MultiObjectiveRewardWrapper(env, reward_weights)
        
        env.reset(seed=seed + rank)
        return env
    return _init


class MultiObjectiveRewardWrapper(gym.Wrapper):
    """
    Wrapper to add physics-informed terms to reward.
    
    Reward = -|A| - λ_E×|ΔE/E₀| - λ_H×|ΔH/H₀| - λ_RMP×|I|²
    """
    
    def __init__(self, env, weights):
        super().__init__(env)
        self.weights = weights
        
        # Reference values (set on first step)
        self.E0 = None
        self.H0 = None
        
        print(f"Multi-objective reward wrapper initialized:")
        print(f"  λ_energy: {weights.get('energy', 0.0)}")
        print(f"  λ_helicity: {weights.get('helicity', 0.0)}")
        print(f"  λ_rmp: {weights.get('rmp', 0.0)}")
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Set reference values
        self.E0 = info.get('energy', 1.0)
        self.H0 = info.get('helicity', 1.0) if 'helicity' in info else 1.0
        
        return obs, info
    
    def step(self, action):
        obs, reward_base, done, trunc, info = self.env.step(action)
        
        # Compute additional reward terms
        reward_total = reward_base  # Start with base -|A|
        
        # Energy conservation penalty
        if 'energy' in self.weights and self.weights['energy'] > 0:
            E = info.get('energy', self.E0)
            energy_drift = abs(E - self.E0) / abs(self.E0) if abs(self.E0) > 1e-10 else 0
            reward_total -= self.weights['energy'] * energy_drift
        
        # Helicity conservation penalty (if available)
        if 'helicity' in self.weights and self.weights['helicity'] > 0:
            H = info.get('helicity', self.H0)
            helicity_drift = abs(H - self.H0) / abs(self.H0) if abs(self.H0) > 1e-10 else 0
            reward_total -= self.weights['helicity'] * helicity_drift
        
        # Control effort penalty
        if 'rmp' in self.weights and self.weights['rmp'] > 0:
            control_effort = np.sum(action**2)
            reward_total -= self.weights['rmp'] * control_effort
        
        # Update info with reward breakdown
        info['reward_base'] = reward_base
        info['reward_total'] = reward_total
        
        return obs, reward_total, done, trunc, info
    
    # gym.Wrapper handles __getattr__ automatically


def train_variant(variant_name, weights, seed=200, total_timesteps=100_000):
    """
    Train single reward variant.
    
    Args:
        variant_name: e.g., 'baseline', 'energy_0.1', 'multi_0.1_0.05_0.01'
        weights: dict or None (baseline)
        seed: random seed
        total_timesteps: training budget
    """
    
    print('\n' + '='*60)
    print(f'Training: {variant_name}')
    print('='*60)
    
    N_ENVS = 8
    LOG_DIR = f'./logs/phase2.1/{variant_name}/'
    
    # Create envs
    print(f'Creating {N_ENVS} parallel environments...')
    env = SubprocVecEnv([make_env(i, seed, weights) for i in range(N_ENVS)])
    eval_env = SubprocVecEnv([make_env(N_ENVS + i, seed, weights) for i in range(3)])
    print('✅ Envs created')
    
    # PPO model
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        n_steps=512 // N_ENVS,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log=LOG_DIR,
        seed=seed
    )
    
    # Callbacks
    os.makedirs(LOG_DIR, exist_ok=True)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=LOG_DIR,
        log_path=LOG_DIR,
        eval_freq=5_000 // N_ENVS,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1
    )
    
    # Training
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        
        print(f'\n✅ {variant_name} complete ({training_time/60:.1f} min)')
        model.save(os.path.join(LOG_DIR, f'final_{variant_name}.zip'))
        
    except Exception as e:
        print(f'\n❌ {variant_name} failed: {e}')
    
    env.close()
    eval_env.close()


def main():
    """
    Run Phase 2.1 reward shaping experiment.
    
    Variants:
    1. Baseline (no additional terms)
    2. Energy only (λ_E = 0.1, 1.0, 10.0)
    3. Multi-objective (λ_E=0.1, λ_H=0.05, λ_RMP=0.01)
    """
    
    print('='*60)
    print('PHASE 2.1: PHYSICS-INFORMED REWARD SHAPING')
    print('='*60)
    
    variants = [
        # Baseline
        ('baseline', None),
        
        # Energy conservation variants
        ('energy_0.1', {'energy': 0.1}),
        ('energy_1.0', {'energy': 1.0}),
        ('energy_10.0', {'energy': 10.0}),
        
        # Multi-objective
        ('multi', {'energy': 0.1, 'helicity': 0.05, 'rmp': 0.01}),
    ]
    
    print(f'\nPlanned variants: {len(variants)}')
    for name, weights in variants:
        print(f'  - {name}: {weights}')
    
    print(f'\nTotal training: {len(variants)} × 100k = {len(variants)*100}k steps')
    print(f'Expected time: ~{len(variants)*20} min (sequential, 8-core)')
    
    print('\n' + '='*60)
    print('Starting automated training...')
    print('='*60)
    
    # Train all variants sequentially
    for variant_name, weights in variants:
        train_variant(variant_name, weights, seed=200 + hash(variant_name) % 100, total_timesteps=100_000)
    
    # Analysis
    print('\n' + '='*60)
    print('PHASE 2.1 ANALYSIS')
    print('='*60)
    print('\nComparing all variants...')
    
    results = []
    for variant_name, _ in variants:
        try:
            data = np.load(f'./logs/phase2.1/{variant_name}/evaluations.npz')
            final_reward = data['results'][-1].mean()
            final_length = data['ep_lengths'][-1].mean()
            results.append((variant_name, final_reward, final_length))
        except:
            results.append((variant_name, None, None))
    
    print('\nVariant         | Final Reward | Episode Length')
    print('----------------|--------------|----------------')
    for name, reward, length in results:
        if reward is not None:
            print(f'{name:15s} | {reward:12.2f} | {length:14.1f}')
        else:
            print(f'{name:15s} | {"N/A":12s} | {"N/A":14s}')
    
    # Best variant
    valid_results = [(n, r, l) for n, r, l in results if r is not None]
    if valid_results:
        best_variant = max(valid_results, key=lambda x: x[1])
        print(f'\n✅ Best variant: {best_variant[0]}')
        print(f'   Reward: {best_variant[1]:.2f}')
        print(f'   Length: {best_variant[2]:.1f} steps')
    
    print('\n' + '='*60)
    print('Phase 2.1 Complete')
    print('='*60)


if __name__ == '__main__':
    main()
