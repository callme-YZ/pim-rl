"""
Phase A.3: Parameter Sweep Training (v2 - with env parameter injection)

Properly injects β (as pressure_scale) and η into MHDElsasserEnv
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../v2.0'))

import numpy as np
from pathlib import Path
import argparse

# Parse arguments first
parser = argparse.ArgumentParser()
parser.add_argument('--beta', type=float, required=True, help='Plasma beta (mapped to pressure_scale)')
parser.add_argument('--eta', type=float, required=True, help='Resistivity')
parser.add_argument('--lambda_h', type=float, required=True, help='Hamiltonian weight (0.0 or 1.0)')
parser.add_argument('--steps', type=int, default=50000, help='Training steps')
args = parser.parse_args()

print('=' * 70)
print(f'Parameter Sweep Training')
print('=' * 70)
print(f'β (plasma pressure): {args.beta}')
print(f'η (resistivity): {args.eta}')
print(f'λ_H (Hamiltonian weight): {args.lambda_h}')
print(f'Training steps: {args.steps:,}')
print('=' * 70)

# Import after args (for clean output)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv  # Use DummyVecEnv (no multiprocessing issues)
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from mhd_elsasser_env import MHDElsasserEnv

# Setup paths
exp_root = Path(__file__).parent.parent
run_name = f'sweep_beta{args.beta}_eta{args.eta}_lambda{args.lambda_h}'
log_dir = exp_root / 'logs' / run_name
log_dir.mkdir(parents=True, exist_ok=True)

print(f'\nLog directory: {log_dir}')

# Environment factory with custom β, η
def make_env(beta, eta, rank=0):
    """Create environment with custom parameters
    
    Args:
        beta: Plasma pressure parameter (mapped to pressure_scale via小P formula)
        eta: Resistivity
        rank: Environment ID for parallel
    """
    def _init():
        # Create environment with custom parameters
        # β → pressure_scale mapping (小P approved: pressure_scale = 1.2 * β)
        pressure_scale = 1.2 * beta
        
        env = MHDElsasserEnv(
            grid_shape=(32, 64, 32),
            n_coils=4,
            epsilon=0.323,
            eta=eta,  # Custom resistivity
            pressure_scale=pressure_scale,  # β → pressure via小P mapping
            dt_rl=0.02,
            steps_per_action=5,
            max_episode_steps=200
        )
        
        env = Monitor(env, filename=str(log_dir / f'monitor_env{rank}.csv'))
        return env
    return _init

# Create vectorized environment (8 parallel, using DummyVecEnv to avoid macOS multiprocessing issues)
n_envs = 8
print(f'\nCreating {n_envs} parallel environments (DummyVecEnv)...')
env = DummyVecEnv([make_env(args.beta, args.eta, i) for i in range(n_envs)])

# Create eval environment
print('Creating evaluation environment...')
eval_env = Monitor(
    MHDElsasserEnv(
        epsilon=0.323,
        eta=args.eta,
        pressure_scale=1.2 * args.beta  # 小P mapping
    ),
    filename=str(log_dir / 'eval_monitor.csv')
)

# Setup callbacks
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=str(log_dir),
    log_path=str(log_dir),
    eval_freq=10000 // n_envs,
    n_eval_episodes=10,
    deterministic=True,
    render=False
)

checkpoint_callback = CheckpointCallback(
    save_freq=20000 // n_envs,
    save_path=str(log_dir / 'checkpoints'),
    name_prefix=run_name
)

# Select policy
if args.lambda_h > 0:
    from sb3_policy import HamiltonianActorCriticPolicy
    policy_class = HamiltonianActorCriticPolicy
    policy_kwargs = {'lambda_h': args.lambda_h}
    print(f'\n✅ Hamiltonian policy (λ_H={args.lambda_h})')
else:
    policy_class = 'MlpPolicy'
    policy_kwargs = {}
    print(f'\n✅ Baseline policy (λ_H=0.0)')

# Create PPO model
print('\nInitializing PPO...')
model = PPO(
    policy_class,
    env,
    policy_kwargs=policy_kwargs,
    learning_rate=3e-4,
    n_steps=2048 // n_envs,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log=str(log_dir / 'tensorboard')
)

print(f'\n{"=" * 70}')
print(f'Starting training: {args.steps:,} steps')
print(f'Config: β={args.beta}, η={args.eta}, λ_H={args.lambda_h}')
print(f'{"=" * 70}\n')

# Train
model.learn(
    total_timesteps=args.steps,
    callback=[eval_callback, checkpoint_callback],
    progress_bar=False  # Disable for background runs
)

# Save final model
final_path = log_dir / 'final_model.zip'
model.save(final_path)
print(f'\n✅ Training complete!')
print(f'Final model: {final_path}')

# Cleanup
env.close()
eval_env.close()

print(f'\n{"=" * 70}')
print(f'Run {run_name} finished successfully')
print(f'β={args.beta}, η={args.eta}, λ_H={args.lambda_h}')
print(f'{"=" * 70}')
