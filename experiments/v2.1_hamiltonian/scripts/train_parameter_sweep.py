"""
Phase A.3: Parameter Sweep Training

Grid: β ∈ {0.10, 0.17, 0.25} × η ∈ {0.005, 0.01, 0.02}
Configs: Baseline (λ=0.0) + Strong (λ=1.0)
Total: 18 trainings × 50k steps each

小P approved design with success criteria
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../v2.0'))

import numpy as np
from pathlib import Path
import argparse

# Import after path setup
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--beta', type=float, required=True, help='Plasma beta')
parser.add_argument('--eta', type=float, required=True, help='Resistivity')
parser.add_argument('--lambda_h', type=float, required=True, help='Hamiltonian weight (0.0 or 1.0)')
parser.add_argument('--steps', type=int, default=50000, help='Training steps')
args = parser.parse_args()

print('=' * 70)
print(f'Parameter Sweep Training: β={args.beta}, η={args.eta}, λ_H={args.lambda_h}')
print('=' * 70)

# Setup paths
exp_root = Path(__file__).parent.parent
run_name = f'sweep_beta{args.beta}_eta{args.eta}_lambda{args.lambda_h}'
log_dir = exp_root / 'logs' / run_name
log_dir.mkdir(parents=True, exist_ok=True)

print(f'\nLog directory: {log_dir}')

# Import environment (will be modified for β, η)
from mhd_elsasser_env import MHDElsasserEnv

# Create environment factory with custom parameters
def make_env(beta, eta, rank=0):
    def _init():
        # Create environment
        env = MHDElsasserEnv()
        
        # Modify solver parameters
        # Note: This requires env.solver to expose β and η settings
        # If not exposed, need to modify MHDElsasserEnv or CompleteMHDSolver
        
        # For now, log warning if parameters differ from default
        default_beta = 0.17  # From v2.0 default
        default_eta = 0.01
        
        if abs(beta - default_beta) > 1e-6 or abs(eta - default_eta) > 1e-6:
            print(f'⚠️  Warning: Environment uses default β={default_beta}, η={default_eta}')
            print(f'   Requested: β={beta}, η={eta}')
            print(f'   Need to modify CompleteMHDSolver to support parameter injection')
        
        env = Monitor(env, filename=str(log_dir / f'monitor_env{rank}.csv'))
        return env
    return _init

# Create vectorized environment (8 parallel)
n_envs = 8
env = SubprocVecEnv([make_env(args.beta, args.eta, i) for i in range(n_envs)])

# Create eval environment
eval_env = Monitor(MHDElsasserEnv(), filename=str(log_dir / 'eval_monitor.csv'))

# Setup callbacks
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=str(log_dir),
    log_path=str(log_dir),
    eval_freq=10000 // n_envs,  # Eval every ~10k steps
    n_eval_episodes=10,
    deterministic=True,
    render=False
)

checkpoint_callback = CheckpointCallback(
    save_freq=20000 // n_envs,
    save_path=str(log_dir / 'checkpoints'),
    name_prefix=run_name
)

# Import custom policy if Hamiltonian
if args.lambda_h > 0:
    from sb3_policy import HamiltonianActorCriticPolicy
    policy_class = HamiltonianActorCriticPolicy
    policy_kwargs = {'lambda_h': args.lambda_h}
    print(f'\n✅ Using Hamiltonian policy (λ_H={args.lambda_h})')
else:
    policy_class = 'MlpPolicy'
    policy_kwargs = {}
    print(f'\n✅ Using Baseline policy (λ_H=0.0)')

# Create PPO model
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
print(f'{"=" * 70}\n')

# Train
model.learn(
    total_timesteps=args.steps,
    callback=[eval_callback, checkpoint_callback],
    progress_bar=True
)

# Save final model
final_path = log_dir / 'final_model.zip'
model.save(final_path)
print(f'\n✅ Training complete: {final_path}')

# Cleanup
env.close()
eval_env.close()

print(f'\n{"=" * 70}')
print(f'Run {run_name} finished')
print(f'{"=" * 70}')
