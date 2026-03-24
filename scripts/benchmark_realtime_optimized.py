#!/usr/bin/env python3
"""
Real-time Performance Benchmark - Optimized

Tests different observation frequencies to find optimal trade-off.

Author: 小A 🤖
Date: 2026-03-24
Issue: #30 Phase 2
"""

import sys
sys.path.insert(0, 'src')

import time
import numpy as np
from pytokmhd.rl.hamiltonian_env import make_hamiltonian_mhd_env
from pytokmhd.rl.classical_controllers import make_baseline_agent
from pim_rl.physics.v2.tearing_ic import create_tearing_ic
import jax.numpy as jnp

print("=" * 60)
print("Real-time Optimization: Observation Frequency Tuning")
print("=" * 60)
print()

# ==============================================================================
# Setup
# ==============================================================================

print("Setting up environment...")
env = make_hamiltonian_mhd_env(
    nr=32, ntheta=64, nz=8,
    dt=1e-4, max_steps=1000,
    eta=0.05, nu=1e-4,
    normalize_obs=False
)

psi, phi = create_tearing_ic(nr=32, ntheta=64)
env.mhd_solver.initialize(jnp.array(psi, dtype=jnp.float32), jnp.array(phi, dtype=jnp.float32))
env.current_step = 0
env.obs_computer.reset()
obs = env.obs_computer.compute_observation(jnp.array(psi, dtype=jnp.float32), jnp.array(phi, dtype=jnp.float32))

env._last_obs = obs
env._last_psi = jnp.array(psi, dtype=jnp.float32)
env._last_phi = jnp.array(phi, dtype=jnp.float32)
env.psi = jnp.array(psi, dtype=jnp.float32)
env.phi = jnp.array(phi, dtype=jnp.float32)

agent = make_baseline_agent('pid', env.action_space, Kp=5.0, Ki=0.5, Kd=0.01, target=0.0, dt=1e-4)

# Warm-up
for _ in range(10):
    _ = agent.act(obs)
    _ = env.step(agent.act(obs), compute_obs=False)

print("✅ Setup complete\n")

# ==============================================================================
# Test Different Observation Frequencies
# ==============================================================================

n_steps = 1000
obs_intervals = [1, 10, 50, 100, 200, 500]  # Steps between full observations

print("Testing different observation frequencies:\n")

results = []

for interval in obs_intervals:
    print(f"Testing obs_interval = {interval} steps...")
    
    # Reset environment
    env.mhd_solver.initialize(jnp.array(psi, dtype=jnp.float32), jnp.array(phi, dtype=jnp.float32))
    env.current_step = 0
    env._last_obs = obs
    env._last_psi = jnp.array(psi, dtype=jnp.float32)
    env._last_phi = jnp.array(phi, dtype=jnp.float32)
    
    # Run benchmark
    step_times = []
    start_total = time.time()
    
    for step in range(n_steps):
        step_start = time.perf_counter()
        action = agent.act(env._last_obs)
        compute_now = (step % interval == 0)
        obs_new, r, term, trunc, info = env.step(action, compute_obs=compute_now)
        step_end = time.perf_counter()
        step_times.append((step_end - step_start) * 1000)
    
    end_total = time.time()
    
    step_times = np.array(step_times)
    total_time = end_total - start_total
    avg_freq = n_steps / total_time
    
    results.append({
        'interval': interval,
        'total_time': total_time,
        'avg_freq': avg_freq,
        'mean_step': step_times.mean(),
        'p99_step': np.percentile(step_times, 99),
        'max_step': step_times.max(),
        'jitter': np.std(step_times)
    })
    
    print(f"  Frequency: {avg_freq:6.1f} Hz")
    print(f"  Mean step: {step_times.mean():6.2f} ms")
    print()

# ==============================================================================
# Results Summary
# ==============================================================================

print("=" * 60)
print("RESULTS: Observation Frequency vs Performance")
print("=" * 60)
print()

print(f"{'Interval':>10s} {'Frequency':>12s} {'Mean Step':>12s} {'P99 Step':>12s} {'Status':>10s}")
print("-" * 60)

target_freq = 100.0
best_config = None

for r in results:
    status = "✅ MET" if r['avg_freq'] >= target_freq else "⚠️ MISSED"
    print(f"{r['interval']:>10d} {r['avg_freq']:>11.1f} Hz {r['mean_step']:>11.2f} ms {r['p99_step']:>11.2f} ms {status:>10s}")
    
    if r['avg_freq'] >= target_freq and (best_config is None or r['interval'] < best_config['interval']):
        best_config = r

print()

# ==============================================================================
# Recommendation
# ==============================================================================

print("=" * 60)
print("RECOMMENDATION")
print("=" * 60)
print()

if best_config:
    print(f"✅ REAL-TIME TARGET ACHIEVABLE!")
    print()
    print(f"Recommended configuration:")
    print(f"  Observation interval: {best_config['interval']} steps")
    print(f"  Expected frequency:   {best_config['avg_freq']:.1f} Hz")
    print(f"  Mean step time:       {best_config['mean_step']:.2f} ms")
    print()
    print(f"This provides:")
    print(f"  - Control loop >100 Hz ✅")
    print(f"  - Full observation every {best_config['interval'] * 1e-4:.4f} s")
    print(f"  - {1000 / best_config['interval']:.0f} full observations per 0.1s episode")
else:
    print("⚠️ 100 Hz TARGET NOT ACHIEVABLE with current setup")
    print()
    print("Best performance:")
    best = max(results, key=lambda x: x['avg_freq'])
    print(f"  Observation interval: {best['interval']} steps")
    print(f"  Max frequency:        {best['avg_freq']:.1f} Hz")
    print()
    print("Recommendations:")
    print("  1. Reduce grid resolution (nr, ntheta)")
    print("  2. Implement fast diagnostic (approximate obs)")
    print("  3. GPU acceleration for Poisson solver")

print()
print("=" * 60)
