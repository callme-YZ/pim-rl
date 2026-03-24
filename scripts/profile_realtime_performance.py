#!/usr/bin/env python3
"""
Real-time Performance Profiling for Issue #30

Measures:
1. Policy inference latency
2. Environment step breakdown
3. End-to-end control loop timing
4. Sustained throughput

Author: 小A 🤖
Date: 2026-03-24
Issue: #30
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
print("Real-time Performance Profiling (Issue #30)")
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

# Initialize with tearing IC
psi, phi = create_tearing_ic(nr=32, ntheta=64)
env.mhd_solver.initialize(jnp.array(psi, dtype=jnp.float32), jnp.array(phi, dtype=jnp.float32))
env.current_step = 0
env.obs_computer.reset()
obs = env.obs_computer.compute_observation(jnp.array(psi, dtype=jnp.float32), jnp.array(phi, dtype=jnp.float32))

# Cache initialization
env._last_obs = obs
env._last_psi = jnp.array(psi, dtype=jnp.float32)
env._last_phi = jnp.array(phi, dtype=jnp.float32)
env.psi = jnp.array(psi, dtype=jnp.float32)
env.phi = jnp.array(phi, dtype=jnp.float32)

# Create PID agent
agent = make_baseline_agent('pid', env.action_space, Kp=5.0, Ki=0.5, Kd=0.01, target=0.0, dt=1e-4)

print("✅ Setup complete\n")

# ==============================================================================
# Benchmark 1: Policy Inference Latency
# ==============================================================================

print("=" * 60)
print("Benchmark 1: Policy Inference Latency")
print("=" * 60)
print()

# Warm-up (JIT compilation)
print("Warming up (JIT compilation)...")
for _ in range(10):
    _ = agent.act(obs)
print("✅ Warm-up complete\n")

# Measure inference time
n_trials = 1000
latencies = []

print(f"Running {n_trials} inference trials...")
for i in range(n_trials):
    start = time.perf_counter()
    action = agent.act(obs)
    end = time.perf_counter()
    latencies.append((end - start) * 1000)  # Convert to ms

latencies = np.array(latencies)

print(f"\n📊 Policy Inference Latency (PID, {n_trials} trials):")
print(f"  Mean:   {latencies.mean():.3f} ms")
print(f"  Median: {np.median(latencies):.3f} ms")
print(f"  P50:    {np.percentile(latencies, 50):.3f} ms")
print(f"  P95:    {np.percentile(latencies, 95):.3f} ms")
print(f"  P99:    {np.percentile(latencies, 99):.3f} ms")
print(f"  Max:    {latencies.max():.3f} ms")
print()

# Target check
target_latency = 1.0  # ms
if np.percentile(latencies, 99) < target_latency:
    print(f"✅ P99 latency {np.percentile(latencies, 99):.3f} ms < {target_latency} ms (TARGET MET)")
else:
    print(f"⚠️ P99 latency {np.percentile(latencies, 99):.3f} ms > {target_latency} ms (TARGET MISSED)")
print()

# ==============================================================================
# Benchmark 2: Environment Step Breakdown
# ==============================================================================

print("=" * 60)
print("Benchmark 2: Environment Step Breakdown")
print("=" * 60)
print()

# Measure different step modes
n_trials = 100

# Full observation mode
print(f"Mode 1: Full observation (compute_obs=True, {n_trials} trials)...")
times_full = []
for _ in range(n_trials):
    action = agent.act(obs)
    start = time.perf_counter()
    obs, r, term, trunc, info = env.step(action, compute_obs=True)
    end = time.perf_counter()
    times_full.append((end - start) * 1000)

times_full = np.array(times_full)
print(f"  Mean: {times_full.mean():.1f} ms")
print(f"  P99:  {np.percentile(times_full, 99):.1f} ms")
print()

# Cached observation mode
print(f"Mode 2: Cached observation (compute_obs=False, {n_trials} trials)...")
times_cached = []
for _ in range(n_trials):
    action = agent.act(obs)
    start = time.perf_counter()
    obs, r, term, trunc, info = env.step(action, compute_obs=False)
    end = time.perf_counter()
    times_cached.append((end - start) * 1000)

times_cached = np.array(times_cached)
print(f"  Mean: {times_cached.mean():.1f} ms")
print(f"  P99:  {np.percentile(times_cached, 99):.1f} ms")
print()

speedup = times_full.mean() / times_cached.mean()
print(f"📊 Speedup (cached vs full): {speedup:.1f}×")
print()

# ==============================================================================
# Benchmark 3: End-to-End Control Loop
# ==============================================================================

print("=" * 60)
print("Benchmark 3: End-to-End Control Loop (Policy + Env)")
print("=" * 60)
print()

n_trials = 100

# Full mode
print(f"Full mode (compute_obs=True, {n_trials} steps)...")
times_e2e_full = []
for _ in range(n_trials):
    start = time.perf_counter()
    action = agent.act(obs)
    obs, r, term, trunc, info = env.step(action, compute_obs=True)
    end = time.perf_counter()
    times_e2e_full.append((end - start) * 1000)

times_e2e_full = np.array(times_e2e_full)
freq_full = 1000.0 / times_e2e_full.mean()
print(f"  Mean latency: {times_e2e_full.mean():.1f} ms")
print(f"  Frequency:    {freq_full:.1f} Hz")
print()

# Cached mode (real-time)
print(f"Cached mode (compute_obs=False, {n_trials} steps)...")
times_e2e_cached = []
for _ in range(n_trials):
    start = time.perf_counter()
    action = agent.act(obs)
    obs, r, term, trunc, info = env.step(action, compute_obs=False)
    end = time.perf_counter()
    times_e2e_cached.append((end - start) * 1000)

times_e2e_cached = np.array(times_e2e_cached)
freq_cached = 1000.0 / times_e2e_cached.mean()
print(f"  Mean latency: {times_e2e_cached.mean():.1f} ms")
print(f"  Frequency:    {freq_cached:.1f} Hz")
print()

# Target check
target_freq = 100.0  # Hz
if freq_cached >= target_freq:
    print(f"✅ Control frequency {freq_cached:.1f} Hz >= {target_freq} Hz (TARGET MET)")
else:
    print(f"⚠️ Control frequency {freq_cached:.1f} Hz < {target_freq} Hz (TARGET MISSED)")
print()

# ==============================================================================
# Benchmark 4: Sustained Throughput
# ==============================================================================

print("=" * 60)
print("Benchmark 4: Sustained Throughput")
print("=" * 60)
print()

n_steps = 1000
print(f"Running {n_steps} steps in real-time mode (cached obs)...")

step_times = []
start_total = time.time()

for step in range(n_steps):
    step_start = time.perf_counter()
    action = agent.act(obs)
    obs, r, term, trunc, info = env.step(action, compute_obs=(step % 10 == 0))  # Obs every 10 steps
    step_end = time.perf_counter()
    step_times.append((step_end - step_start) * 1000)

end_total = time.time()

step_times = np.array(step_times)
total_time = end_total - start_total
avg_freq = n_steps / total_time

print(f"\n📊 Sustained Performance ({n_steps} steps):")
print(f"  Total time:      {total_time:.2f} s")
print(f"  Avg frequency:   {avg_freq:.1f} Hz")
print(f"  Mean step time:  {step_times.mean():.2f} ms")
print(f"  P99 step time:   {np.percentile(step_times, 99):.2f} ms")
print(f"  Max step time:   {step_times.max():.2f} ms")
print()

# Jitter analysis
jitter = np.std(step_times)
print(f"  Jitter (std):    {jitter:.2f} ms")
print()

if avg_freq >= target_freq:
    print(f"✅ Sustained frequency {avg_freq:.1f} Hz >= {target_freq} Hz (TARGET MET)")
else:
    print(f"⚠️ Sustained frequency {avg_freq:.1f} Hz < {target_freq} Hz (TARGET MISSED)")
print()

# ==============================================================================
# Summary
# ==============================================================================

print("=" * 60)
print("SUMMARY: Real-time Performance")
print("=" * 60)
print()

print("📋 Requirements vs Actual:")
print()
print(f"1. Policy inference <1 ms (P99):")
print(f"   Target: 1.0 ms")
print(f"   Actual: {np.percentile(latencies, 99):.3f} ms")
print(f"   Status: {'✅ MET' if np.percentile(latencies, 99) < 1.0 else '⚠️ MISSED'}")
print()

print(f"2. Control loop >100 Hz:")
print(f"   Target: 100 Hz")
print(f"   Actual: {freq_cached:.1f} Hz (cached mode)")
print(f"   Status: {'✅ MET' if freq_cached >= 100 else '⚠️ MISSED'}")
print()

print(f"3. Sustained throughput >100 Hz:")
print(f"   Target: 100 Hz")
print(f"   Actual: {avg_freq:.1f} Hz (1000 steps)")
print(f"   Status: {'✅ MET' if avg_freq >= 100 else '⚠️ MISSED'}")
print()

# Overall assessment
all_met = (
    np.percentile(latencies, 99) < 1.0 and
    freq_cached >= 100 and
    avg_freq >= 100
)

if all_met:
    print("🎉 ALL REAL-TIME TARGETS MET! 🎉")
    print()
    print("System is ready for real-time deployment!")
else:
    print("⚠️ SOME TARGETS MISSED")
    print()
    print("Optimization needed before real-time deployment.")

print()
print("=" * 60)
print("Profiling Complete")
print("=" * 60)
