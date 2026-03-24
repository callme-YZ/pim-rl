#!/usr/bin/env python3
"""
Quick JIT speedup test for Issue #30

Compares:
1. Original CompleteMHDSolver
2. JIT-optimized CompleteMHDSolverJIT

Author: 小A 🤖
Date: 2026-03-24
Issue: #30 Phase 2
"""

import sys
sys.path.insert(0, 'src')

import time
import jax.numpy as jnp
from pim_rl.physics.v2.complete_solver_v2 import CompleteMHDSolver
from pim_rl.physics.v2.complete_solver_v2_jit import CompleteMHDSolverJIT
from pim_rl.physics.v2.elsasser_bracket import ElsasserState
from pim_rl.physics.v2.tearing_ic import create_tearing_ic

print("=" * 60)
print("JIT Speedup Test (Issue #30)")
print("=" * 60)
print()

# Setup
grid_shape = (32, 64, 8)
dr, dtheta, dz = 1/32, 2*3.14159/64, 0.1
epsilon = 0.3
eta = 0.05

print("Creating solvers...")
print()

# Original solver
print("1. Original (non-JIT):")
solver_orig = CompleteMHDSolver(grid_shape, dr, dtheta, dz, epsilon, eta)
print()

# JIT solver
print("2. JIT-optimized:")
solver_jit = CompleteMHDSolverJIT(grid_shape, dr, dtheta, dz, epsilon, eta)
print()

# Initial condition
print("Creating initial condition...")
psi, phi = create_tearing_ic(nr=32, ntheta=64)
state = ElsasserState(
    z_plus=jnp.array(psi + phi, dtype=jnp.float32),
    z_minus=jnp.array(psi - phi, dtype=jnp.float32),
    P=jnp.zeros_like(psi, dtype=jnp.float32)
)
print("✅ Done\n")

# Warm-up (JIT compilation)
print("Warming up JIT solver (compilation)...")
dt = 1e-4
_ = solver_jit.step(state, dt)
print("✅ JIT compiled\n")

# Benchmark
n_steps = 100
print(f"Benchmarking {n_steps} steps...\n")

# Original solver
print("Original solver:")
start = time.perf_counter()
state_tmp = state
for _ in range(n_steps):
    state_tmp = solver_orig.step(state_tmp, dt)
end = time.perf_counter()
time_orig = (end - start) * 1000  # ms
time_per_step_orig = time_orig / n_steps
freq_orig = 1000.0 / time_per_step_orig

print(f"  Total time:    {time_orig:.1f} ms")
print(f"  Time/step:     {time_per_step_orig:.2f} ms")
print(f"  Frequency:     {freq_orig:.1f} Hz")
print()

# JIT solver
print("JIT solver:")
start = time.perf_counter()
state_tmp = state
for _ in range(n_steps):
    state_tmp = solver_jit.step(state_tmp, dt)
end = time.perf_counter()
time_jit = (end - start) * 1000  # ms
time_per_step_jit = time_jit / n_steps
freq_jit = 1000.0 / time_per_step_jit

print(f"  Total time:    {time_jit:.1f} ms")
print(f"  Time/step:     {time_per_step_jit:.2f} ms")
print(f"  Frequency:     {freq_jit:.1f} Hz")
print()

# Speedup
speedup = time_orig / time_jit
print("=" * 60)
print("RESULTS")
print("=" * 60)
print()
print(f"Speedup:        {speedup:.2f}×")
print(f"Original:       {freq_orig:.1f} Hz")
print(f"JIT-optimized:  {freq_jit:.1f} Hz")
print()

# Check 100 Hz target
target = 100.0
if freq_jit >= target:
    print(f"✅ TARGET MET: {freq_jit:.1f} Hz >= {target} Hz")
else:
    print(f"⚠️ TARGET MISSED: {freq_jit:.1f} Hz < {target} Hz")
    print(f"   Need {target/freq_jit:.2f}× more speedup to reach 100 Hz")

print()
print("=" * 60)
