#!/usr/bin/env python3
"""
Issue #33: Direct JIT Test (bypass env)

Test JIT functionality directly on solver without environment wrapper.

Author: 小A 🤖
Date: 2026-03-25
"""

import sys
sys.path.insert(0, 'src')

import time
import numpy as np
import jax.numpy as jnp
from pim_rl.physics.v2.complete_solver_v2 import CompleteMHDSolver
from pim_rl.physics.v2.elsasser_bracket import ElsasserState
from pim_rl.physics.v2.tearing_ic import create_tearing_ic

print("=" * 70)
print("Issue #33: Direct JIT Test")
print("=" * 70)
print()

# Create solver
print("Creating solver...")
solver = CompleteMHDSolver(
    grid_shape=(32, 64, 8),
    dr=1/32,
    dtheta=2*np.pi/64,
    dz=0.1,
    epsilon=0.3,
    eta=0.05,
    pressure_scale=0.2
)
print()

# Create initial state
print("Creating initial state...")
psi, phi = create_tearing_ic(nr=32, ntheta=64)

# Convert to Elsasser
from pim_rl.physics.v2.elsasser_mhd_solver import ElsasserMHDSolver
temp_solver = ElsasserMHDSolver(solver)
temp_solver.initialize(jnp.array(psi, dtype=jnp.float32), jnp.array(phi, dtype=jnp.float32))
state0 = temp_solver._state_els

print(f"State shapes:")
print(f"  z_plus:  {state0.z_plus.shape}")
print(f"  z_minus: {state0.z_minus.shape}")
print(f"  P:       {state0.P.shape}")
print()

# Test 1: First call (compilation)
print("Test 1: JIT Compilation (first call)")
print("-" * 70)
start = time.perf_counter()
dstate1 = solver.rhs(state0)
compile_time = (time.perf_counter() - start) * 1000
print(f"First RHS call: {compile_time:.1f} ms (includes compilation)")
print()

# Test 2: Warm calls (compiled)
print("Test 2: Compiled Performance (100 calls)")
print("-" * 70)
times = []
for _ in range(100):
    start = time.perf_counter()
    dstate = solver.rhs(state0)
    times.append((time.perf_counter() - start) * 1000)

mean_time = np.mean(times)
std_time = np.std(times)
print(f"RHS time: {mean_time:.3f} ± {std_time:.3f} ms")
print()

# Compare to baseline
baseline = 16.76  # From Issue #21
speedup = baseline / mean_time
print(f"Baseline (Issue #21): {baseline:.2f} ms")
print(f"With JIT (Issue #33): {mean_time:.2f} ms")
print(f"Speedup:              {speedup:.2f}×")
print()

if speedup >= 2.0:
    print("✅ SUCCESS: ≥2× speedup achieved!")
elif speedup >= 1.5:
    print("⚠️ PARTIAL: 1.5-2× speedup")
else:
    print("❌ BELOW TARGET: <1.5× speedup")
print()

# Test 3: Physics correctness
print("Test 3: Physics Correctness (100 steps)")
print("-" * 70)

H0 = solver.hamiltonian(state0)
print(f"Initial energy: {H0:.6e}")

# Evolve
state = state0
energies = [H0]
for _ in range(100):
    state = solver.step(state, 1e-4)
    H = solver.hamiltonian(state)
    energies.append(H)

energies = np.array(energies)
drift = abs(energies[-1] - energies[0]) / abs(energies[0]) * 100

print(f"Final energy:   {energies[-1]:.6e}")
print(f"Energy drift:   {drift:.2f}%")
print()

if drift < 5.0:
    print("✅ GOOD: <5% energy drift")
elif drift < 20.0:
    print("⚠️ ACCEPTABLE: 5-20% drift")
else:
    print("❌ POOR: >20% drift")
print()

# Summary
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print(f"Compilation:  {compile_time:.1f} ms (one-time cost)")
print(f"Performance:  {speedup:.2f}× speedup ({mean_time:.2f} ms → {1000/mean_time:.0f} Hz)")
print(f"Correctness:  {drift:.2f}% energy drift")
print()

if speedup >= 2.0 and drift < 5.0:
    print("🎉 Issue #33 SUCCESS: JIT working perfectly!")
    print(f"   Target: 2× speedup → Achieved: {speedup:.2f}×")
    print(f"   Physics: <5% drift → Achieved: {drift:.2f}%")
elif speedup >= 1.5:
    print("⚠️ Issue #33 PARTIAL SUCCESS")
    print(f"   Speedup below target: {speedup:.2f}× (target: ≥2×)")
else:
    print("❌ Issue #33 NEEDS INVESTIGATION")
    print(f"   Speedup too low: {speedup:.2f}×")

print()
print("=" * 70)
