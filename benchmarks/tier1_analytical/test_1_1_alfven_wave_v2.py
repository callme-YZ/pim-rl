"""
Test 1.1: Alfvén Wave Propagation Benchmark (v2 - Normalized Units)

Verify v2.0 MHD solver against analytical Alfvén wave solution.

Key Fix: Use v2.0's normalized units (B~1, v_A~1)

Theory (normalized):
- v_A = 1 (when B₀=1, ρ₀=1 in code units)
- Phase velocity = v_A = 1
- Energy conserved in ideal MHD

Author: 小P ⚛️
Date: 2026-03-23
"""

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import sys
sys.path.insert(0, '../../src')

from pim_rl.physics.v2.complete_solver import CompleteMHDSolver
from pim_rl.physics.v2.elsasser_bracket import ElsasserState


def create_alfven_ic(grid_shape=(32, 64, 32), kz=1.0, amplitude=0.01):
    """Alfvén wave IC in normalized units
    
    Background: B₀ = 1 (z direction), v₀ = 0
    Perturbation: δv_x = δB_x = A sin(kz)
    → Elsasser: z⁺ = δv_x + (B₀ + δB_x), z⁻ = δv_x - (B₀ + δB_x)
    """
    Nr, Ntheta, Nz = grid_shape
    
    z = jnp.linspace(0, 2*jnp.pi, Nz)[None, None, :]
    
    # Background (normalized)
    B0 = 1.0
    
    # Perturbation
    delta = amplitude * jnp.sin(kz * z)
    
    # Elsasser (assuming perturbation in x-direction, background in z)
    # Simplified: z⁺ ≈ B₀ + δ, z⁻ ≈ B₀ - δ
    z_plus = (B0 + delta) * jnp.ones((Nr, Ntheta, Nz))
    z_minus = (B0 - delta) * jnp.ones((Nr, Ntheta, Nz))
    P = jnp.zeros((Nr, Ntheta, Nz))
    
    return ElsasserState(z_plus=z_plus, z_minus=z_minus, P=P)


def measure_wave_velocity(states, times, Nz=32):
    """Extract phase velocity from wave propagation"""
    z_grid = np.linspace(0, 2*np.pi, Nz)
    
    # Track peak position
    peaks = []
    for state in states:
        profile = np.mean(np.array(state.z_plus), axis=(0,1))
        peak_idx = np.argmax(np.abs(profile - np.mean(profile)))
        peaks.append(z_grid[peak_idx])
    
    # Unwrap and fit
    peaks = np.unwrap(np.array(peaks))
    v_phase = np.polyfit(times, peaks, 1)[0]
    
    return v_phase, peaks


def run_benchmark():
    """Run Alfvén wave benchmark"""
    
    print("\n" + "="*60)
    print("Test 1.1: Alfvén Wave (Normalized Units)")
    print("="*60 + "\n")
    
    # Setup
    grid_shape = (32, 64, 32)
    solver = CompleteMHDSolver(
        grid_shape=grid_shape,
        dr=0.05, dtheta=2*np.pi/64, dz=2*np.pi/32,
        epsilon=0.3,
        eta=0.0,  # Ideal MHD
        pressure_scale=0.0  # No pressure
    )
    
    print("Theory: v_A = 1 (normalized units)\n")
    
    # Initial condition
    state0 = create_alfven_ic(grid_shape, kz=1.0, amplitude=0.01)
    
    # Simulate
    print("Running simulation...")
    states = [state0]
    times = [0.0]
    dt = 0.01
    n_steps = 100
    
    for i in range(n_steps):
        state0 = solver.step_rk2(state0, dt)
        states.append(state0)
        times.append((i+1)*dt)
        if (i+1) % 20 == 0:
            print(f"  Step {i+1}/{n_steps}")
    
    times = np.array(times)
    
    # Measure
    print("\nMeasuring phase velocity...")
    v_phase, peaks = measure_wave_velocity(states, times, grid_shape[2])
    
    print("Measuring energy conservation...")
    energies = np.array([solver.hamiltonian(s) for s in states])
    drift = (energies[-1] - energies[0]) / energies[0] * 100
    
    # Results
    v_theory = 1.0
    error = abs(v_phase - v_theory) / v_theory * 100
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"v_A (theory):      {v_theory:.4f}")
    print(f"v_phase (measured): {v_phase:.4f}")
    print(f"Error:             {error:.2f}%")
    print(f"Energy drift:      {drift:.4f}%")
    print("="*60)
    
    # Pass/fail
    pass_vel = error < 1.0
    pass_energy = abs(drift) < 0.1
    
    if pass_vel and pass_energy:
        print("✅ PASS")
    else:
        print("❌ FAIL")
        if not pass_vel:
            print(f"   Velocity error {error:.2f}% > 1%")
        if not pass_energy:
            print(f"   Energy drift {drift:.4f}% > 0.1%")
    
    return {
        'pass': pass_vel and pass_energy,
        'v_theory': v_theory,
        'v_measured': v_phase,
        'error_pct': error,
        'drift_pct': drift,
        'times': times,
        'energies': energies,
        'peaks': peaks
    }


if __name__ == '__main__':
    results = run_benchmark()
    print("\nTest 1.1 complete ⚛️")
