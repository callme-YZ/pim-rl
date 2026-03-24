"""
Grid Convergence Study for PyTokMHD v2.0

Issue #19: Verify spatial discretization error

Tests ballooning mode growth rate on multiple grids.

Author: 小P ⚛️
Date: 2026-03-24
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Tuple, Dict


def run_ballooning_simulation(nr: int, ntheta: int, nz: int, 
                               dt: float = 0.01, 
                               n_steps: int = 100) -> Tuple[float, float]:
    """
    Run ballooning mode simulation on given grid.
    
    Args:
        nr, ntheta, nz: Grid resolution
        dt: Time step
        n_steps: Number of steps
        
    Returns:
        (growth_rate, final_energy)
    """
    from pim_rl.physics.v2.complete_solver import CompleteMHDSolver
    from pim_rl.physics.v2.elsasser_bracket import ElsasserState
    
    # Create solver
    dr = 1.0 / nr
    dtheta = 2*jnp.pi / ntheta
    dz = 2*jnp.pi / nz
    
    solver = CompleteMHDSolver(
        grid_shape=(nr, ntheta, nz),
        dr=dr, dtheta=dtheta, dz=dz,
        epsilon=0.3,
        eta=0.01,
        pressure_scale=0.1
    )
    
    # Initial condition: ballooning-like perturbation
    r = jnp.linspace(0, 1, nr)[:, None, None]
    theta = jnp.linspace(0, 2*jnp.pi, ntheta)[None, :, None]
    z = jnp.linspace(0, 2*jnp.pi, nz)[None, None, :]
    
    # Ballooning mode structure
    amplitude = 0.01
    z_plus = amplitude * jnp.sin(jnp.pi * r) * jnp.cos(theta) * jnp.sin(z)
    z_minus = amplitude * jnp.cos(jnp.pi * r) * jnp.sin(2*theta) * jnp.cos(z)
    P = jnp.ones((nr, ntheta, nz)) * 0.01
    
    state = ElsasserState(z_plus=z_plus, z_minus=z_minus, P=P)
    
    # Track amplitude
    amplitudes = []
    times = []
    
    for step in range(n_steps):
        # Measure amplitude
        amp = jnp.max(jnp.abs(state.z_plus))
        amplitudes.append(float(amp))
        times.append(step * dt)
        
        # Evolve
        state = solver.step_rk2(state, dt=dt)
    
    # Fit exponential growth: A(t) = A₀ exp(γt)
    amplitudes = np.array(amplitudes)
    times = np.array(times)
    
    # Linear fit to log(A)
    log_amps = np.log(amplitudes[10:])  # Skip initial transient
    t_fit = times[10:]
    
    growth_rate = np.polyfit(t_fit, log_amps, 1)[0]
    
    final_energy = float(solver.hamiltonian(state))
    
    return growth_rate, final_energy


def convergence_study(grids: List[Tuple[int, int, int]],
                     dt: float = 0.01,
                     n_steps: int = 100) -> Dict:
    """
    Run convergence study on multiple grids.
    
    Args:
        grids: List of (nr, ntheta, nz) tuples
        dt: Time step (fixed)
        n_steps: Evolution steps
        
    Returns:
        Results dictionary
    """
    print("=" * 70)
    print("Grid Convergence Study - PyTokMHD v2.0")
    print("=" * 70)
    print(f"Time step: dt = {dt}")
    print(f"Evolution: {n_steps} steps (T = {n_steps*dt:.2f})")
    print()
    
    results = {
        'grids': [],
        'growth_rates': [],
        'energies': [],
        'grid_sizes': [],
    }
    
    for i, (nr, ntheta, nz) in enumerate(grids):
        print(f"[{i+1}/{len(grids)}] Grid: {nr}×{ntheta}×{nz}")
        
        gamma, energy = run_ballooning_simulation(nr, ntheta, nz, dt, n_steps)
        
        results['grids'].append((nr, ntheta, nz))
        results['growth_rates'].append(gamma)
        results['energies'].append(energy)
        results['grid_sizes'].append(nr)  # Use nr as characteristic size
        
        print(f"  Growth rate γ = {gamma:.4f}")
        print(f"  Final energy H = {energy:.6e}")
        print()
    
    # Analyze convergence
    print("=" * 70)
    print("Convergence Analysis")
    print("=" * 70)
    
    gamma_values = np.array(results['growth_rates'])
    grid_sizes = np.array(results['grid_sizes'])
    
    # Richardson extrapolation (assuming 2nd order)
    if len(gamma_values) >= 3:
        # Use last 3 grids
        g1, g2, g3 = gamma_values[-3:]
        n1, n2, n3 = grid_sizes[-3:]
        
        # Extrapolate to infinite resolution
        # γ(n) ≈ γ∞ + C/n²
        # Richardson: γ∞ ≈ (4*γ₃ - γ₂) / 3  (for n₃=2n₂)
        if n3 == 2*n2 and n2 == 2*n1:
            gamma_inf = (4*g3 - g2) / 3
            print(f"Richardson extrapolation: γ∞ ≈ {gamma_inf:.4f}")
        
        # Check convergence order
        ratio_12 = abs(g2 - g1) / abs(g3 - g2) if abs(g3 - g2) > 1e-10 else 0
        expected_ratio = 4.0  # For 2nd order
        
        print(f"Convergence ratio: {ratio_12:.2f} (expected ~4 for 2nd order)")
        
        if 3.0 < ratio_12 < 5.0:
            print("✅ Convergence order consistent with 2nd order")
        else:
            print("⚠️  Convergence order may not be 2nd order")
    
    # Relative changes
    print()
    print("Relative changes:")
    for i in range(1, len(gamma_values)):
        rel_change = abs(gamma_values[i] - gamma_values[i-1]) / abs(gamma_values[i])
        print(f"  Grid {i} → {i+1}: {rel_change:.2%}")
    
    return results


def main():
    """Main convergence study"""
    
    # Test grids (increasing resolution)
    grids = [
        (16, 32, 16),   # Coarse
        (32, 64, 32),   # Medium
        (64, 128, 64),  # Fine
    ]
    
    # Run study
    results = convergence_study(
        grids=grids,
        dt=0.01,
        n_steps=100
    )
    
    # Save results
    np.savez('convergence_results.npz',
             grids=results['grids'],
             growth_rates=results['growth_rates'],
             energies=results['energies'])
    
    print()
    print("Results saved to: convergence_results.npz")
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    
    gamma_values = results['growth_rates']
    print(f"Growth rates: {gamma_values}")
    print(f"Finest grid γ = {gamma_values[-1]:.4f}")
    print(f"Convergence: {abs(gamma_values[-1] - gamma_values[-2]) / abs(gamma_values[-1]):.2%} change")


if __name__ == "__main__":
    main()
