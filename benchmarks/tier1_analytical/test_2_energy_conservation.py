"""
Test 2: Energy Conservation (Formalized)
Validate v2.0 structure-preserving properties
Author: 小P ⚛️
"""
import numpy as np
import jax.numpy as jnp
import sys
sys.path.insert(0, '../../src')

from pim_rl.physics.v2.complete_solver import CompleteMHDSolver
from pim_rl.physics.v2.elsasser_bracket import ElsasserState


def simple_ic(grid_shape=(32, 64, 32), amp=0.1):
    """Simple perturbation IC for energy conservation test"""
    Nr, Ntheta, Nz = grid_shape
    
    # Background + small perturbation
    z_plus = jnp.ones((Nr, Ntheta, Nz)) + amp * jnp.sin(jnp.linspace(0, 2*jnp.pi, Nr))[:, None, None]
    z_minus = jnp.ones((Nr, Ntheta, Nz)) - amp * jnp.sin(jnp.linspace(0, 2*jnp.pi, Nr))[:, None, None]
    P = jnp.zeros((Nr, Ntheta, Nz))
    
    return ElsasserState(z_plus=z_plus, z_minus=z_minus, P=P)


def test_energy_conservation():
    """Test energy conservation over long evolution"""
    
    print("\n" + "="*60)
    print("Test 2: Energy Conservation")
    print("="*60)
    
    # Setup (ideal MHD for strict conservation)
    grid_shape = (32, 64, 32)
    solver = CompleteMHDSolver(
        grid_shape=grid_shape,
        dr=0.05, dtheta=2*jnp.pi/64, dz=2*jnp.pi/32,
        epsilon=0.3,
        eta=0.0,  # Ideal (no dissipation)
        pressure_scale=0.0  # No external forcing
    )
    
    print("\nPhysics: Ideal MHD (η=0, no ∇p)")
    print("Expected: Energy drift <0.1% (structure-preserving)\n")
    
    # Initial condition
    state = simple_ic(grid_shape, amp=0.1)
    E0 = solver.hamiltonian(state)
    
    print(f"Initial energy: {E0:.6f}")
    
    # Simulate
    print("\nRunning 300-step simulation...")
    energies = [E0]
    times = [0.0]
    dt = 0.01
    n_steps = 300
    
    for i in range(n_steps):
        state = solver.step_rk2(state, dt)
        E = solver.hamiltonian(state)
        energies.append(E)
        times.append((i+1)*dt)
        
        if (i+1) % 60 == 0:
            drift = (E - E0) / E0 * 100
            print(f"  Step {i+1}/{n_steps}, E={E:.6f}, drift={drift:.4f}%")
    
    # Results
    energies = np.array(energies)
    times = np.array(times)
    
    E_final = energies[-1]
    drift_final = (E_final - E0) / E0 * 100
    drift_max = np.max(np.abs((energies - E0) / E0)) * 100
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Initial energy:  {E0:.6f}")
    print(f"Final energy:    {E_final:.6f}")
    print(f"Final drift:     {drift_final:.4f}%")
    print(f"Max drift:       {drift_max:.4f}%")
    print("="*60)
    
    # Pass/fail
    if abs(drift_final) < 0.1:
        print("✅ PASS: Energy drift <0.1% (structure-preserving verified)")
        passed = True
    else:
        print(f"❌ FAIL: Energy drift {abs(drift_final):.4f}% > 0.1%")
        passed = False
    
    # Additional checks
    print("\nStructure-preserving properties:")
    print(f"  Drift magnitude: {abs(drift_final):.4f}%")
    print(f"  No secular growth: {drift_max < 1.0}")
    
    return {
        'passed': passed,
        'E0': E0,
        'E_final': E_final,
        'drift_pct': drift_final,
        'drift_max_pct': drift_max
    }


def test_multi_parameter():
    """Test energy conservation across different parameters"""
    
    print("\n" + "="*60)
    print("Multi-Parameter Energy Conservation Test")
    print("="*60)
    
    # Test different resistivities
    etas = [0.0, 0.001, 0.01]
    results = []
    
    for eta in etas:
        print(f"\nTesting η={eta}...")
        
        solver = CompleteMHDSolver(
            grid_shape=(32, 64, 32),
            dr=0.05, dtheta=2*jnp.pi/64, dz=2*jnp.pi/32,
            epsilon=0.3,
            eta=eta,
            pressure_scale=0.0
        )
        
        state = simple_ic((32, 64, 32), amp=0.05)
        E0 = solver.hamiltonian(state)
        
        # Short run
        for _ in range(100):
            state = solver.step_rk2(state, 0.01)
        
        E_final = solver.hamiltonian(state)
        drift = (E_final - E0) / E0 * 100
        
        print(f"  Drift: {drift:.4f}%")
        results.append({'eta': eta, 'drift': drift})
    
    print("\n" + "-"*60)
    print("Summary:")
    for r in results:
        status = "✅" if abs(r['drift']) < 1.0 else "⚠️"
        print(f"  η={r['eta']:.3f}: drift={r['drift']:.4f}% {status}")
    print("-"*60)
    
    return results


if __name__ == '__main__':
    # Main test
    print("\n" + "#"*60)
    print("# Test 2: Energy Conservation Validation")
    print("#"*60)
    
    result_main = test_energy_conservation()
    
    print("\n")
    result_multi = test_multi_parameter()
    
    print("\n" + "="*60)
    print("Test 2 complete ⚛️")
    print("="*60)
