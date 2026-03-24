"""
Tests for ElsasserMHDSolver (Issue #26 Phase 1)

Tests:
1. Round-trip conversion with BC storage
2. Evolution stability
3. Observation consistency

Author: 小P ⚛️
Date: 2026-03-24
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import pytest
import jax.numpy as jnp
import numpy as np

from pim_rl.physics.v2.elsasser_mhd_solver import ElsasserMHDSolver
from pim_rl.physics.v2.complete_solver_v2 import CompleteMHDSolver
from pim_rl.physics.v2.time_integrators import RK2Integrator

from pytokmhd.geometry import ToroidalGrid


class TestElsasserMHDSolver:
    """Test ElsasserMHDSolver wrapper."""
    
    @pytest.fixture
    def setup(self):
        """Create solver."""
        # Grid (must match: nr >= 32 for ToroidalGrid)
        nr, ntheta = 32, 64
        nz = 8
        
        dr, dtheta, dz = 0.01, 0.1, 0.2
        
        # Physics solver
        physics_solver = CompleteMHDSolver(
            (nr, ntheta, nz), dr, dtheta, dz,
            epsilon=0.3, eta=0.01,
            integrator=RK2Integrator()
        )
        
        # Geometry for Poisson solver
        grid = ToroidalGrid(R0=1.0, a=0.3, nr=nr, ntheta=ntheta)
        
        # Wrapper
        solver = ElsasserMHDSolver(physics_solver, grid)
        
        return solver, grid
    
    def test_round_trip_with_bc(self, setup):
        """
        Test: (ψ, φ) → (z⁺, z⁻) → (ψ, φ) with BC storage.
        
        Expected: <5% error (vs 100% with zero BC).
        """
        solver, grid = setup
        
        # Initial state
        r = grid.r_grid
        theta = grid.theta_grid
        
        psi_0 = jnp.array(r**2 * np.sin(theta))
        phi_0 = jnp.array(r**2 * np.cos(theta))
        
        # Initialize
        solver.initialize(psi_0, phi_0)
        
        # Get back
        psi_recovered, phi_recovered = solver.get_mhd_state()
        
        # Error
        psi_error = float(jnp.max(jnp.abs(psi_recovered - psi_0)))
        phi_error = float(jnp.max(jnp.abs(phi_recovered - phi_0)))
        
        psi_rel = psi_error / float(jnp.max(jnp.abs(psi_0)))
        phi_rel = phi_error / float(jnp.max(jnp.abs(phi_0)))
        
        print(f"\nRound-trip with BC storage:")
        print(f"  ψ error: {psi_error:.3e} (rel: {psi_rel*100:.2f}%)")
        print(f"  φ error: {phi_error:.3e} (rel: {phi_rel*100:.2f}%)")
        
        # Tolerance: <10% (Poisson solver + FD errors)
        # Should be MUCH better than 100% with zero BC!
        assert psi_rel < 0.1, f"ψ round-trip error too large: {psi_rel*100:.1f}%"
        assert phi_rel < 0.1, f"φ round-trip error too large: {phi_rel*100:.1f}%"
        
        print("  ✅ PASSED (BC fix works!)")
    
    def test_evolution_stability(self, setup):
        """Test: Evolution for 100 steps without NaN/Inf."""
        
        solver, grid = setup
        
        # Initial state
        r = grid.r_grid
        theta = grid.theta_grid
        
        psi_0 = jnp.array(r**2 * np.sin(theta))
        phi_0 = jnp.array(r**2 * np.cos(theta))
        
        solver.initialize(psi_0, phi_0)
        
        # Evolve
        dt = 0.001
        n_steps = 100
        
        H_init = solver.hamiltonian()
        
        for i in range(n_steps):
            solver.step(dt)
            
            # Check state
            state = solver.get_elsasser_state()
            assert jnp.all(jnp.isfinite(state.z_plus)), f"z⁺ has NaN/Inf at step {i}"
            assert jnp.all(jnp.isfinite(state.z_minus)), f"z⁻ has NaN/Inf at step {i}"
        
        H_final = solver.hamiltonian()
        
        print(f"\nEvolution stability test:")
        print(f"  Steps: {n_steps}")
        print(f"  H_init: {H_init:.3e}")
        print(f"  H_final: {H_final:.3e}")
        print(f"  ΔH/H: {abs(H_final - H_init)/abs(H_init)*100:.2f}%")
        
        # Energy should be bounded (may drift due to resistivity)
        assert abs(H_final) < 10 * abs(H_init), "Energy exploded"
        
        print("  ✅ PASSED (stable evolution)")
    
    def test_observation_interface(self, setup):
        """Test: get_mhd_state() returns valid (ψ, φ) for observation."""
        
        solver, grid = setup
        
        # Initialize
        r = grid.r_grid
        theta = grid.theta_grid
        
        psi_0 = jnp.array(r**2 * np.sin(theta))
        phi_0 = jnp.array(r**2 * np.cos(theta))
        
        solver.initialize(psi_0, phi_0)
        
        # Evolve a bit
        solver.step(0.01)
        
        # Get MHD state
        psi, phi = solver.get_mhd_state()
        
        # Check shape
        assert psi.shape == psi_0.shape, f"ψ shape mismatch: {psi.shape} vs {psi_0.shape}"
        assert phi.shape == phi_0.shape, f"φ shape mismatch: {phi.shape} vs {phi_0.shape}"
        
        # Check finite
        assert jnp.all(jnp.isfinite(psi)), "ψ has NaN/Inf"
        assert jnp.all(jnp.isfinite(phi)), "φ has NaN/Inf"
        
        print(f"\nObservation interface test:")
        print(f"  ψ shape: {psi.shape}")
        print(f"  φ shape: {phi.shape}")
        print(f"  ψ range: [{float(psi.min()):.3e}, {float(psi.max()):.3e}]")
        print(f"  φ range: [{float(phi.min()):.3e}, {float(phi.max()):.3e}]")
        
        print("  ✅ PASSED (observation ready)")


if __name__ == "__main__":
    pytest.main([__file__, '-v', '-s'])
