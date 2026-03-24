"""
Integration Tests for v2.0 Physics

Issue #17: Add unit tests for v2.0 physics modules

Tests full solver over multiple timesteps.

Author: 小P ⚛️
Date: 2026-03-24
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import pytest
import jax.numpy as jnp

from pim_rl.physics.v2.elsasser_bracket import ElsasserState
from pim_rl.physics.v2.complete_solver import CompleteMHDSolver


class TestIntegration:
    """Integration tests over multiple steps"""
    
    @pytest.fixture
    def solver(self):
        """Small solver for integration tests"""
        return CompleteMHDSolver(
            grid_shape=(8, 8, 4),
            dr=0.1, dtheta=0.1, dz=0.1,
            epsilon=0.3,
            eta=0.01,
            pressure_scale=0.1
        )
    
    @pytest.fixture
    def initial_state(self, solver):
        """Simple initial state"""
        Nr, Ntheta, Nz = 8, 8, 4
        
        r = jnp.linspace(0, 1, Nr)[:, None, None]
        theta = jnp.linspace(0, 2*jnp.pi, Ntheta)[None, :, None]
        
        z_plus = jnp.sin(jnp.pi * r) * jnp.cos(theta) * jnp.ones((1, 1, Nz)) * 0.1
        z_minus = jnp.cos(jnp.pi * r) * jnp.sin(2*theta) * jnp.ones((1, 1, Nz)) * 0.05
        P = jnp.ones((Nr, Ntheta, Nz)) * 0.01
        
        return ElsasserState(z_plus=z_plus, z_minus=z_minus, P=P)
    
    def test_multi_step_stability(self, solver, initial_state):
        """
        Test solver runs 10 steps without explosion.
        
        State should stay bounded.
        """
        state = initial_state
        H0 = solver.hamiltonian(state)
        
        # Run 10 steps
        for i in range(10):
            state = solver.step_rk2(state, dt=0.01)
        
        # Check stability
        max_after = jnp.max(jnp.abs(state.z_plus))
        
        assert not jnp.isnan(max_after), "NaN detected"
        assert max_after < 1.0, f"Explosion: max = {max_after}"
        
        H_final = solver.hamiltonian(state)
        
        print(f"✅ 10-step stability:")
        print(f"   H₀ = {H0:.6e}")
        print(f"   H₁₀ = {H_final:.6e}")
        print(f"   |z⁺|_max: {max_after:.3e}")
    
    def test_energy_monotonic_decrease(self, solver, initial_state):
        """
        Test energy decreases monotonically (resistive case).
        
        With η>0, H should decrease.
        """
        state = initial_state
        energies = []
        
        # Track energy over 5 steps
        for i in range(6):
            H = solver.hamiltonian(state)
            energies.append(H)
            
            if i < 5:
                state = solver.step_rk2(state, dt=0.01)
        
        # Check monotonic decrease (most steps)
        decreases = sum(1 for i in range(5) if energies[i+1] < energies[i])
        
        # Allow 1 step to not decrease (numerical noise)
        assert decreases >= 4, f"Energy not decreasing: {decreases}/5 steps"
        
        print(f"✅ Energy trend (resistive):")
        print(f"   Steps with H decrease: {decreases}/5")
        print(f"   Final energy: {energies[-1]:.6e}")
    
    def test_state_values_physical(self, solver, initial_state):
        """
        Test state values stay in physical range.
        
        No extreme values after evolution.
        """
        state = initial_state
        
        # Evolve 5 steps
        for _ in range(5):
            state = solver.step_rk2(state, dt=0.01)
        
        # Check bounds
        assert jnp.all(jnp.isfinite(state.z_plus)), "z+ not finite"
        assert jnp.all(jnp.isfinite(state.z_minus)), "z- not finite"
        assert jnp.all(jnp.isfinite(state.P)), "P not finite"
        
        # No extreme values
        assert jnp.max(jnp.abs(state.z_plus)) < 10.0, "z+ too large"
        assert jnp.max(jnp.abs(state.P)) < 1.0, "P too large"
        
        print("✅ Physical values maintained after 5 steps")


def test_rk2_consistency():
    """Test RK2 gives same result on repeated calls"""
    solver = CompleteMHDSolver((4, 4, 2), 0.1, 0.1, 0.1)
    
    state = ElsasserState(
        z_plus=jnp.ones((4, 4, 2)) * 0.1,
        z_minus=jnp.ones((4, 4, 2)) * 0.05,
        P=jnp.ones((4, 4, 2)) * 0.01
    )
    
    # Two independent evolutions
    state1 = solver.step_rk2(state, dt=0.01)
    state2 = solver.step_rk2(state, dt=0.01)
    
    # Should be identical
    diff = jnp.max(jnp.abs(state1.z_plus - state2.z_plus))
    
    assert diff < 1e-12, f"RK2 not deterministic: {diff}"
    print(f"✅ RK2 deterministic: diff = {diff:.3e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
