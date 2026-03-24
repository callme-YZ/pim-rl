"""
Unit Tests for Elsasser Bracket (v2.0 Physics) - Simplified

Issue #17: Add unit tests for v2.0 physics modules

Tests basic Poisson bracket properties on 2D slices.

Author: 小P ⚛️
Date: 2026-03-24
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import pytest
import jax.numpy as jnp

from pim_rl.physics.v2.elsasser_bracket import MorrisonBracket


class TestPoissonBracket2D:
    """Test 2D Poisson bracket properties"""
    
    @pytest.fixture
    def grid(self):
        """Small 2D grid for testing"""
        return MorrisonBracket((16, 16, 1), dr=0.1, dtheta=0.1, dz=0.1)
    
    @pytest.fixture
    def fields_2d(self):
        """2D test fields (r, θ)"""
        Nr, Ntheta = 16, 16
        
        r = jnp.linspace(0, 1, Nr)[:, None]
        theta = jnp.linspace(0, 2*jnp.pi, Ntheta)[None, :]
        
        F = jnp.sin(jnp.pi * r) * jnp.cos(theta)
        G = jnp.cos(jnp.pi * r) * jnp.sin(2*theta)
        H = r**2 * jnp.cos(3*theta)
        
        return F, G, H
    
    def test_antisymmetry(self, grid, fields_2d):
        """
        Test {F,G} = -{G,F}
        
        Fundamental property: bracket is antisymmetric.
        """
        F, G, _ = fields_2d
        
        FG = grid.poisson_bracket_2d(F, G, grid.dr, grid.dtheta)
        GF = grid.poisson_bracket_2d(G, F, grid.dr, grid.dtheta)
        
        max_diff = jnp.max(jnp.abs(FG + GF))
        
        assert max_diff < 1e-6, f"Antisymmetry violated: |{{F,G}} + {{G,F}}|_max = {max_diff:.3e}"
        print(f"✅ Antisymmetry: max error = {max_diff:.3e}")
    
    def test_constant_bracket(self, grid, fields_2d):
        """
        Test {F, const} = 0
        
        Bracket with constant field vanishes.
        """
        F, _, _ = fields_2d
        const = jnp.ones_like(F) * 5.0
        
        F_const = grid.poisson_bracket_2d(F, const, grid.dr, grid.dtheta)
        
        max_val = jnp.max(jnp.abs(F_const))
        
        assert max_val < 1e-8, f"Bracket with constant: max = {max_val:.3e}"
        print(f"✅ Constant bracket: max = {max_val:.3e}")
    
    def test_self_bracket(self, grid, fields_2d):
        """
        Test {F, F} = 0
        
        Self-bracket vanishes (antisymmetry corollary).
        """
        F, _, _ = fields_2d
        
        F_F = grid.poisson_bracket_2d(F, F, grid.dr, grid.dtheta)
        
        max_val = jnp.max(jnp.abs(F_F))
        
        assert max_val < 1e-7, f"Self-bracket: max = {max_val:.3e}"
        print(f"✅ Self-bracket: max = {max_val:.3e}")
    
    def test_linearity(self, grid, fields_2d):
        """
        Test linearity: {aF + bG, H} = a{F,H} + b{G,H}
        
        Bracket is linear in first argument.
        """
        F, G, H = fields_2d
        
        a, b = 2.0, 3.0
        
        # LHS: {aF + bG, H}
        aF_plus_bG = a*F + b*G
        lhs = grid.poisson_bracket_2d(aF_plus_bG, H, grid.dr, grid.dtheta)
        
        # RHS: a{F,H} + b{G,H}
        FH = grid.poisson_bracket_2d(F, H, grid.dr, grid.dtheta)
        GH = grid.poisson_bracket_2d(G, H, grid.dr, grid.dtheta)
        rhs = a*FH + b*GH
        
        max_diff = jnp.max(jnp.abs(lhs - rhs))
        
        assert max_diff < 1e-4, f"Linearity violated: max = {max_diff:.3e}"
        print(f"✅ Linearity: max error = {max_diff:.3e}")


def test_grid_initialization():
    """Test MorrisonBracket grid creation"""
    grid = MorrisonBracket((8, 16, 32), dr=0.05, dtheta=0.1, dz=0.2)
    
    assert grid.Nr == 8
    assert grid.Ntheta == 16
    assert grid.Nz == 32
    assert grid.dr == 0.05
    assert grid.dtheta == 0.1
    assert grid.dz == 0.2
    
    print("✅ Grid initialization test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
