"""
Tests for Standard Tokamak Benchmarks

Issue #13

Author: 小P ⚛️
Date: 2026-03-24
"""

import pytest
import jax.numpy as jnp
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from benchmarks.iter_baseline import (
    iter_baseline_equilibrium,
    iter_q_profile,
    ITER_PARAMS,
    validate_iter_equilibrium
)


class TestITERBenchmark:
    """Test ITER baseline benchmark"""
    
    def test_iter_params(self):
        """Test ITER parameters are defined"""
        assert ITER_PARAMS['R0'] == 6.2
        assert ITER_PARAMS['a'] == 2.0
        assert ITER_PARAMS['q_0'] == 1.0
        assert ITER_PARAMS['q_95'] == 3.0
        
        print("✅ ITER parameters defined")
    
    def test_iter_q_profile(self):
        """Test ITER q-profile"""
        psi = jnp.linspace(0, 1, 100)
        q = iter_q_profile(psi, q_0=1.0, q_95=3.0)
        
        # Check central value
        assert abs(q[0] - 1.0) < 0.01, f"q(0) = {q[0]}, expected 1.0"
        
        # Check edge value
        idx_95 = 95
        assert abs(q[idx_95] - 3.0) < 0.2, f"q(0.95) = {q[idx_95]}, expected 3.0"
        
        # Check monotonic
        assert jnp.all(q[1:] >= q[:-1]), "q-profile not monotonic"
        
        print(f"✅ ITER q-profile: q(0)={q[0]:.2f}, q(0.95)={q[95]:.2f}")
    
    def test_iter_equilibrium_generation(self):
        """Test ITER equilibrium generation"""
        iter_eq = iter_baseline_equilibrium()
        
        # Check structure
        assert 'name' in iter_eq
        assert 'params' in iter_eq
        assert 'profiles' in iter_eq
        
        # Check profiles callable
        psi_test = jnp.linspace(0, 1, 10)
        q_test = iter_eq['profiles']['q'](psi_test)
        
        assert q_test.shape == (10,)
        assert jnp.all(jnp.isfinite(q_test))
        
        print(f"✅ ITER equilibrium generated: {iter_eq['name']}")
    
    def test_iter_aspect_ratio(self):
        """Test ITER aspect ratio"""
        A = ITER_PARAMS['R0'] / ITER_PARAMS['a']
        
        assert 3.0 < A < 3.2, f"Aspect ratio A={A:.1f}, expected ~3.1"
        
        print(f"✅ ITER aspect ratio: A={A:.1f}")
    
    def test_profile_bounds(self):
        """Test profiles stay in physical range"""
        psi = jnp.linspace(0, 1, 100)
        
        # q-profile
        q = iter_q_profile(psi)
        assert jnp.all(q > 0), "q-profile has negative values"
        assert jnp.all(q < 20), "q-profile unreasonably large"
        
        # Pressure
        iter_eq = iter_baseline_equilibrium()
        p = iter_eq['profiles']['pressure'](psi)
        assert jnp.all(p >= 0), "Pressure negative"
        
        print("✅ Profiles in physical range")


def test_benchmark_module_import():
    """Test benchmark module can be imported"""
    from benchmarks import iter_baseline_equilibrium
    
    assert iter_baseline_equilibrium is not None
    print("✅ Benchmark module imports")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])


class TestDIIIDBenchmark:
    """Test DIII-D H-mode benchmark"""
    
    def test_diiid_params(self):
        """Test DIII-D parameters"""
        from benchmarks.diiid_hmode import DIIID_PARAMS
        
        assert DIIID_PARAMS['R0'] == 1.67
        assert DIIID_PARAMS['a'] == 0.67
        assert DIIID_PARAMS['q_95'] == 4.0
        
        print("✅ DIII-D parameters defined")
    
    def test_diiid_equilibrium(self):
        """Test DIII-D equilibrium generation"""
        from benchmarks.diiid_hmode import diiid_hmode_equilibrium
        
        diiid_eq = diiid_hmode_equilibrium()
        
        assert 'name' in diiid_eq
        assert 'DIII-D' in diiid_eq['name']
        
        # Test q-profile
        psi = jnp.linspace(0, 1, 100)
        q = diiid_eq['profiles']['q'](psi)
        
        assert q[0] > 1.0, f"q(0) = {q[0]}"
        assert q[95] > 3.5, f"q(0.95) = {q[95]}"
        
        print(f"✅ DIII-D equilibrium: q(0)={q[0]:.2f}, q(0.95)={q[95]:.2f}")


class TestEASTBenchmark:
    """Test EAST reference benchmark"""
    
    def test_east_params(self):
        """Test EAST parameters"""
        from benchmarks.east_reference import EAST_PARAMS
        
        assert EAST_PARAMS['R0'] == 1.85
        assert EAST_PARAMS['a'] == 0.45
        assert EAST_PARAMS['q_95'] == 5.0
        
        print("✅ EAST parameters defined")
    
    def test_east_equilibrium(self):
        """Test EAST equilibrium generation"""
        from benchmarks.east_reference import east_reference_equilibrium
        
        east_eq = east_reference_equilibrium()
        
        assert 'name' in east_eq
        assert 'EAST' in east_eq['name']
        
        # Test q-profile
        psi = jnp.linspace(0, 1, 100)
        q = east_eq['profiles']['q'](psi)
        
        assert q[0] > 1.0, f"q(0) = {q[0]}"
        assert q[95] > 4.5, f"q(0.95) = {q[95]}"
        
        print(f"✅ EAST equilibrium: q(0)={q[0]:.2f}, q(0.95)={q[95]:.2f}")


def test_all_three_benchmarks():
    """Test all three benchmarks can be imported and used"""
    from benchmarks import (
        iter_baseline_equilibrium,
        diiid_hmode_equilibrium,
        east_reference_equilibrium
    )
    
    benchmarks = [
        ('ITER', iter_baseline_equilibrium()),
        ('DIII-D', diiid_hmode_equilibrium()),
        ('EAST', east_reference_equilibrium()),
    ]
    
    for name, eq in benchmarks:
        assert 'params' in eq
        assert 'profiles' in eq
        print(f"✅ {name} benchmark available")
    
    print("\n✅ All three standard benchmarks ready!")
