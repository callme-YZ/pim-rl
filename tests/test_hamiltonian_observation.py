"""
Tests for Hamiltonian Observation (Issue #25)

Validates:
1. Conserved quantities computation (helicity, enstrophy)
2. Dissipation rate dH/dt
3. Observation structure and dimensionality
4. Performance

Author: 小A 🤖
Date: 2026-03-24
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import pytest
import jax.numpy as jnp
import numpy as np

from pytokmhd.geometry.toroidal import ToroidalGrid
from pytokmhd.solvers.hamiltonian_mhd_grad import HamiltonianGradientComputer
from pytokmhd.rl.hamiltonian_observation import (
    HamiltonianObservation,
    HamiltonianObservationScalar,
    ObservationNormalizer
)


class TestHamiltonianObservation:
    """Test full Hamiltonian observation."""
    
    @pytest.fixture
    def grid(self):
        """Small grid for tests."""
        return ToroidalGrid(
            R0=1.5,
            a=0.5,
            nr=32,
            ntheta=64
        )
    
    @pytest.fixture
    def grad_computer(self, grid):
        """Gradient computer from Issue #24."""
        return HamiltonianGradientComputer(grid)
    
    @pytest.fixture
    def observer(self, grid, grad_computer):
        """Observation computer."""
        return HamiltonianObservation(grid, grad_computer, dt=0.01)
    
    @pytest.fixture
    def test_state(self, grid):
        """Simple test state."""
        nr, ntheta = 32, 64
        r = jnp.linspace(0.1, 1.0, nr)[:, None]
        theta = jnp.linspace(0, 2*jnp.pi, ntheta)[None, :]
        
        psi = jnp.sin(jnp.pi * r) * jnp.cos(theta) * 0.1
        phi = jnp.cos(jnp.pi * r) * jnp.sin(2*theta) * 0.05
        
        return psi, phi
    
    def test_observation_structure(self, observer, test_state):
        """Test observation has correct structure."""
        psi, phi = test_state
        obs = observer.compute_observation(psi, phi)
        
        # Check top-level keys
        assert 'hamiltonian' in obs
        assert 'conserved' in obs
        assert 'dissipation' in obs
        assert 'state_summary' in obs
        
        # Check hamiltonian keys
        assert 'H' in obs['hamiltonian']
        assert 'grad_psi' in obs['hamiltonian']
        assert 'grad_phi' in obs['hamiltonian']
        
        # Check conserved keys
        assert 'energy' in obs['conserved']
        assert 'helicity' in obs['conserved']
        assert 'enstrophy' in obs['conserved']
        
        # Check dissipation keys
        assert 'dH_dt' in obs['dissipation']
        assert 'energy_drift' in obs['dissipation']
        
        print("✅ Observation structure correct")
    
    def test_hamiltonian_value(self, observer, test_state):
        """Test Hamiltonian energy is computed."""
        psi, phi = test_state
        obs = observer.compute_observation(psi, phi)
        
        H = obs['hamiltonian']['H']
        
        assert isinstance(H, float)
        assert not jnp.isnan(H)
        assert not jnp.isinf(H)
        
        print(f"✅ Hamiltonian: H = {H:.6e}")
    
    def test_conserved_quantities(self, observer, test_state):
        """Test conserved quantities are finite."""
        psi, phi = test_state
        obs = observer.compute_observation(psi, phi)
        
        K = obs['conserved']['helicity']
        Omega = obs['conserved']['enstrophy']
        
        # Check finite
        assert not jnp.isnan(K)
        assert not jnp.isnan(Omega)
        assert not jnp.isinf(K)
        assert not jnp.isinf(Omega)
        
        # Enstrophy should be positive (J² ≥ 0)
        assert Omega >= 0, "Enstrophy must be non-negative"
        
        print(f"✅ Conserved quantities: K={K:.6e}, Ω={Omega:.6e}")
    
    def test_dissipation_rate(self, observer, test_state):
        """Test dissipation rate computation."""
        psi, phi = test_state
        
        # First call (no previous H)
        obs1 = observer.compute_observation(psi, phi)
        dH_dt1 = obs1['dissipation']['dH_dt']
        
        assert dH_dt1 == 0.0, "First step should have dH/dt = 0"
        
        # Second call (with slightly different state → energy change)
        psi2 = psi * 0.99  # Slightly reduce energy
        obs2 = observer.compute_observation(psi2, phi)
        dH_dt2 = obs2['dissipation']['dH_dt']
        
        # Should have negative dH/dt (energy decreased)
        assert dH_dt2 < 0, f"Energy decreased, dH/dt should be < 0, got {dH_dt2}"
        
        print(f"✅ Dissipation rate: dH/dt = {dH_dt2:.6e}")
    
    def test_gradients_shape(self, observer, test_state):
        """Test gradient fields have correct shape."""
        psi, phi = test_state
        obs = observer.compute_observation(psi, phi)
        
        grad_psi = obs['hamiltonian']['grad_psi']
        grad_phi = obs['hamiltonian']['grad_phi']
        
        # Should match input shape
        assert grad_psi.shape == psi.shape
        assert grad_phi.shape == phi.shape
        
        # Should be finite
        assert jnp.all(jnp.isfinite(grad_psi))
        assert jnp.all(jnp.isfinite(grad_phi))
        
        print("✅ Gradient shapes correct")
    
    def test_state_summary(self, observer, test_state):
        """Test state summary features."""
        psi, phi = test_state
        obs = observer.compute_observation(psi, phi)
        
        grad_norm = obs['state_summary']['grad_norm']
        max_current = obs['state_summary']['max_current']
        
        # Should be positive
        assert grad_norm > 0
        assert max_current >= 0
        
        print(f"✅ State summary: ||∇H||={grad_norm:.6e}, max|J|={max_current:.6e}")
    
    def test_reset(self, observer, test_state):
        """Test reset clears state."""
        psi, phi = test_state
        
        # Compute once
        observer.compute_observation(psi, phi)
        assert observer.H_prev is not None
        
        # Reset
        observer.reset()
        assert observer.H_prev is None
        
        print("✅ Reset works")


class TestHamiltonianObservationScalar:
    """Test scalar observation (22D)."""
    
    @pytest.fixture
    def observer(self):
        """Scalar observer."""
        grid = ToroidalGrid(R0=1.5, a=0.5, nr=32, ntheta=64)
        grad_computer = HamiltonianGradientComputer(grid)
        return HamiltonianObservationScalar(grid, grad_computer, dt=0.01, n_modes=8)
    
    @pytest.fixture
    def test_state(self):
        """Test state."""
        nr, ntheta = 32, 64
        r = jnp.linspace(0.1, 1.0, nr)[:, None]
        theta = jnp.linspace(0, 2*jnp.pi, ntheta)[None, :]
        
        psi = jnp.sin(jnp.pi * r) * jnp.cos(theta) * 0.1
        phi = jnp.cos(jnp.pi * r) * jnp.sin(2*theta) * 0.05
        
        return psi, phi
    
    def test_observation_dimension(self, observer, test_state):
        """Test observation is 22D."""
        psi, phi = test_state
        obs = observer.compute_observation(psi, phi)
        
        expected_dim = 7 + 8 + 8  # 7 scalars + 8 psi modes + 8 phi modes
        assert obs.shape == (expected_dim,), f"Expected shape ({expected_dim},), got {obs.shape}"
        
        print(f"✅ Observation dimension: {obs.shape[0]}D")
    
    def test_observation_finite(self, observer, test_state):
        """Test all values are finite."""
        psi, phi = test_state
        obs = observer.compute_observation(psi, phi)
        
        assert np.all(np.isfinite(obs)), "All observation values must be finite"
        
        print("✅ All values finite")
    
    def test_fourier_modes(self, observer, test_state):
        """Test Fourier modes are extracted."""
        psi, phi = test_state
        obs = observer.compute_observation(psi, phi)
        
        # Last 16 elements are Fourier modes (8 psi + 8 phi)
        psi_modes = obs[7:15]
        phi_modes = obs[15:23]
        
        # Should be non-zero (for non-trivial state)
        assert np.any(psi_modes > 0)
        assert np.any(phi_modes > 0)
        
        print(f"✅ Fourier modes: psi={psi_modes[:3]}, phi={phi_modes[:3]}")


class TestObservationNormalizer:
    """Test observation normalization."""
    
    def test_normalization(self):
        """Test mean/std normalization."""
        normalizer = ObservationNormalizer(obs_dim=5)
        
        # Feed same observation multiple times
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        
        for _ in range(100):
            obs_norm = normalizer.normalize(obs)
        
        # After many samples, normalized obs should be ~ 0
        assert np.allclose(obs_norm, 0.0, atol=0.1)
        
        print(f"✅ Normalization: mean ≈ {normalizer.mean}, std ≈ {np.sqrt(normalizer.var)}")
    
    def test_running_stats(self):
        """Test running statistics update."""
        normalizer = ObservationNormalizer(obs_dim=3)
        
        # Feed sequence
        for i in range(10):
            obs = np.array([i, i*2, i*3], dtype=np.float32)
            normalizer.normalize(obs)
        
        # Check mean is around middle value
        assert normalizer.count == 10
        assert np.allclose(normalizer.mean, [4.5, 9.0, 13.5], atol=0.5)
        
        print(f"✅ Running stats: count={normalizer.count}, mean={normalizer.mean}")
    
    def test_clip(self):
        """Test clipping outliers."""
        normalizer = ObservationNormalizer(obs_dim=2, clip=3.0)
        
        # Feed normal values
        for _ in range(10):
            normalizer.normalize(np.array([0.0, 0.0]))
        
        # Feed outlier
        obs_outlier = np.array([100.0, 100.0])
        obs_norm = normalizer.normalize(obs_outlier)
        
        # Should be clipped to [-3, 3]
        assert np.all(np.abs(obs_norm) <= 3.0)
        
        print(f"✅ Clipping works: outlier normalized to {obs_norm}")


class TestPerformance:
    """Test observation computation performance."""
    
    def test_compute_time(self):
        """Test observation computation is fast."""
        import time
        
        grid = ToroidalGrid(R0=1.5, a=0.5, nr=32, ntheta=64)
        grad_computer = HamiltonianGradientComputer(grid)
        observer = HamiltonianObservation(grid, grad_computer)
        
        # Test state
        nr, ntheta = 32, 64
        r = jnp.linspace(0.1, 1.0, nr)[:, None]
        theta = jnp.linspace(0, 2*jnp.pi, ntheta)[None, :]
        psi = jnp.sin(jnp.pi * r) * jnp.cos(theta) * 0.1
        phi = jnp.cos(jnp.pi * r) * jnp.sin(2*theta) * 0.05
        
        # Warmup
        for _ in range(3):
            observer.compute_observation(psi, phi)
        
        # Benchmark
        n_runs = 100
        start = time.time()
        for _ in range(n_runs):
            obs = observer.compute_observation(psi, phi)
        elapsed = time.time() - start
        
        time_per_call = (elapsed / n_runs) * 1e6  # microseconds
        
        print(f"\n{'='*60}")
        print(f"Performance Benchmark (32×64 grid)")
        print(f"{'='*60}")
        print(f"  Time per compute_observation(): {time_per_call:.2f} μs")
        print(f"  Target: < 100 μs")
        print(f"  Status: {'✅ PASS' if time_per_call < 100 else '❌ FAIL'}")
        print(f"{'='*60}\n")
        
        assert time_per_call < 100, f"Too slow: {time_per_call:.2f} μs > 100 μs"


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "-s"])
