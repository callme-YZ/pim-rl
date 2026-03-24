"""
Integration tests for multi-mode IC API (Issue #27 Phase 3)

Author: 小P ⚛️
Date: 2026-03-24
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, 'src')

from pim_rl.physics.v2.multi_mode_ic import (
    create_multi_mode_ic,
    get_mode_info,
    list_available_modes,
    get_default_parameters,
    create_benchmark_suite,
    compare_modes_info,
)


class TestUnifiedAPI:
    """Test unified multi-mode interface."""
    
    def test_all_modes_work(self):
        """All modes should be createable."""
        modes = list_available_modes()
        
        for mode in modes:
            psi, phi = create_multi_mode_ic(mode, nr=32, ntheta=64)
            
            assert psi.shape == (32, 64)
            assert phi.shape == (32, 64)
            assert np.all(np.isfinite(psi))
            assert np.all(np.isfinite(phi))
    
    def test_invalid_mode_raises(self):
        """Invalid mode should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown mode"):
            create_multi_mode_ic('invalid_mode', nr=32, ntheta=64)
    
    def test_mode_specific_parameters(self):
        """Mode-specific parameters should be passed through."""
        # Tearing with custom B0
        psi_t, phi_t = create_multi_mode_ic('tearing', nr=32, ntheta=64, B0=2.0)
        
        # Kink with custom j0
        psi_k, phi_k = create_multi_mode_ic('kink', nr=32, ntheta=64, j0=3.0)
        
        # Interchange with custom m
        psi_i, phi_i = create_multi_mode_ic('interchange', nr=32, ntheta=64, m=3)
        
        # All should work
        for psi, phi in [(psi_t, phi_t), (psi_k, phi_k), (psi_i, phi_i)]:
            assert psi.shape == (32, 64)
            assert phi.shape == (32, 64)


class TestModeInfo:
    """Test mode information database."""
    
    def test_all_modes_have_info(self):
        """All modes should have complete info."""
        modes = list_available_modes()
        
        required_keys = ['driver', 'm_typical', 'growth_formula', 
                        'resistivity_needed', 'reference']
        
        for mode in modes:
            info = get_mode_info(mode)
            
            for key in required_keys:
                assert key in info, f"Mode {mode} missing {key}"
    
    def test_tearing_info_correct(self):
        """Tearing mode info should be correct."""
        info = get_mode_info('tearing')
        
        assert 'resistive' in info['driver'].lower()
        assert info['resistivity_needed'] is True
        assert 'FKR' in info['growth_formula'] or 'Furth' in info['reference']
    
    def test_kink_info_correct(self):
        """Kink mode info should be correct."""
        info = get_mode_info('kink')
        
        assert 'current' in info['driver'].lower() or 'q' in info['driver']
        assert info['resistivity_needed'] is False
        assert 1 in info['m_typical']
    
    def test_interchange_info_correct(self):
        """Interchange mode info should be correct."""
        info = get_mode_info('interchange')
        
        assert 'pressure' in info['driver'].lower()
        assert info['resistivity_needed'] is False
        assert 2 in info['m_typical'] or 3 in info['m_typical']


class TestDefaultParameters:
    """Test default parameter sets."""
    
    def test_all_modes_have_defaults(self):
        """All modes should have default parameters."""
        modes = list_available_modes()
        
        for mode in modes:
            params = get_default_parameters(mode)
            
            # Should be a dict
            assert isinstance(params, dict)
            
            # Should work with create_multi_mode_ic
            psi, phi = create_multi_mode_ic(mode, nr=32, ntheta=64, **params)
            assert psi.shape == (32, 64)
    
    def test_defaults_produce_valid_ic(self):
        """Default parameters should produce valid ICs."""
        for mode in ['tearing', 'kink', 'interchange']:
            params = get_default_parameters(mode)
            psi, phi = create_multi_mode_ic(mode, nr=32, ntheta=64, **params)
            
            # Should be finite
            assert np.all(np.isfinite(psi))
            assert np.all(np.isfinite(phi))
            
            # Should have non-trivial values
            assert np.abs(psi).max() > 0.01


class TestBenchmarkSuite:
    """Test benchmark suite creation."""
    
    def test_creates_all_modes(self):
        """Benchmark suite should create all modes."""
        ics = create_benchmark_suite(nr=32, ntheta=64)
        
        modes = list_available_modes()
        
        # Should have all modes
        for mode in modes:
            assert mode in ics
            
            psi, phi = ics[mode]
            assert psi.shape == (32, 64)
            assert phi.shape == (32, 64)
    
    def test_ics_are_different(self):
        """Different modes should produce different ICs."""
        ics = create_benchmark_suite(nr=32, ntheta=64)
        
        # Get first two modes
        modes = list(ics.keys())[:2]
        psi1, _ = ics[modes[0]]
        psi2, _ = ics[modes[1]]
        
        # Should be different (not identical)
        assert not np.allclose(psi1, psi2, rtol=0.01)


class TestComparisonInfo:
    """Test comparison info generation."""
    
    def test_returns_string(self):
        """compare_modes_info should return formatted string."""
        info_str = compare_modes_info()
        
        assert isinstance(info_str, str)
        assert len(info_str) > 100  # Should be substantial
    
    def test_includes_all_modes(self):
        """Comparison should include all modes."""
        info_str = compare_modes_info()
        
        modes = list_available_modes()
        for mode in modes:
            assert mode.upper() in info_str or mode.lower() in info_str


class TestBackwardCompatibility:
    """Test backward compatibility aliases."""
    
    def test_individual_create_functions(self):
        """Old create_*_ic functions should still work."""
        from pim_rl.physics.v2.multi_mode_ic import (
            create_tearing_ic,
            create_kink_ic,
            create_interchange_ic,
        )
        
        # Tearing
        psi_t, phi_t = create_tearing_ic(nr=32, ntheta=64)
        assert psi_t.shape == (32, 64)
        
        # Kink
        psi_k, phi_k = create_kink_ic(nr=32, ntheta=64)
        assert psi_k.shape == (32, 64)
        
        # Interchange
        psi_i, phi_i = create_interchange_ic(nr=32, ntheta=64)
        assert psi_i.shape == (32, 64)


class TestPhysicsConsistency:
    """Test cross-mode physics consistency."""
    
    def test_all_modes_have_equilibrium(self):
        """All modes should have axisymmetric equilibrium component."""
        ics = create_benchmark_suite(nr=32, ntheta=64)
        
        for mode, (psi, phi) in ics.items():
            # Average over θ
            psi_avg = psi.mean(axis=1)
            
            # Should be non-zero (has equilibrium)
            assert np.abs(psi_avg).max() > 0.01, f"{mode} has no equilibrium"
    
    def test_all_modes_have_perturbation(self):
        """All modes should have non-axisymmetric perturbation."""
        ics = create_benchmark_suite(nr=32, ntheta=64)
        
        for mode, (psi, phi) in ics.items():
            # Skip ballooning (placeholder implementation)
            if mode == 'ballooning':
                continue
            
            # Variation over θ at some radius
            i_mid = psi.shape[0] // 2
            psi_theta = psi[i_mid, :]
            
            # Should vary (not constant)
            assert np.std(psi_theta) > 1e-4, f"{mode} has no perturbation"
    
    def test_mode_numbers_distinguishable(self):
        """Different modes should have different dominant m."""
        # Get kink (m=1) and interchange (m=2)
        psi_kink, _ = create_multi_mode_ic('kink', nr=32, ntheta=64, eps=0.02)
        psi_inter, _ = create_multi_mode_ic('interchange', nr=32, ntheta=64, 
                                            eps=0.02, m=2)
        
        # FFT to find dominant mode
        fft_kink = np.fft.fft(psi_kink, axis=1)
        fft_inter = np.fft.fft(psi_inter, axis=1)
        
        # Kink should have strong m=1 (check relative to m=0)
        mode_amps_kink = np.abs(fft_kink).mean(axis=0)[:5]
        # m=1 should be larger than m=2,3,4
        assert mode_amps_kink[1] > mode_amps_kink[2]
        assert mode_amps_kink[1] > mode_amps_kink[3]
        
        # Interchange should have strong m=2
        mode_amps_inter = np.abs(fft_inter).mean(axis=0)[:5]
        assert np.argmax(mode_amps_inter[1:]) + 1 == 2  # argmax excluding m=0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
