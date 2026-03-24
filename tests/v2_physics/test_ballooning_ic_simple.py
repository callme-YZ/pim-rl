"""
Simple Unit Tests for Ballooning IC Module (v2.0 Physics)

Issue #17: Add unit tests for v2.0 physics modules

Lightweight tests without full equilibrium loading.

Author: 小P ⚛️
Date: 2026-03-24
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import pytest
import jax.numpy as jnp


def test_import_ballooning_module():
    """Test ballooning module can be imported"""
    from pim_rl.physics.v2 import ballooning_ic_v2
    
    assert hasattr(ballooning_ic_v2, 'ballooning_mode_ic_v2')
    print("✅ Ballooning IC module imports successfully")


def test_bout_metric():
    """Test BOUT metric creation"""
    from pim_rl.physics.v2.bout_metric import BOUTMetric
    
    metric = BOUTMetric(R0=1.0, a=0.3, epsilon=0.3)
    
    assert metric.R0 == 1.0
    assert metric.a == 0.3
    print(f"✅ BOUT metric: R0={metric.R0}, a={metric.a}")


def test_field_aligned_coords():
    """Test field-aligned coordinates creation"""
    from pim_rl.physics.v2.bout_metric import BOUTMetric
    from pim_rl.physics.v2.field_aligned import FieldAlignedCoordinates
    
    metric = BOUTMetric(R0=1.0, a=0.3)
    fa = FieldAlignedCoordinates(metric)
    
    assert fa.metric == metric
    print("✅ Field-aligned coordinates created")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
