"""
Import Tests for Remaining v2.0 Modules

Issue #17: Add unit tests for v2.0 physics modules

Quick smoke tests for less-critical modules.

Author: 小P ⚛️
Date: 2026-03-24
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import pytest


def test_import_equilibrium_to_v2():
    """Test equilibrium conversion module"""
    from pim_rl.physics.v2 import equilibrium_to_v2
    
    assert hasattr(equilibrium_to_v2, 'solovev_to_toroidal')
    print("✅ equilibrium_to_v2 module imports")


def test_import_rmp_forcing():
    """Test RMP forcing module"""
    from pim_rl.physics.v2 import rmp_forcing
    
    assert hasattr(rmp_forcing, 'rmp_coil_field')
    print("✅ rmp_forcing module imports")


def test_import_complete_solver_with_rmp():
    """Test complete solver with RMP"""
    from pim_rl.physics.v2 import complete_solver_with_rmp
    
    # Module should exist
    assert complete_solver_with_rmp is not None
    print("✅ complete_solver_with_rmp module imports")


def test_import_bout_metric():
    """Test BOUT metric module"""
    from pim_rl.physics.v2.bout_metric import BOUTMetric
    
    metric = BOUTMetric(R0=1.0, a=0.3)
    assert metric.R0 == 1.0
    print("✅ BOUTMetric class works")


def test_import_field_aligned():
    """Test field-aligned coordinates"""
    from pim_rl.physics.v2.field_aligned import FieldAlignedCoordinates
    from pim_rl.physics.v2.bout_metric import BOUTMetric
    
    metric = BOUTMetric(R0=1.0, a=0.3)
    fa = FieldAlignedCoordinates(metric)
    
    assert fa.metric == metric
    print("✅ FieldAlignedCoordinates works")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
