"""
Standard Tokamak Benchmark Equilibria

Provides reference cases for validation:
- ITER baseline scenario
- DIII-D H-mode shot
- EAST experimental case

Issue #13

Author: 小P ⚛️
Date: 2026-03-24
"""

from .iter_baseline import iter_baseline_equilibrium
from .diiid_hmode import diiid_hmode_equilibrium
from .east_reference import east_reference_equilibrium

__all__ = [
    'iter_baseline_equilibrium',
    'diiid_hmode_equilibrium', 
    'east_reference_equilibrium',
]
