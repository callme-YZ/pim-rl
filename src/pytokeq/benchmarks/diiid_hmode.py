"""
DIII-D H-mode Benchmark Equilibrium

Reference: DIII-D shot #158103 (published benchmark)

Parameters (typical H-mode):
- Major radius R0 = 1.67 m
- Minor radius a = 0.67 m
- Aspect ratio A = 2.5
- B0 ~ 2.0 T
- Ip ~ 1.2 MA
- q_95 ~ 4.0

Issue #13

Author: 小P ⚛️
Date: 2026-03-24
"""

import jax.numpy as jnp
from typing import Dict


DIIID_PARAMS = {
    'R0': 1.67,
    'a': 0.67,
    'B0': 2.0,
    'Ip': 1.2,
    'kappa': 1.8,
    'delta': 0.4,
    'q_95': 4.0,
    'q_0': 1.1,
}


def diiid_hmode_equilibrium(nr: int = 65, nz: int = 65) -> Dict:
    """
    DIII-D H-mode benchmark (placeholder).
    
    TODO: Implement with actual shot data
    """
    return {
        'name': 'DIII-D H-mode (placeholder)',
        'reference': 'TBD - need published shot',
        'params': DIIID_PARAMS,
        'profiles': {
            'q': lambda psi: DIIID_PARAMS['q_0'] + (DIIID_PARAMS['q_95'] - DIIID_PARAMS['q_0']) * psi**2,
        },
    }


if __name__ == "__main__":
    print("DIII-D benchmark: TODO")
