"""
EAST Reference Equilibrium

Reference: TBD (need published shot)

Parameters (typical EAST):
- Major radius R0 = 1.85 m
- Minor radius a = 0.45 m
- Aspect ratio A = 4.1
- B0 ~ 2.0 T
- Ip ~ 0.5 MA

Issue #13

Author: 小P ⚛️
Date: 2026-03-24
"""

import jax.numpy as jnp
from typing import Dict


EAST_PARAMS = {
    'R0': 1.85,
    'a': 0.45,
    'B0': 2.0,
    'Ip': 0.5,
    'kappa': 1.6,
    'delta': 0.3,
    'q_95': 5.0,
    'q_0': 1.2,
}


def east_reference_equilibrium(nr: int = 65, nz: int = 65) -> Dict:
    """
    EAST reference benchmark (placeholder).
    
    TODO: Find published shot data
    """
    return {
        'name': 'EAST reference (placeholder)',
        'reference': 'TBD - need published data',
        'params': EAST_PARAMS,
        'profiles': {
            'q': lambda psi: EAST_PARAMS['q_0'] + (EAST_PARAMS['q_95'] - EAST_PARAMS['q_0']) * psi**2,
        },
    }


if __name__ == "__main__":
    print("EAST benchmark: TODO")
