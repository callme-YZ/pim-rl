"""
EAST Reference Equilibrium

Reference: Typical EAST long-pulse scenario
(Based on Wan et al., Nucl. Fusion 55 (2015) and EAST publications)

Parameters (typical long-pulse H-mode):
- Major radius R0 = 1.85 m
- Minor radius a = 0.45 m
- Aspect ratio A = 4.1
- B0 = 2.0 T
- Ip = 0.5 MA (lower current, long pulse)
- κ = 1.6 (moderate elongation)
- δ = 0.3 (moderate triangularity)
- q_95 ~ 5.0 (higher q for stability)
- β_N ~ 1.5

Note: EAST focuses on long-pulse operation.
Parameters optimized for sustained H-mode.

Issue #13

Author: 小P ⚛️
Date: 2026-03-24
"""

import jax.numpy as jnp
from typing import Dict, Optional


EAST_PARAMS = {
    'R0': 1.85,      # Major radius [m]
    'a': 0.45,       # Minor radius [m]
    'B0': 2.0,       # Toroidal field [T]
    'Ip': 0.5,       # Plasma current [MA] (low for long pulse)
    'kappa': 1.6,    # Elongation
    'delta': 0.3,    # Triangularity
    'beta_N': 1.5,   # Normalized beta
    'q_95': 5.0,     # Safety factor (high for stability)
    'q_0': 1.2,      # Central q
}


def east_q_profile(psi_norm: jnp.ndarray, q_0: float = 1.2, q_95: float = 5.0) -> jnp.ndarray:
    """
    EAST q-profile.
    
    Broad profile (low current) for long-pulse stability.
    
    Args:
        psi_norm: Normalized flux
        q_0: Central q
        q_95: Edge q
        
    Returns:
        q-profile
    """
    # Broad profile: α ~ 1.8
    alpha = 1.8
    psi_95 = 0.95
    
    q = q_0 + (q_95 - q_0) * (psi_norm / psi_95)**alpha
    
    # Edge
    q = jnp.where(psi_norm > psi_95, q_95 + (psi_norm - psi_95) * 20.0, q)
    
    return q


def east_pressure_profile(psi_norm: jnp.ndarray, p_0: float = 1.0) -> jnp.ndarray:
    """
    EAST pressure profile.
    
    Moderate gradient (sustainable for long pulse).
    
    Args:
        psi_norm: Normalized flux
        p_0: Central pressure
        
    Returns:
        Pressure profile
    """
    # Moderate gradient
    alpha_p = 1.5
    beta_p = 1.5
    
    p = p_0 * (1 - psi_norm**alpha_p)**beta_p
    
    return jnp.maximum(p, 0.0)


def east_reference_equilibrium(nr: int = 65, nz: int = 65,
                               rmin: Optional[float] = None,
                               rmax: Optional[float] = None,
                               zmin: Optional[float] = None,
                               zmax: Optional[float] = None) -> Dict:
    """
    EAST reference equilibrium.
    
    Typical long-pulse H-mode scenario.
    
    Args:
        nr, nz: Grid resolution
        rmin, rmax, zmin, zmax: Optional grid bounds
        
    Returns:
        Benchmark dictionary
    """
    # Default grid
    if rmin is None:
        rmin = EAST_PARAMS['R0'] - EAST_PARAMS['a'] * 1.2
    if rmax is None:
        rmax = EAST_PARAMS['R0'] + EAST_PARAMS['a'] * 1.2
    if zmin is None:
        zmin = -EAST_PARAMS['a'] * EAST_PARAMS['kappa'] * 1.2
    if zmax is None:
        zmax = EAST_PARAMS['a'] * EAST_PARAMS['kappa'] * 1.2
    
    return {
        'name': 'EAST Long-Pulse Reference',
        'reference': 'Typical scenario (Wan et al., Nucl. Fusion 55, 2015)',
        'params': EAST_PARAMS,
        'grid': {
            'nr': nr,
            'nz': nz,
            'rmin': rmin,
            'rmax': rmax,
            'zmin': zmin,
            'zmax': zmax,
        },
        'profiles': {
            'q': lambda psi: east_q_profile(psi, EAST_PARAMS['q_0'], EAST_PARAMS['q_95']),
            'pressure': lambda psi: east_pressure_profile(psi, p_0=1.0),
        },
    }


def print_east_params():
    """Print EAST parameters"""
    print("=" * 60)
    print("EAST Long-Pulse Reference Parameters")
    print("=" * 60)
    print(f"Major radius R₀:        {EAST_PARAMS['R0']:.2f} m")
    print(f"Minor radius a:         {EAST_PARAMS['a']:.2f} m")
    print(f"Aspect ratio A:         {EAST_PARAMS['R0']/EAST_PARAMS['a']:.1f}")
    print(f"Toroidal field B₀:      {EAST_PARAMS['B0']:.1f} T")
    print(f"Plasma current Ip:      {EAST_PARAMS['Ip']:.1f} MA")
    print(f"Elongation κ:           {EAST_PARAMS['kappa']:.2f}")
    print(f"Triangularity δ:        {EAST_PARAMS['delta']:.2f}")
    print(f"Normalized beta β_N:    {EAST_PARAMS['beta_N']:.1f}")
    print(f"Safety factor q₉₅:      {EAST_PARAMS['q_95']:.1f}")
    print(f"Central q₀:             {EAST_PARAMS['q_0']:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    print_east_params()
    
    east_eq = east_reference_equilibrium()
    
    print(f"\nBenchmark: {east_eq['name']}")
    print(f"Reference: {east_eq['reference']}")
    
    # Test profiles
    psi_test = jnp.linspace(0, 1, 100)
    q_test = east_eq['profiles']['q'](psi_test)
    
    print(f"\nq-profile test:")
    print(f"  q(0) = {q_test[0]:.2f}")
    print(f"  q(0.5) = {q_test[50]:.2f}")
    print(f"  q(0.95) = {q_test[95]:.2f}")
    
    print("\n✅ EAST reference benchmark ready!")
