"""
DIII-D H-mode Benchmark Equilibrium

Reference: Typical DIII-D H-mode parameters
(Based on Lao et al., Nucl. Fusion 30 (1990) and later publications)

Parameters (typical H-mode):
- Major radius R0 = 1.67 m
- Minor radius a = 0.67 m  
- Aspect ratio A = 2.5
- B0 = 2.0 T
- Ip = 1.2 MA
- κ = 1.8 (high elongation)
- δ = 0.4 (moderate triangularity)
- q_95 ~ 4.0
- β_N ~ 2.5 (H-mode)

Note: These are representative H-mode values.
For specific shot data, see DIII-D database.

Issue #13

Author: 小P ⚛️
Date: 2026-03-24
"""

import jax.numpy as jnp
from typing import Dict, Optional


DIIID_PARAMS = {
    'R0': 1.67,      # Major radius [m]
    'a': 0.67,       # Minor radius [m]
    'B0': 2.0,       # Toroidal field [T]
    'Ip': 1.2,       # Plasma current [MA]
    'kappa': 1.8,    # Elongation (high for H-mode)
    'delta': 0.4,    # Triangularity
    'beta_N': 2.5,   # Normalized beta (H-mode)
    'q_95': 4.0,     # Safety factor at 95% flux
    'q_0': 1.1,      # Central safety factor
}


def diiid_q_profile(psi_norm: jnp.ndarray, q_0: float = 1.1, q_95: float = 4.0) -> jnp.ndarray:
    """
    DIII-D H-mode q-profile.
    
    Slightly broader than ITER (lower current density).
    
    Args:
        psi_norm: Normalized flux
        q_0: Central q
        q_95: Edge q at ψ=0.95
        
    Returns:
        q-profile
    """
    # H-mode profile: α ~ 1.5 (broader than ITER)
    alpha = 1.5
    psi_95 = 0.95
    
    q = q_0 + (q_95 - q_0) * (psi_norm / psi_95)**alpha
    
    # Edge extrapolation
    q = jnp.where(psi_norm > psi_95, q_95 + (psi_norm - psi_95) * 15.0, q)
    
    return q


def diiid_pressure_profile(psi_norm: jnp.ndarray, p_0: float = 1.0) -> jnp.ndarray:
    """
    DIII-D H-mode pressure profile.
    
    Pedestal + core gradient typical of H-mode.
    
    Args:
        psi_norm: Normalized flux
        p_0: Central pressure
        
    Returns:
        Pressure with pedestal
    """
    # Core profile
    alpha_core = 2.0
    p_core = p_0 * (1 - psi_norm**alpha_core)**1.5
    
    # Simple pedestal (ψ > 0.9)
    pedestal_height = 0.2 * p_0
    pedestal_width = 0.05
    
    # Tanh pedestal
    psi_ped = 0.95
    pedestal = pedestal_height * 0.5 * (1 - jnp.tanh((psi_norm - psi_ped) / pedestal_width))
    
    p = p_core + pedestal
    
    return jnp.maximum(p, 0.0)


def diiid_hmode_equilibrium(nr: int = 65, nz: int = 65,
                            rmin: Optional[float] = None,
                            rmax: Optional[float] = None,
                            zmin: Optional[float] = None,
                            zmax: Optional[float] = None) -> Dict:
    """
    DIII-D H-mode benchmark equilibrium.
    
    Representative H-mode parameters.
    
    Args:
        nr, nz: Grid resolution
        rmin, rmax, zmin, zmax: Optional grid bounds
        
    Returns:
        Benchmark dictionary
    """
    # Default grid bounds
    if rmin is None:
        rmin = DIIID_PARAMS['R0'] - DIIID_PARAMS['a'] * 1.2
    if rmax is None:
        rmax = DIIID_PARAMS['R0'] + DIIID_PARAMS['a'] * 1.2
    if zmin is None:
        zmin = -DIIID_PARAMS['a'] * DIIID_PARAMS['kappa'] * 1.2
    if zmax is None:
        zmax = DIIID_PARAMS['a'] * DIIID_PARAMS['kappa'] * 1.2
    
    return {
        'name': 'DIII-D H-mode Reference',
        'reference': 'Typical H-mode (Lao et al., Nucl. Fusion 30, 1990)',
        'params': DIIID_PARAMS,
        'grid': {
            'nr': nr,
            'nz': nz,
            'rmin': rmin,
            'rmax': rmax,
            'zmin': zmin,
            'zmax': zmax,
        },
        'profiles': {
            'q': lambda psi: diiid_q_profile(psi, DIIID_PARAMS['q_0'], DIIID_PARAMS['q_95']),
            'pressure': lambda psi: diiid_pressure_profile(psi, p_0=1.0),
        },
    }


def print_diiid_params():
    """Print DIII-D parameters"""
    print("=" * 60)
    print("DIII-D H-mode Reference Parameters")
    print("=" * 60)
    print(f"Major radius R₀:        {DIIID_PARAMS['R0']:.2f} m")
    print(f"Minor radius a:         {DIIID_PARAMS['a']:.2f} m")
    print(f"Aspect ratio A:         {DIIID_PARAMS['R0']/DIIID_PARAMS['a']:.1f}")
    print(f"Toroidal field B₀:      {DIIID_PARAMS['B0']:.1f} T")
    print(f"Plasma current Ip:      {DIIID_PARAMS['Ip']:.1f} MA")
    print(f"Elongation κ:           {DIIID_PARAMS['kappa']:.2f}")
    print(f"Triangularity δ:        {DIIID_PARAMS['delta']:.2f}")
    print(f"Normalized beta β_N:    {DIIID_PARAMS['beta_N']:.1f}")
    print(f"Safety factor q₉₅:      {DIIID_PARAMS['q_95']:.1f}")
    print(f"Central q₀:             {DIIID_PARAMS['q_0']:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    print_diiid_params()
    
    diiid_eq = diiid_hmode_equilibrium()
    
    print(f"\nBenchmark: {diiid_eq['name']}")
    print(f"Reference: {diiid_eq['reference']}")
    
    # Test profiles
    psi_test = jnp.linspace(0, 1, 100)
    q_test = diiid_eq['profiles']['q'](psi_test)
    
    print(f"\nq-profile test:")
    print(f"  q(0) = {q_test[0]:.2f}")
    print(f"  q(0.5) = {q_test[50]:.2f}")
    print(f"  q(0.95) = {q_test[95]:.2f}")
    
    print("\n✅ DIII-D H-mode benchmark ready!")
