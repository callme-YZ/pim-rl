"""
ITER Baseline Scenario Equilibrium

Reference: ITER Physics Basis, Nucl. Fusion 47 (2007)

Parameters:
- Major radius R0 = 6.2 m
- Minor radius a = 2.0 m
- Aspect ratio A = 3.1
- Plasma current Ip = 15 MA
- Toroidal field B0 = 5.3 T
- β_N ~ 1.8
- q_95 ~ 3.0

Issue #13

Author: 小P ⚛️
Date: 2026-03-24
"""

import jax.numpy as jnp
from typing import Dict, Optional


# ITER baseline parameters (from Physics Basis 2007)
ITER_PARAMS = {
    'R0': 6.2,      # Major radius [m]
    'a': 2.0,       # Minor radius [m]
    'B0': 5.3,      # Toroidal field [T]
    'Ip': 15.0,     # Plasma current [MA]
    'kappa': 1.7,   # Elongation
    'delta': 0.33,  # Triangularity
    'beta_N': 1.8,  # Normalized beta
    'q_95': 3.0,    # Safety factor at 95% flux
    'q_0': 1.0,     # Central safety factor
}


def iter_q_profile(psi_norm: jnp.ndarray, q_0: float = 1.0, q_95: float = 3.0) -> jnp.ndarray:
    """
    ITER-like q-profile.
    
    Uses parabolic-like profile:
    q(ψ) = q_0 + (q_95 - q_0) * ψ^α
    
    where α ~ 2 gives typical ITER shape.
    
    Args:
        psi_norm: Normalized poloidal flux (0 to 1)
        q_0: Central safety factor
        q_95: Edge safety factor at ψ=0.95
        
    Returns:
        q-profile
    """
    # Typical ITER q-profile shape: α ~ 2
    alpha = 2.0
    
    # q(ψ) = q_0 + (q_95 - q_0) * (ψ/0.95)^α
    # This ensures q(0.95) = q_95
    psi_95 = 0.95
    q = q_0 + (q_95 - q_0) * (psi_norm / psi_95)**alpha
    
    # Clamp for ψ > 0.95
    q = jnp.where(psi_norm > psi_95, q_95 + (psi_norm - psi_95) * 10.0, q)
    
    return q


def iter_pressure_profile(psi_norm: jnp.ndarray, p_0: float = 1.0) -> jnp.ndarray:
    """
    ITER-like pressure profile.
    
    Peaked profile: p(ψ) = p_0 * (1 - ψ^α_p)^β_p
    
    Typical ITER: α_p ~ 1.5, β_p ~ 2
    
    Args:
        psi_norm: Normalized poloidal flux
        p_0: Central pressure
        
    Returns:
        Pressure profile
    """
    alpha_p = 1.5
    beta_p = 2.0
    
    p = p_0 * (1 - psi_norm**alpha_p)**beta_p
    
    return jnp.maximum(p, 0.0)


def iter_current_density_profile(psi_norm: jnp.ndarray, j_0: float = 1.0) -> jnp.ndarray:
    """
    ITER-like toroidal current density profile.
    
    Relation: j_φ ~ dp/dψ + f·df/dψ (Grad-Shafranov)
    
    Simplified peaked profile for ITER.
    
    Args:
        psi_norm: Normalized flux
        j_0: Central current density
        
    Returns:
        Current density profile
    """
    # Peaked profile
    alpha_j = 1.0
    beta_j = 1.5
    
    j = j_0 * (1 - psi_norm**alpha_j)**beta_j
    
    return jnp.maximum(j, 0.0)


def iter_baseline_equilibrium(nr: int = 65, nz: int = 65,
                              rmin: Optional[float] = None,
                              rmax: Optional[float] = None,
                              zmin: Optional[float] = None,
                              zmax: Optional[float] = None) -> Dict:
    """
    Generate ITER baseline scenario equilibrium.
    
    Returns standard ITER parameters and profile functions.
    
    Args:
        nr: Radial grid points
        nz: Vertical grid points
        rmin, rmax, zmin, zmax: Optional grid bounds (default: ITER geometry)
        
    Returns:
        Dictionary with:
        - params: ITER_PARAMS
        - grid: (nr, nz, rmin, rmax, zmin, zmax)
        - profiles: q, p, j profile functions
        
    Example:
    -------
    >>> iter_eq = iter_baseline_equilibrium()
    >>> psi_norm = jnp.linspace(0, 1, 100)
    >>> q_profile = iter_eq['profiles']['q'](psi_norm)
    >>> print(f"q_0 = {q_profile[0]:.2f}, q_95 = {q_profile[95]:.2f}")
    """
    # Default grid bounds (ITER geometry)
    if rmin is None:
        rmin = ITER_PARAMS['R0'] - ITER_PARAMS['a'] * 1.2  # 3.8 m
    if rmax is None:
        rmax = ITER_PARAMS['R0'] + ITER_PARAMS['a'] * 1.2  # 8.6 m
    if zmin is None:
        zmin = -ITER_PARAMS['a'] * ITER_PARAMS['kappa'] * 1.2  # -4.08 m
    if zmax is None:
        zmax = ITER_PARAMS['a'] * ITER_PARAMS['kappa'] * 1.2   # +4.08 m
    
    # Return benchmark data
    return {
        'name': 'ITER Baseline Scenario',
        'reference': 'ITER Physics Basis, Nucl. Fusion 47 (2007)',
        'params': ITER_PARAMS,
        'grid': {
            'nr': nr,
            'nz': nz,
            'rmin': rmin,
            'rmax': rmax,
            'zmin': zmin,
            'zmax': zmax,
        },
        'profiles': {
            'q': lambda psi: iter_q_profile(psi, ITER_PARAMS['q_0'], ITER_PARAMS['q_95']),
            'pressure': lambda psi: iter_pressure_profile(psi, p_0=1.0),
            'current': lambda psi: iter_current_density_profile(psi, j_0=1.0),
        },
    }


def validate_iter_equilibrium(eq_solution: Dict) -> Dict:
    """
    Validate ITER equilibrium against reference values.
    
    Args:
        eq_solution: Solved equilibrium (must have q_profile, beta, etc.)
        
    Returns:
        Validation metrics
    """
    metrics = {}
    
    # Check q_95
    if 'q_profile' in eq_solution:
        q = eq_solution['q_profile']
        psi_norm = eq_solution.get('psi_norm', jnp.linspace(0, 1, len(q)))
        
        # Find q at ψ=0.95
        idx_95 = jnp.argmin(jnp.abs(psi_norm - 0.95))
        q_95_computed = q[idx_95]
        q_95_ref = ITER_PARAMS['q_95']
        
        metrics['q_95'] = {
            'computed': float(q_95_computed),
            'reference': q_95_ref,
            'error': abs(q_95_computed - q_95_ref) / q_95_ref,
            'pass': abs(q_95_computed - q_95_ref) / q_95_ref < 0.1  # 10% tolerance
        }
    
    # Check β_N
    if 'beta_N' in eq_solution:
        beta_N_computed = eq_solution['beta_N']
        beta_N_ref = ITER_PARAMS['beta_N']
        
        metrics['beta_N'] = {
            'computed': float(beta_N_computed),
            'reference': beta_N_ref,
            'error': abs(beta_N_computed - beta_N_ref) / beta_N_ref,
            'pass': abs(beta_N_computed - beta_N_ref) / beta_N_ref < 0.2  # 20% tolerance
        }
    
    return metrics


def print_iter_params():
    """Print ITER baseline parameters"""
    print("=" * 60)
    print("ITER Baseline Scenario Parameters")
    print("=" * 60)
    print(f"Major radius R₀:        {ITER_PARAMS['R0']:.1f} m")
    print(f"Minor radius a:         {ITER_PARAMS['a']:.1f} m")
    print(f"Aspect ratio A:         {ITER_PARAMS['R0']/ITER_PARAMS['a']:.1f}")
    print(f"Toroidal field B₀:      {ITER_PARAMS['B0']:.1f} T")
    print(f"Plasma current Ip:      {ITER_PARAMS['Ip']:.0f} MA")
    print(f"Elongation κ:           {ITER_PARAMS['kappa']:.2f}")
    print(f"Triangularity δ:        {ITER_PARAMS['delta']:.2f}")
    print(f"Normalized beta β_N:    {ITER_PARAMS['beta_N']:.1f}")
    print(f"Safety factor q₉₅:      {ITER_PARAMS['q_95']:.1f}")
    print(f"Central q₀:             {ITER_PARAMS['q_0']:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    print_iter_params()
    
    # Generate benchmark
    iter_eq = iter_baseline_equilibrium()
    
    print(f"\nBenchmark: {iter_eq['name']}")
    print(f"Reference: {iter_eq['reference']}")
    
    # Test profiles
    psi_test = jnp.linspace(0, 1, 100)
    q_test = iter_eq['profiles']['q'](psi_test)
    
    print(f"\nq-profile test:")
    print(f"  q(0) = {q_test[0]:.2f}")
    print(f"  q(0.5) = {q_test[50]:.2f}")
    print(f"  q(0.95) = {q_test[95]:.2f}")
    
    print("\n✅ ITER baseline benchmark ready!")
