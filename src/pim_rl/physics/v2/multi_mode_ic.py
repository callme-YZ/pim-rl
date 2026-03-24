"""
Unified Multi-Mode IC API (Issue #27 Phase 3)

Provides single interface for multiple instability modes:
- Tearing mode (m=1, resistive)
- Kink mode (m=1, current-driven ideal)
- Interchange mode (m=2-4, pressure-driven)
- Ballooning mode (existing, m>>1)

Author: 小P ⚛️
Date: 2026-03-24
"""

import jax.numpy as jnp
from typing import Tuple, Literal

# Import mode-specific functions
from .tearing_ic import create_tearing_ic as _create_tearing_ic
from .kink_ic import create_kink_ic as _create_kink_ic
from .interchange_ic import create_interchange_ic as _create_interchange_ic
from .ballooning_ic_v2 import ballooning_mode_ic_v2 as _create_ballooning_ic


ModeType = Literal['tearing', 'kink', 'interchange', 'ballooning']


def create_multi_mode_ic(
    mode: ModeType,
    nr: int = 32,
    ntheta: int = 64,
    **mode_kwargs
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Unified interface for creating instability mode ICs.
    
    Parameters
    ----------
    mode : str
        Instability mode type:
        - 'tearing': m=1 resistive reconnection (Harris sheet)
        - 'kink': m=1 current-driven ideal (q≈1)
        - 'interchange': m=2-4 pressure-driven (Rayleigh-Taylor)
        - 'ballooning': m>>1 pressure+curvature (existing)
    nr, ntheta : int
        Grid resolution
    **mode_kwargs : dict
        Mode-specific parameters (see individual create_*_ic functions)
        
    Returns
    -------
    psi : array, shape (nr, ntheta)
        Flux function
    phi : array, shape (nr, ntheta)
        Stream function
        
    Examples
    --------
    >>> # Tearing mode (Harris sheet)
    >>> psi, phi = create_multi_mode_ic('tearing', nr=32, ntheta=64, 
    ...                                  B0=1.0, L=0.5, eps=0.01)
    
    >>> # Kink mode (q≈1)
    >>> psi, phi = create_multi_mode_ic('kink', nr=32, ntheta=64,
    ...                                  j0=2.0, r_res=0.5, eps=0.01)
    
    >>> # Interchange mode (pressure bump)
    >>> psi, phi = create_multi_mode_ic('interchange', nr=32, ntheta=64,
    ...                                  p0=1.0, r_peak=0.6, m=2, eps=0.01)
    
    >>> # Ballooning mode (existing)
    >>> psi, phi = create_multi_mode_ic('ballooning', nr=32, ntheta=64)
    
    Notes
    -----
    This provides a unified API for benchmark experiments and multi-mode
    RL training. Each mode has different physics and growth characteristics.
    
    See Also
    --------
    create_tearing_ic : Tearing mode details
    create_kink_ic : Kink mode details
    create_interchange_ic : Interchange mode details
    create_ballooning_ic : Ballooning mode details
    """
    if mode == 'tearing':
        return _create_tearing_ic(nr=nr, ntheta=ntheta, **mode_kwargs)
    elif mode == 'kink':
        return _create_kink_ic(nr=nr, ntheta=ntheta, **mode_kwargs)
    elif mode == 'interchange':
        return _create_interchange_ic(nr=nr, ntheta=ntheta, **mode_kwargs)
    elif mode == 'ballooning':
        # Ballooning has different signature (requires metric, fa)
        # For now, create simple placeholder
        # TODO: Integrate with PyTokEq for full ballooning support
        r = jnp.linspace(0, 1, nr)
        theta = jnp.linspace(0, 2*jnp.pi, ntheta, endpoint=False)
        R, Theta = jnp.meshgrid(r, theta, indexing='ij')
        # Placeholder: axisymmetric equilibrium
        psi = R**2
        phi = jnp.zeros_like(psi)
        return psi, phi
    else:
        raise ValueError(f"Unknown mode: {mode}. "
                        f"Must be one of: 'tearing', 'kink', 'interchange', 'ballooning'")


def get_mode_info(mode: ModeType) -> dict:
    """
    Get physics information for a mode.
    
    Parameters
    ----------
    mode : str
        Mode type
        
    Returns
    -------
    info : dict
        Physics characteristics:
        - 'driver': Physical driver (current, pressure, etc.)
        - 'm_typical': Typical mode numbers
        - 'growth_formula': Growth rate formula
        - 'resistivity_needed': Whether resistivity is required
        - 'reference': Key paper reference
        
    Examples
    --------
    >>> info = get_mode_info('kink')
    >>> print(info['driver'])
    'Current (q≈1)'
    >>> print(info['growth_formula'])
    'γ ≈ 0.3 V_A / R₀'
    """
    mode_database = {
        'tearing': {
            'driver': 'Current sheet (resistive reconnection)',
            'm_typical': [1, 2],
            'growth_formula': 'γ = (η/τ_A)^(3/5) ω_A (FKR 1963)',
            'resistivity_needed': True,
            'reference': 'Furth, Killeen, Rosenbluth (1963)',
            'typical_growth': 'Slow (γ ~ 1-10 s⁻¹)',
        },
        'kink': {
            'driver': 'Current (q≈1)',
            'm_typical': [1],
            'growth_formula': 'γ ≈ 0.3 V_A / R₀ (Freidberg 1987)',
            'resistivity_needed': False,
            'reference': 'Kadomtsev (1975), Freidberg (1987)',
            'typical_growth': 'Fast (γ ~ V_A/R₀, ideal MHD)',
        },
        'interchange': {
            'driver': 'Pressure gradient (Rayleigh-Taylor)',
            'm_typical': [2, 3, 4],
            'growth_formula': 'γ ≈ √(p₀/ρ) / L_p (Freidberg Ch 9)',
            'resistivity_needed': False,
            'reference': 'Freidberg (1987), Wesson (2011)',
            'typical_growth': 'Medium (γ ~ 5-10 s⁻¹)',
        },
        'ballooning': {
            'driver': 'Pressure + bad curvature',
            'm_typical': [10, 20, 50],  # High-n
            'growth_formula': 'γ ~ ω_A (ideal MHD)',
            'resistivity_needed': False,
            'reference': 'Connor et al. (1978)',
            'typical_growth': 'Fast (γ ~ ω_A)',
        },
    }
    
    if mode not in mode_database:
        raise ValueError(f"Unknown mode: {mode}")
    
    return mode_database[mode]


def list_available_modes() -> list[str]:
    """
    List all available instability modes.
    
    Returns
    -------
    modes : list of str
        Available mode names
    """
    return ['tearing', 'kink', 'interchange', 'ballooning']


def get_default_parameters(mode: ModeType) -> dict:
    """
    Get default parameters for a mode.
    
    Parameters
    ----------
    mode : str
        Mode type
        
    Returns
    -------
    params : dict
        Default parameters for create_multi_mode_ic
        
    Examples
    --------
    >>> params = get_default_parameters('kink')
    >>> psi, phi = create_multi_mode_ic('kink', **params)
    """
    defaults = {
        'tearing': {
            'B0': 1.0,
            'lam': 0.1,  # Current sheet width
            'eps': 0.01,
        },
        'kink': {
            'r_res': 0.5,
            'j0': 2.0,
            'a': 0.8,
            'eps': 0.01,
            'B0': 1.0,
        },
        'interchange': {
            'p0': 1.0,
            'r_peak': 0.6,
            'width': 0.15,
            'eps': 0.01,
            'm': 2,
            'B0': 1.0,
        },
        'ballooning': {
            # Use ballooning_ic_v2 defaults
        },
    }
    
    if mode not in defaults:
        raise ValueError(f"Unknown mode: {mode}")
    
    return defaults[mode]


# Convenience functions for common use cases

def create_benchmark_suite(
    nr: int = 32,
    ntheta: int = 64
) -> dict[str, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Create ICs for all modes (benchmark suite).
    
    Parameters
    ----------
    nr, ntheta : int
        Grid resolution
        
    Returns
    -------
    ics : dict
        Dictionary mapping mode names to (psi, phi) tuples
        
    Examples
    --------
    >>> ics = create_benchmark_suite(nr=32, ntheta=64)
    >>> tearing_psi, tearing_phi = ics['tearing']
    >>> kink_psi, kink_phi = ics['kink']
    
    Notes
    -----
    This is useful for multi-mode validation experiments where you
    want to test RL on all instability types.
    """
    modes = list_available_modes()
    ics = {}
    
    for mode in modes:
        params = get_default_parameters(mode)
        ics[mode] = create_multi_mode_ic(mode, nr=nr, ntheta=ntheta, **params)
    
    return ics


def compare_modes_info() -> str:
    """
    Print comparison table of all modes.
    
    Returns
    -------
    table : str
        Formatted comparison table
    """
    modes = list_available_modes()
    
    lines = []
    lines.append("=" * 80)
    lines.append("Instability Mode Comparison")
    lines.append("=" * 80)
    lines.append("")
    
    for mode in modes:
        info = get_mode_info(mode)
        lines.append(f"Mode: {mode.upper()}")
        lines.append(f"  Driver:       {info['driver']}")
        lines.append(f"  m (typical):  {info['m_typical']}")
        lines.append(f"  Growth rate:  {info['growth_formula']}")
        lines.append(f"  Resistivity:  {'Required' if info['resistivity_needed'] else 'Not needed'}")
        lines.append(f"  Reference:    {info['reference']}")
        lines.append(f"  Growth speed: {info['typical_growth']}")
        lines.append("")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)


# For backward compatibility with existing code
def create_tearing_ic(*args, **kwargs):
    """Alias for backward compatibility."""
    return _create_tearing_ic(*args, **kwargs)

def create_kink_ic(*args, **kwargs):
    """Alias for backward compatibility."""
    return _create_kink_ic(*args, **kwargs)

def create_interchange_ic(*args, **kwargs):
    """Alias for backward compatibility."""
    return _create_interchange_ic(*args, **kwargs)

def create_ballooning_ic(*args, **kwargs):
    """Alias for backward compatibility."""
    return _create_ballooning_ic(*args, **kwargs)
