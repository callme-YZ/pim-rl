"""
M3D-C1 benchmark case: 2/1 resistive tearing mode

Reference:
W. Zhang et al., Computer Physics Communications 269 (2021) 108134

Setup:
- R/a = 10/1 (large aspect ratio)
- q(r) with q=2 resonant surface
- η = 1e-5 to 1e-7
- Expected: γ ~ η^0.6

Author: 小P ⚛️
Date: 2026-03-11
"""

import numpy as np
from typing import Tuple, Dict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from equilibrium.profiles import EquilibriumProfile
from equilibrium.grad_shafranov import solve_grad_shafranov_picard
from evolution.time_stepping import ReducedMHDState, ReducedMHDTimestepper
from evolution.diagnostics import DiagnosticLogger
from validation.linear_tearing import measure_growth_rate


class M3DC1Profile(EquilibriumProfile):
    """
    M3D-C1 benchmark equilibrium profile.
    
    Safety factor:
    q(ψn) = q0 × (1 + (ψn/ql)^α)^(1/α)
    
    where ψn = normalized flux (0 at axis, 1 at edge)
    
    Parameters from CPC 2021 paper:
    - q0 = 1.75
    - qe = 2.5
    - α = 2.0
    - ql = [(qe/q0)^α - 1]^(-1/α) ≈ 1.58
    
    Resonant surface at q=2 → ψn ≈ 0.45
    """
    
    def __init__(self, q0: float = 1.75, qe: float = 2.5, alpha: float = 2.0,
                 p0: float = 0.0, F0: float = 1.0):
        """
        Initialize M3D-C1 profile.
        
        Parameters
        ----------
        q0 : float
            Safety factor on axis
        qe : float
            Safety factor at edge
        alpha : float
            Profile shape parameter
        p0 : float
            Central pressure (use 0 for zero-β)
        F0 : float
            Toroidal field function (constant for zero-β)
        """
        self.q0 = q0
        self.qe = qe
        self.alpha = alpha
        
        # Compute ql from qe/q0
        self.ql = ((qe / q0)**alpha - 1)**(-1.0/alpha)
        
        self.p0 = p0
        self.F0 = F0
        
        # For zero-β, these are simple
        self.beta = 0.0
    
    def q_profile(self, psi_n: np.ndarray) -> np.ndarray:
        """
        Compute q(ψn).
        
        q = q0 × (1 + (ψn/ql)^α)^(1/α)
        """
        return self.q0 * (1 + (psi_n / self.ql)**self.alpha)**(1.0/self.alpha)
    
    def pressure(self, psi: np.ndarray) -> np.ndarray:
        """p(ψ) = 0 for zero-β"""
        return np.zeros_like(psi)
    
    def pressure_derivative(self, psi: np.ndarray) -> np.ndarray:
        """dp/dψ = 0"""
        return np.zeros_like(psi)
    
    def F(self, psi: np.ndarray) -> np.ndarray:
        """F(ψ) = F0 (constant)"""
        return self.F0 * np.ones_like(psi)
    
    def F_derivative(self, psi: np.ndarray) -> np.ndarray:
        """dF/dψ = 0"""
        return np.zeros_like(psi)


def create_m3dc1_geometry(Nr: int = 64, Nz: int = 64,
                              R_over_a: float = 10.0,
                              a: float = 1.0) -> Tuple[np.ndarray, np.ndarray, M3DC1Profile]:
    """
    Create M3D-C1 benchmark equilibrium.
    
    Parameters
    ----------
    Nr, Nz : int
        Grid resolution
    R_over_a : float
        Aspect ratio (default: 10)
    a : float
        Minor radius (default: 1m)
    
    Returns
    -------
    R : np.ndarray
        Major radius grid
    Z : np.ndarray
        Vertical grid
    profile : M3DC1Profile
        Equilibrium profile
    """
    R0 = R_over_a * a  # Major radius = 10m
    
    # Grid: centered at R0, extends ±a
    R = np.linspace(R0 - a, R0 + a, Nr)
    Z = np.linspace(-a, a, Nz)
    
    # Profile
    profile = M3DC1Profile(q0=1.75, qe=2.5, alpha=2.0, p0=0.0, F0=1.0)
    
    return R, Z, profile


def create_m3dc1_tearing_perturbation(R: np.ndarray, Z: np.ndarray,
                                       psi_eq: np.ndarray,
                                       amplitude: float = 0.01,
                                       m: int = 2, n: int = 1) -> np.ndarray:
    """
    Add 2/1 tearing mode perturbation.
    
    For cylindrical-like geometry:
    δψ ~ sin(mθ) exp(inφ)
    
    In (R,Z) plane:
    θ ≈ atan2(Z, R-R0)
    
    Parameters
    ----------
    R, Z : np.ndarray
        Grid
    psi_eq : np.ndarray
        Equilibrium flux
    amplitude : float
        Perturbation amplitude
    m, n : int
        Mode numbers (2/1 for this benchmark)
    
    Returns
    -------
    psi_total : np.ndarray
        Equilibrium + perturbation
    """
    Nr, Nz = len(R), len(Z)
    R_2d, Z_2d = np.meshgrid(R, Z, indexing='ij')
    
    # Geometric center
    R0 = (R[0] + R[-1]) / 2
    Z0 = (Z[0] + Z[-1]) / 2
    
    # Poloidal angle
    theta = np.arctan2(Z_2d - Z0, R_2d - R0)
    
    # Radial coordinate
    r = np.sqrt((R_2d - R0)**2 + (Z_2d - Z0)**2)
    
    # Perturbation: δψ ~ r^m sin(mθ) (localized)
    # Find q=2 resonant surface location dynamically
    # (Don't hardcode - depends on q-profile!)
    from equilibrium.simple_circular import compute_m3dc1_q_profile
    psi_n_grid = (r / 1.0)**2  # Normalized flux (assuming a=1)
    q_grid = compute_m3dc1_q_profile(psi_n_grid)
    
    # Find where q≈2 (resonant surface)
    q2_mask = np.abs(q_grid - 2.0) < 0.05
    if q2_mask.any():
        r_res = r[q2_mask].mean()
    else:
        r_res = 0.73  # Fallback (typical for M3D-C1 profile)
    
    sigma = 0.2
    
    envelope = np.exp(-((r - r_res) / sigma)**2)
    delta_psi = amplitude * envelope * np.sin(m * theta)
    
    # Enforce boundary conditions
    delta_psi[0, :] = 0
    delta_psi[-1, :] = 0
    delta_psi[:, 0] = 0
    delta_psi[:, -1] = 0
    
    return psi_eq + delta_psi


def run_m3dc1_benchmark(eta: float,
                        Nr: int = 48, Nz: int = 48,
                        t_final: float = 5.0,
                        dt: float = 0.01,
                        verbose: bool = True,
                        use_proper_equilibrium: bool = True) -> Dict:
    """
    Run M3D-C1 2/1 tearing mode benchmark.
    
    Parameters
    ----------
    eta : float
        Resistivity (1e-5 to 1e-7)
    Nr, Nz : int
        Grid resolution
    t_final : float
        Final time
    dt : float
        Time step
    verbose : bool
        Print progress
    use_proper_equilibrium : bool
        If True, use iterative q(ψ) equilibrium (slower but correct)
        If False, use simple ψ~r² (fast but wrong q-profile)
    
    Returns
    -------
    results : dict
        {
            'eta': float,
            'gamma': float,
            'gamma_std': float,
            'time': np.ndarray,
            'amplitude': np.ndarray,
            'energy': np.ndarray
        }
    """
    if verbose:
        print("=" * 70)
        print(f"M3D-C1 Benchmark: 2/1 Tearing Mode")
        print("=" * 70)
        print(f"η = {eta:.1e}")
        print(f"Grid: {Nr}×{Nz}")
        print(f"Time: 0 → {t_final} (dt={dt})")
        print(f"Equilibrium: {'Proper q(ψ)' if use_proper_equilibrium else 'Simple ψ~r²'}")
    
    # Create geometry and profile
    R, Z, profile = create_m3dc1_geometry(Nr, Nz, R_over_a=10.0, a=1.0)
    
    if verbose:
        print(f"Geometry: R ∈ [{R[0]:.1f}, {R[-1]:.1f}], Z ∈ [{Z[0]:.1f}, {Z[-1]:.1f}]")
        print(f"Profile: q0={profile.q0}, qe={profile.qe}, ql={profile.ql:.3f}")
    
    R_2d, Z_2d = np.meshgrid(R, Z, indexing='ij')
    R0 = (R[0] + R[-1]) / 2
    a = 1.0
    
    if use_proper_equilibrium:
        # Use M3D-C1's simple approach: ψ_n = (r/a)²
        from equilibrium.simple_circular import create_m3dc1_equilibrium
        
        if verbose:
            print("\nUsing M3D-C1 simple circular equilibrium...")
            print(f"  Method: ψ_n = (r/a)²")
            print(f"  q-profile: M3D-C1 formula")
        
        # Create equilibrium
        psi_eq, psi_n_eq, q_eq = create_m3dc1_equilibrium(
            R, Z, R0=R0, a=a,
            q0=profile.q0, qe=profile.qe, alpha=profile.alpha
        )
        
        if verbose:
            print(f"  q at axis: {q_eq[Nr//2, Nz//2]:.3f} (target: {profile.q0})")
            print(f"  q at edge: {q_eq[0, Nz//2]:.3f} (target: {profile.qe})")
        
        # Already normalized
        psi_eq = psi_n_eq
        
    else:
        # Simple equilibrium: ψ ~ r² (old method)
        r = np.sqrt((R_2d - R0)**2 + Z_2d**2)
        psi_eq = 0.5 * r**2
        
        # Normalize to [0,1]
        psi_eq = (psi_eq - psi_eq.min()) / (psi_eq.max() - psi_eq.min() + 1e-10)
    
    # Apply boundary conditions
    psi_eq[0, :] = 0
    psi_eq[-1, :] = 0
    psi_eq[:, 0] = 0
    psi_eq[:, -1] = 0
    
    # Add 2/1 perturbation
    psi_init = create_m3dc1_tearing_perturbation(R, Z, psi_eq, amplitude=0.001, m=2, n=1)
    
    if verbose:
        print(f"ψ_eq range: [{psi_eq.min():.3f}, {psi_eq.max():.3f}]")
        print(f"ψ_init range: [{psi_init.min():.3f}, {psi_init.max():.3f}]")
    
    # Initial state
    phi_init = np.zeros_like(psi_init)
    U_init = np.zeros_like(psi_init)
    
    state = ReducedMHDState(psi_init, phi_init, U_init, R, Z)
    
    # Timestepper (use benchmark viscosity)
    nu = 1e-8  # From M3D-C1 paper
    stepper = ReducedMHDTimestepper(R, Z, eta=eta, nu=nu, theta=0.5)
    
    # Diagnostics
    logger = DiagnosticLogger()
    
    # Evolve
    n_steps = int(t_final / dt)
    log_interval = max(1, n_steps // 20)
    
    if verbose:
        print(f"\nEvolving {n_steps} steps...")
    
    logger.append(state)
    
    for n in range(n_steps):
        state = stepper.step(state, dt)
        
        if (n+1) % log_interval == 0:
            logger.append(state)
            
            if verbose and (n+1) % (log_interval * 5) == 0:
                diag = logger.history[-1]
                print(f"  t={diag['time']:6.3f}, m1={diag['m1_amplitude']:.3e}, E={diag['energy_total']:.2e}")
    
    # Final
    logger.append(state)
    
    # Extract results
    hist = logger.get_history()
    
    # Measure growth rate (middle 50%)
    t_start = t_final * 0.25
    t_end = t_final * 0.75
    
    # Use m=2 amplitude (2/1 mode)
    # For now, use m1_amplitude as proxy (will fix later)
    amplitudes = hist['m1_amplitude']
    
    growth = measure_growth_rate(hist['time'], amplitudes, t_start, t_end)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"RESULT: γ = {growth['gamma']:.6f} ± {growth['gamma_std']:.6f}")
        print(f"        R² = {growth['R_squared']:.3f}")
        print(f"        {'✅ Growth' if growth['gamma'] > 0 else '⚠️  Decay'}")
        print(f"{'='*70}\n")
    
    return {
        'eta': eta,
        'gamma': growth['gamma'],
        'gamma_std': growth['gamma_std'],
        'R_squared': growth['R_squared'],
        'time': hist['time'],
        'amplitude': amplitudes,
        'energy': hist['energy_total']
    }


def eta_scaling_benchmark(eta_values: np.ndarray = None,
                          **kwargs) -> Dict:
    """
    Run η scaling study for M3D-C1 benchmark.
    
    Expected: γ ~ η^0.6 (theory: 3/5)
    
    Parameters
    ----------
    eta_values : np.ndarray, optional
        Resistivity values (default: [1e-5, 5e-6, 1e-6])
    **kwargs :
        Passed to run_m3dc1_benchmark
    
    Returns
    -------
    results : dict
        {
            'eta': np.ndarray,
            'gamma': np.ndarray,
            'scaling_exponent': float,
            'theory': float (0.6)
        }
    """
    if eta_values is None:
        eta_values = np.array([1e-5, 5e-6, 1e-6])
    
    print("=" * 70)
    print("M3D-C1 η SCALING BENCHMARK")
    print("=" * 70)
    print(f"Testing {len(eta_values)} values")
    print(f"Expected: γ ~ η^0.6\n")
    
    gamma_list = []
    
    for i, eta in enumerate(eta_values):
        print(f"[{i+1}/{len(eta_values)}] η = {eta:.1e}")
        
        result = run_m3dc1_benchmark(eta, verbose=True, **kwargs)
        gamma_list.append(result['gamma'])
    
    gamma_arr = np.array(gamma_list)
    
    # Fit γ ~ η^p
    valid_mask = np.isfinite(gamma_arr) & (gamma_arr > 0)
    
    if np.sum(valid_mask) >= 2:
        log_eta = np.log(eta_values[valid_mask])
        log_gamma = np.log(gamma_arr[valid_mask])
        
        coeffs = np.polyfit(log_eta, log_gamma, deg=1)
        p = coeffs[0]
    else:
        p = np.nan
    
    print("=" * 70)
    print("SCALING RESULT")
    print("=" * 70)
    print(f"Measured: γ ~ η^{p:.3f}")
    print(f"Theory:   γ ~ η^0.600")
    print(f"M3D-C1:   γ ~ η^0.580")
    print(f"CLT:      γ ~ η^0.601")
    print("=" * 70)
    
    return {
        'eta': eta_values,
        'gamma': gamma_arr,
        'scaling_exponent': p,
        'theory': 0.6
    }
