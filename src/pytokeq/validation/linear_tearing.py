"""
Linear tearing mode validation

Tests:
- Growth rate measurement
- Scaling with η: γ ~ η^p
- Mode structure
- Comparison with theory (FKR 1963, Furth-Killeen-Rosenbluth)

Author: 小P ⚛️
Date: 2026-03-11
"""

import numpy as np
from typing import Tuple, Dict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from equilibrium.profiles import ZeroBetaProfile, EquilibriumProfile
from equilibrium.grad_shafranov import solve_grad_shafranov_picard, create_initial_guess_soloviev
from evolution.time_stepping import ReducedMHDState, ReducedMHDTimestepper
from evolution.diagnostics import DiagnosticLogger


def create_tearing_equilibrium(R: np.ndarray, Z: np.ndarray,
                                 current_width: float = 0.1,
                                 B0: float = 1.0) -> Tuple[np.ndarray, EquilibriumProfile]:
    """
    Create equilibrium with current sheet (tearing-unstable).
    
    Uses Harris sheet profile:
    Bz ~ tanh(r/a)
    
    For simplified 2D case, create ψ with current sheet.
    
    Parameters
    ----------
    R, Z : np.ndarray
        Grid
    current_width : float
        Width of current sheet
    B0 : float
        Background field strength
    
    Returns
    -------
    psi_eq : np.ndarray
        Equilibrium flux
    profile : EquilibriumProfile
        Equilibrium profile (zero-β)
    """
    Nr, Nz = len(R), len(Z)
    R_2d, Z_2d = np.meshgrid(R, Z, indexing='ij')
    
    # Current sheet centered at R0
    R0 = (R[0] + R[-1]) / 2
    a = current_width
    
    # ψ for Harris-like sheet
    # Simple form: ψ ~ (R-R0) tanh((R-R0)/a)
    psi_eq = (R_2d - R0) * np.tanh((R_2d - R0) / a)
    psi_eq *= B0
    
    # Apply boundary conditions
    psi_eq[0, :] = 0
    psi_eq[-1, :] = 0
    psi_eq[:, 0] = 0
    psi_eq[:, -1] = 0
    
    # Use zero-β profile (no pressure gradient)
    profile = ZeroBetaProfile(F0=1.0)
    
    return psi_eq, profile


def add_tearing_perturbation(psi_eq: np.ndarray,
                             R: np.ndarray, Z: np.ndarray,
                             amplitude: float = 0.01,
                             m: int = 1, n: int = 1) -> np.ndarray:
    """
    Add tearing mode perturbation to equilibrium.
    
    δψ = ε sin(m π r / L_r) sin(n π z / L_z)
    
    Parameters
    ----------
    psi_eq : np.ndarray
        Equilibrium flux
    R, Z : np.ndarray
        Grid
    amplitude : float
        Perturbation amplitude
    m, n : int
        Mode numbers
    
    Returns
    -------
    psi_total : np.ndarray
        Equilibrium + perturbation
    """
    Nr, Nz = len(R), len(Z)
    R_2d, Z_2d = np.meshgrid(R, Z, indexing='ij')
    
    L_R = R[-1] - R[0]
    L_Z = Z[-1] - Z[0]
    
    # Perturbation
    delta_psi = amplitude * np.sin(m * np.pi * (R_2d - R[0]) / L_R) * \
                            np.sin(n * np.pi * (Z_2d - Z[0]) / L_Z)
    
    # Enforce boundaries
    delta_psi[0, :] = 0
    delta_psi[-1, :] = 0
    delta_psi[:, 0] = 0
    delta_psi[:, -1] = 0
    
    return psi_eq + delta_psi


def measure_growth_rate(time: np.ndarray, 
                        amplitude: np.ndarray,
                        t_start: float = None,
                        t_end: float = None) -> Dict:
    """
    Measure exponential growth rate from amplitude(t).
    
    Fits: A(t) = A0 exp(γt)
    or: ln(A) = ln(A0) + γt
    
    Parameters
    ----------
    time : np.ndarray
        Time points
    amplitude : np.ndarray
        Mode amplitude at each time
    t_start, t_end : float, optional
        Time window for fit (default: full range)
    
    Returns
    -------
    result : dict
        {
            'gamma': float,
            'gamma_std': float,
            'A0': float,
            'R_squared': float
        }
    """
    # Select time window
    if t_start is None:
        t_start = time[0]
    if t_end is None:
        t_end = time[-1]
    
    mask = (time >= t_start) & (time <= t_end)
    t_fit = time[mask]
    A_fit = amplitude[mask]
    
    # Remove zeros/negatives
    positive_mask = A_fit > 1e-15
    t_fit = t_fit[positive_mask]
    A_fit = A_fit[positive_mask]
    
    if len(t_fit) < 3:
        return {
            'gamma': np.nan,
            'gamma_std': np.nan,
            'A0': np.nan,
            'R_squared': 0.0
        }
    
    # Log-linear fit
    log_A = np.log(A_fit)
    
    # Linear regression
    coeffs = np.polyfit(t_fit, log_A, deg=1)
    gamma = coeffs[0]  # Slope
    log_A0 = coeffs[1]  # Intercept
    A0 = np.exp(log_A0)
    
    # R-squared
    log_A_fit = gamma * t_fit + log_A0
    ss_res = np.sum((log_A - log_A_fit)**2)
    ss_tot = np.sum((log_A - log_A.mean())**2)
    R_squared = 1 - ss_res / (ss_tot + 1e-15)
    
    # Standard error on gamma
    residuals = log_A - log_A_fit
    mse = np.sum(residuals**2) / (len(t_fit) - 2)
    var_t = np.sum((t_fit - t_fit.mean())**2)
    gamma_std = np.sqrt(mse / var_t) if var_t > 0 else np.nan
    
    return {
        'gamma': gamma,
        'gamma_std': gamma_std,
        'A0': A0,
        'R_squared': R_squared
    }


def run_linear_tearing_test(R: np.ndarray, Z: np.ndarray,
                            eta: float, nu: float,
                            t_final: float = 2.0,
                            dt: float = 0.01,
                            perturbation_amp: float = 0.001,
                            verbose: bool = True) -> Dict:
    """
    Run single linear tearing mode evolution.
    
    Parameters
    ----------
    R, Z : np.ndarray
        Grid
    eta : float
        Resistivity
    nu : float
        Viscosity
    t_final : float
        Final time
    dt : float
        Time step
    perturbation_amp : float
        Initial perturbation amplitude
    verbose : bool
        Print progress
    
    Returns
    -------
    results : dict
        {
            'time': np.ndarray,
            'amplitude': np.ndarray,
            'gamma': float,
            'gamma_std': float,
            'energy': np.ndarray,
            'div_B': np.ndarray
        }
    """
    if verbose:
        print(f"Running linear tearing test: η={eta:.1e}, ν={nu:.1e}")
    
    # Create equilibrium
    psi_eq, profile = create_tearing_equilibrium(R, Z, current_width=0.1)
    
    # Add perturbation
    psi_init = add_tearing_perturbation(psi_eq, R, Z, amplitude=perturbation_amp, m=1, n=1)
    
    # Initial state (no flow)
    phi_init = np.zeros_like(psi_init)
    U_init = np.zeros_like(psi_init)
    
    state = ReducedMHDState(psi_init, phi_init, U_init, R, Z)
    
    # Timestepper
    stepper = ReducedMHDTimestepper(R, Z, eta=eta, nu=nu, theta=0.5)
    
    # Diagnostics
    logger = DiagnosticLogger()
    
    # Evolve
    n_steps = int(t_final / dt)
    log_interval = max(1, n_steps // 20)  # Log ~20 points
    
    logger.append(state)
    
    for n in range(n_steps):
        state = stepper.step(state, dt)
        
        if (n+1) % log_interval == 0:
            logger.append(state)
            
            if verbose and (n+1) % (log_interval * 5) == 0:
                diag = logger.history[-1]
                print(f"  t={diag['time']:.3f}, m1={diag['m1_amplitude']:.3e}, E={diag['energy_total']:.2e}")
    
    # Final log
    logger.append(state)
    
    # Extract results
    hist = logger.get_history()
    
    # Measure growth rate (use middle 50% of time for linear regime)
    t_start = t_final * 0.25
    t_end = t_final * 0.75
    
    growth = measure_growth_rate(hist['time'], hist['m1_amplitude'], t_start, t_end)
    
    if verbose:
        print(f"  → γ = {growth['gamma']:.4f} ± {growth['gamma_std']:.4f}, R²={growth['R_squared']:.3f}")
    
    return {
        'time': hist['time'],
        'amplitude': hist['m1_amplitude'],
        'gamma': growth['gamma'],
        'gamma_std': growth['gamma_std'],
        'R_squared': growth['R_squared'],
        'energy': hist['energy_total'],
        'div_B': hist['div_B_max']
    }


def eta_scaling_study(R: np.ndarray, Z: np.ndarray,
                     eta_values: np.ndarray,
                     nu: float = 0.001,
                     **kwargs) -> Dict:
    """
    Study growth rate scaling with η.
    
    Theory (FKR 1963): γ ~ η^(3/5) = η^0.6
    
    Parameters
    ----------
    R, Z : np.ndarray
        Grid
    eta_values : np.ndarray
        Resistivity values to test
    nu : float
        Viscosity (fixed)
    **kwargs : 
        Passed to run_linear_tearing_test
    
    Returns
    -------
    results : dict
        {
            'eta': np.ndarray,
            'gamma': np.ndarray,
            'gamma_std': np.ndarray,
            'scaling_exponent': float,
            'scaling_exponent_std': float
        }
    """
    print("=" * 70)
    print("η Scaling Study")
    print("=" * 70)
    print(f"Testing {len(eta_values)} values: {eta_values}")
    print(f"Fixed ν = {nu:.1e}\n")
    
    gamma_list = []
    gamma_std_list = []
    
    for i, eta in enumerate(eta_values):
        print(f"[{i+1}/{len(eta_values)}] η = {eta:.1e}")
        
        result = run_linear_tearing_test(R, Z, eta=eta, nu=nu, verbose=True, **kwargs)
        
        gamma_list.append(result['gamma'])
        gamma_std_list.append(result['gamma_std'])
        
        print()
    
    gamma_arr = np.array(gamma_list)
    gamma_std_arr = np.array(gamma_std_list)
    
    # Fit γ ~ η^p
    # log(γ) = p log(η) + const
    valid_mask = np.isfinite(gamma_arr) & (gamma_arr > 0)
    
    if np.sum(valid_mask) >= 2:
        log_eta = np.log(eta_values[valid_mask])
        log_gamma = np.log(gamma_arr[valid_mask])
        
        coeffs = np.polyfit(log_eta, log_gamma, deg=1)
        p = coeffs[0]  # Scaling exponent
        
        # Standard error
        log_gamma_fit = p * log_eta + coeffs[1]
        residuals = log_gamma - log_gamma_fit
        mse = np.sum(residuals**2) / (len(log_eta) - 2)
        var_log_eta = np.sum((log_eta - log_eta.mean())**2)
        p_std = np.sqrt(mse / var_log_eta) if var_log_eta > 0 else np.nan
    else:
        p = np.nan
        p_std = np.nan
    
    print("=" * 70)
    print("SCALING RESULT")
    print("=" * 70)
    print(f"γ ~ η^p")
    print(f"p = {p:.3f} ± {p_std:.3f}")
    print(f"Theory (FKR): p = 0.6")
    print(f"Agreement: {abs(p - 0.6) < 3*p_std if np.isfinite(p_std) else 'N/A'}")
    print("=" * 70)
    
    return {
        'eta': eta_values,
        'gamma': gamma_arr,
        'gamma_std': gamma_std_arr,
        'scaling_exponent': p,
        'scaling_exponent_std': p_std
    }
