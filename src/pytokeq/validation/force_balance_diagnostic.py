"""
Force Balance Diagnostic

Purpose:
- Verify equilibrium satisfies G-S equation: Δ*ψ + μ₀RJ = 0
- Independent check (recompute J from ψ)
- Quantify numerical accuracy

Implementation notes:
- Recompute J from converged ψ (independent verification)
- Use solver's Δ* operator (same discretization)
- Interior only (exclude boundary where residual = 0 by definition)
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from equilibrium.picard_gs_solver import Grid, compute_current_density, MU0


def delta_star_operator(psi, R, Z):
    """
    Compute Δ*ψ using SAME stencil as scipy sparse solver
    
    From linear_solver_sparse.py:
    Δ*ψ = c_im·ψ_{i-1,j} + c_ip·ψ_{i+1,j} + c_jm·ψ_{i,j-1} + c_jp·ψ_{i,j+1} + c_ij·ψ_{i,j}
    
    where:
    c_im = 1/dR² + 1/(2R·dR)
    c_ip = 1/dR² - 1/(2R·dR)
    c_jm = 1/dZ²
    c_jp = 1/dZ²
    c_ij = -(2/dR² + 2/dZ²)
    """
    nr, nz = psi.shape
    dR = R[1,0] - R[0,0]
    dZ = Z[0,1] - Z[0,0]
    
    delta_star = np.zeros_like(psi)
    
    for i in range(1, nr-1):
        for j in range(1, nz-1):
            Rc = R[i,j]
            
            # Stencil coefficients (SAME as solver!)
            c_im = 1/dR**2 + 1/(2*Rc*dR)  # ψ_{i-1,j}
            c_ip = 1/dR**2 - 1/(2*Rc*dR)  # ψ_{i+1,j}
            c_jm = 1/dZ**2                 # ψ_{i,j-1}
            c_jp = 1/dZ**2                 # ψ_{i,j+1}
            c_ij = -(2/dR**2 + 2/dZ**2)   # ψ_{i,j}
            
            # Δ*ψ = sum of stencil
            delta_star[i,j] = (c_im * psi[i-1,j] + 
                              c_ip * psi[i+1,j] +
                              c_jm * psi[i,j-1] +
                              c_jp * psi[i,j+1] +
                              c_ij * psi[i,j])
    
    return delta_star


def compute_residual(psi, grid, profile, psi_axis):
    """
    Compute force balance residual: R = |Δ*ψ + μ₀RJ|
    
    Args:
        psi: (nr, nz) Converged equilibrium flux [Wb]
        grid: Grid object
        profile: Profile model for J computation
        psi_axis: Magnetic axis value [Wb]
    
    Returns:
        residual: (nr, nz) Residual array [T]
        metrics: dict with L2, Linf, relative errors
    """
    nr, nz = psi.shape
    
    # Step 1: Recompute J from ψ (independent check)
    Jtor = compute_current_density(psi, grid, profile, psi_axis)
    
    # Step 2: Compute Δ*ψ
    delta_star_psi = delta_star_operator(psi, grid.R, grid.Z)
    
    # Step 3: Compute RHS = -μ₀RJ
    RHS = -MU0 * grid.R * Jtor
    
    # Step 4: Residual = |LHS - RHS| = |Δ*ψ + μ₀RJ|
    residual = np.abs(delta_star_psi - RHS)
    
    # Step 5: Metrics (interior only, exclude boundary)
    interior = residual[1:-1, 1:-1]
    RHS_interior = np.abs(RHS[1:-1, 1:-1])
    
    metrics = {
        'L2': np.sqrt(np.mean(interior**2)),
        'Linf': np.max(interior),
        'relative': np.sqrt(np.mean(interior**2)) / np.sqrt(np.mean(RHS_interior**2)) if np.any(RHS_interior > 0) else 0.0,
        'mean': np.mean(interior),
        'std': np.std(interior)
    }
    
    return residual, metrics


def check_force_balance(psi, grid, profile, psi_axis, verbose=True):
    """
    Check force balance and report results
    
    Args:
        psi: Converged solution
        grid: Grid object
        profile: Profile model
        psi_axis: Magnetic axis value
        verbose: Print detailed report
    
    Returns:
        metrics: dict with error metrics
        passed: bool (True if all criteria met)
    """
    residual, metrics = compute_residual(psi, grid, profile, psi_axis)
    
    # Success criteria (from benchmark plan)
    criteria = {
        'L2': 1e-6,
        'Linf': 1e-5,
        'relative': 1e-8
    }
    
    passed = (
        metrics['L2'] < criteria['L2'] and
        metrics['Linf'] < criteria['Linf'] and
        metrics['relative'] < criteria['relative']
    )
    
    if verbose:
        print("="*70)
        print("FORCE BALANCE DIAGNOSTIC")
        print("="*70)
        
        print(f"\nEquation: Δ*ψ + μ₀RJ = 0")
        print(f"Residual: R = |Δ*ψ + μ₀RJ|")
        
        print(f"\nMetrics (interior only):")
        print(f"  L2 norm:        {metrics['L2']:.3e} T")
        print(f"  L∞ norm:        {metrics['Linf']:.3e} T")
        print(f"  Relative error: {metrics['relative']:.3e}")
        print(f"  Mean:           {metrics['mean']:.3e} T")
        print(f"  Std dev:        {metrics['std']:.3e} T")
        
        print(f"\nSuccess criteria:")
        status_L2 = "✅" if metrics['L2'] < criteria['L2'] else "❌"
        status_Linf = "✅" if metrics['Linf'] < criteria['Linf'] else "❌"
        status_rel = "✅" if metrics['relative'] < criteria['relative'] else "❌"
        
        print(f"  {status_L2} L2 < {criteria['L2']:.0e}:  {metrics['L2']:.3e}")
        print(f"  {status_Linf} L∞ < {criteria['Linf']:.0e}: {metrics['Linf']:.3e}")
        print(f"  {status_rel} Relative < {criteria['relative']:.0e}: {metrics['relative']:.3e}")
        
        print(f"\n{'='*70}")
        if passed:
            print("✅ FORCE BALANCE CHECK PASSED")
        else:
            print("❌ FORCE BALANCE CHECK FAILED")
        print(f"{'='*70}")
    
    return metrics, passed


def visualize_residual(residual, grid, save_path=None):
    """
    Plot residual distribution (optional visualization)
    
    Args:
        residual: (nr, nz) Residual array
        grid: Grid object
        save_path: Path to save figure (optional)
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 2D contour
        levels = np.logspace(-10, -4, 20)
        c1 = ax1.contourf(grid.R, grid.Z, residual, levels=levels, cmap='viridis')
        ax1.set_xlabel('R [m]')
        ax1.set_ylabel('Z [m]')
        ax1.set_title('Force Balance Residual |Δ*ψ + μ₀RJ|')
        plt.colorbar(c1, ax=ax1, label='Residual [T]')
        ax1.set_aspect('equal')
        
        # Histogram
        interior = residual[1:-1, 1:-1].flatten()
        ax2.hist(np.log10(interior[interior > 0]), bins=50, edgecolor='black')
        ax2.set_xlabel('log₁₀(Residual) [T]')
        ax2.set_ylabel('Count')
        ax2.set_title('Residual Distribution (Interior)')
        ax2.axvline(np.log10(1e-6), color='r', linestyle='--', label='L2 criterion')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"\n📊 Residual plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except ImportError:
        print("\n⚠️  matplotlib not available, skipping visualization")


if __name__ == "__main__":
    # Test on Step 6 M3D-C1 result
    print("Testing force balance diagnostic on Step 6 M3D-C1 equilibrium...")
    
    from equilibrium.picard_gs_solver import Grid, solve_picard_free_boundary, CoilSet, Constraints
    from equilibrium.m3dc1_profile import M3DC1Profile
    
    # Setup (same as Step 6)
    R_1d = np.linspace(1.0, 2.0, 256)
    Z_1d = np.linspace(-0.5, 0.5, 256)
    grid = Grid.from_1d(R_1d, Z_1d)
    
    profile = M3DC1Profile()
    coils = CoilSet(R=np.array([]), Z=np.array([]), I=np.array([]))
    constraints = Constraints(xpoint=[], isoflux=[])
    
    print("\nRunning Picard solver...")
    result = solve_picard_free_boundary(
        profile=profile,
        grid=grid,
        coils=coils,
        constraints=constraints,
        max_outer=100,
        tol_psi=1e-6,
        damping=0.5
    )
    
    print(f"Converged in {result.niter} iterations\n")
    
    # Check force balance
    metrics, passed = check_force_balance(
        psi=result.psi,
        grid=grid,
        profile=profile,
        psi_axis=result.psi_axis,
        verbose=True
    )
    
    # Optional: visualize
    # visualize_residual(residual, grid, save_path='../reports/force_balance_residual.png')
    
    sys.exit(0 if passed else 1)
