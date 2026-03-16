"""
Comprehensive diagnosis of constraint optimization instability

Following 小A's systematic approach
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from equilibrium.picard_gs_solver import Grid, ProfileModel, CoilSet, Constraints, MU0
from equilibrium.constraint_optimizer import (
    evaluate_constraints_impl,
    compute_sensitivity_matrix_impl,
    optimize_coils_impl
)


class SimpleProfile(ProfileModel):
    """Simple profile for testing"""
    def pprime(self, psi_norm):
        return -2 * MU0 * np.ones_like(psi_norm)
    def ffprime(self, psi_norm):
        return np.zeros_like(psi_norm)


def setup_test_case():
    """Setup same test case as free-boundary test"""
    R_1d = np.linspace(0.5, 2.5, 41)
    Z_1d = np.linspace(-1.0, 1.0, 41)
    grid = Grid.from_1d(R_1d, Z_1d)
    
    # 4 coils
    coils = CoilSet(
        R=np.array([0.8, 2.2, 0.8, 2.2]),
        Z=np.array([-0.6, -0.6, 0.6, 0.6]),
        I=np.array([1e4, 1e4, 1e4, 1e4])
    )
    
    # Constraints
    xpoint = [(1.5, -0.4)]
    isoflux = [(1.0, 0.0), (2.0, 0.0), (1.5, 0.5), (1.5, -0.5)]
    constraints = Constraints(xpoint=xpoint, isoflux=isoflux)
    
    return grid, coils, constraints


def diagnose_sensitivity_matrix(grid, coils, constraints):
    """Complete diagnosis of sensitivity matrix"""
    print("\n" + "="*70)
    print("DIAGNOSIS: Sensitivity Matrix")
    print("="*70)
    
    # Create initial psi (simple parabolic)
    R0 = 1.5
    psi = -((grid.R - R0)**2 + grid.Z**2)
    
    # Simple Jtor
    Jtor = -2 * np.ones_like(grid.R)
    
    # Compute sensitivity matrix
    print("\nComputing sensitivity matrix...")
    A = compute_sensitivity_matrix_impl(
        psi=psi,
        grid=grid,
        Jtor=Jtor,
        coil_R=coils.R,
        coil_Z=coils.Z,
        I_coil=coils.I,
        xpoint=constraints.xpoint,
        isoflux=constraints.isoflux,
        Ip_target=None,
        dI=1.0
    )
    
    print(f"\n1. MATRIX SHAPE & SIZE:")
    print(f"   Shape: {A.shape}")
    print(f"   (n_constraints={A.shape[0]}, n_coils={A.shape[1]})")
    
    print(f"\n2. MATRIX MAGNITUDE:")
    print(f"   Frobenius norm: {np.linalg.norm(A):.3e}")
    print(f"   Max element: {np.abs(A).max():.3e}")
    print(f"   Min element: {np.abs(A).min():.3e}")
    print(f"   Mean |element|: {np.abs(A).mean():.3e}")
    
    print(f"\n3. MATRIX CONDITION:")
    cond = np.linalg.cond(A)
    print(f"   Condition number: {cond:.3e}")
    if cond > 1e10:
        print(f"   ⚠️  ILL-CONDITIONED (cond > 1e10)")
    elif cond > 1e6:
        print(f"   ⚠️  MODERATELY ILL-CONDITIONED (cond > 1e6)")
    else:
        print(f"   ✓  WELL-CONDITIONED (cond < 1e6)")
    
    print(f"\n4. MATRIX RANK:")
    rank = np.linalg.matrix_rank(A)
    full_rank = min(A.shape)
    print(f"   Rank: {rank}/{full_rank}")
    if rank < full_rank:
        print(f"   ⚠️  RANK DEFICIENT! ({full_rank - rank} missing)")
        print(f"   → Constraints are linearly dependent!")
    else:
        print(f"   ✓  FULL RANK")
    
    print(f"\n5. SINGULAR VALUES:")
    svd = np.linalg.svd(A, compute_uv=False)
    print(f"   Largest: {svd[0]:.3e}")
    print(f"   Smallest: {svd[-1]:.3e}")
    print(f"   Ratio: {svd[0]/svd[-1]:.3e}")
    print(f"   All values: {svd}")
    
    return A, cond


def diagnose_constraints(psi, grid, Jtor, constraints):
    """Diagnose constraint evaluation"""
    print("\n" + "="*70)
    print("DIAGNOSIS: Constraint Evaluation")
    print("="*70)
    
    b = evaluate_constraints_impl(
        psi=psi,
        grid=grid,
        Jtor=Jtor,
        xpoint=constraints.xpoint,
        isoflux=constraints.isoflux,
        Ip_target=None
    )
    
    print(f"\n1. CONSTRAINT ERRORS:")
    print(f"   Number: {len(b)}")
    print(f"   Norm: {np.linalg.norm(b):.3e}")
    print(f"   Max: {np.abs(b).max():.3e}")
    print(f"   Mean: {np.abs(b).mean():.3e}")
    
    print(f"\n2. INDIVIDUAL ERRORS:")
    idx = 0
    for i, (R, Z) in enumerate(constraints.xpoint):
        print(f"   X-point {i} at ({R:.2f}, {Z:.2f}):")
        print(f"     Br error: {b[idx]:.3e}")
        print(f"     Bz error: {b[idx+1]:.3e}")
        idx += 2
    
    if len(constraints.isoflux) > 1:
        print(f"   Isoflux constraints:")
        for i in range(len(constraints.isoflux)-1):
            print(f"     Point {i+1}: {b[idx]:.3e}")
            idx += 1
    
    return b


def diagnose_perturbation(A, b, coils):
    """Diagnose computed perturbation"""
    print("\n" + "="*70)
    print("DIAGNOSIS: Computed Perturbation")
    print("="*70)
    
    # Solve without regularization
    try:
        delta_I_unreg = np.linalg.lstsq(A, b, rcond=None)[0]
        print(f"\n1. UNREGULARIZED SOLUTION:")
        print(f"   ΔI norm: {np.linalg.norm(delta_I_unreg):.3e}")
        print(f"   ΔI max: {np.abs(delta_I_unreg).max():.3e}")
        print(f"   ΔI values: {delta_I_unreg}")
        
        # Relative to current
        rel = np.abs(delta_I_unreg) / np.abs(coils.I)
        print(f"\n   Relative to I_coil:")
        print(f"     Max ratio: {rel.max():.3e}")
        print(f"     Mean ratio: {rel.mean():.3e}")
        
        if rel.max() > 100:
            print(f"   ⚠️  HUGE PERTURBATION (>100× current)")
        elif rel.max() > 10:
            print(f"   ⚠️  LARGE PERTURBATION (>10× current)")
        else:
            print(f"   ✓  REASONABLE PERTURBATION")
    
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        delta_I_unreg = None
    
    # Solve with regularization
    gamma_values = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4]
    print(f"\n2. REGULARIZATION EFFECTS:")
    
    for gamma in gamma_values:
        ATA = A.T @ A
        ATb = A.T @ b
        reg = gamma**2 * np.eye(len(coils.I))
        
        try:
            delta_I_reg = np.linalg.solve(ATA + reg, ATb)
            norm_delta = np.linalg.norm(delta_I_reg)
            max_delta = np.abs(delta_I_reg).max()
            
            print(f"   γ={gamma:.0e}: ||ΔI||={norm_delta:.2e}, max={max_delta:.2e}")
        except:
            print(f"   γ={gamma:.0e}: FAILED")
    
    return delta_I_unreg


def main():
    """Main diagnosis"""
    print("\n" + "="*70)
    print("CONSTRAINT OPTIMIZER COMPREHENSIVE DIAGNOSIS")
    print("Following 小A's systematic approach")
    print("="*70)
    
    # Setup
    grid, coils, constraints = setup_test_case()
    
    print(f"\nTest case:")
    print(f"  Grid: {grid.nr}×{grid.nz}")
    print(f"  Coils: {len(coils.R)}")
    print(f"  Constraints: {constraints.num_equations()} equations")
    
    # Initial state
    R0 = 1.5
    psi = -((grid.R - R0)**2 + grid.Z**2)
    Jtor = -2 * np.ones_like(grid.R)
    
    # Run diagnoses
    A, cond = diagnose_sensitivity_matrix(grid, coils, constraints)
    b = diagnose_constraints(psi, grid, Jtor, constraints)
    delta_I = diagnose_perturbation(A, b, coils)
    
    # Summary and recommendations
    print("\n" + "="*70)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*70)
    
    if cond > 1e10:
        print("\n⚠️  CRITICAL: Matrix is ILL-CONDITIONED")
        print("   Recommendation:")
        print("   1. Increase perturbation dI (1.0 → 10.0)")
        print("   2. Use stronger regularization (γ=1e-8)")
        print("   3. Check constraint placement (too close?)")
        print("   Time: 30-40 min to fix")
    
    elif cond > 1e6:
        print("\n⚠️  WARNING: Matrix is moderately ill-conditioned")
        print("   Recommendation:")
        print("   1. Use moderate regularization (γ=1e-10)")
        print("   2. Add damping to coil updates (α=0.1)")
        print("   Time: 15-20 min to fix")
    
    else:
        print("\n✓  Matrix is well-conditioned")
        print("   Recommendation:")
        print("   1. Problem is STEP SIZE, not matrix")
        print("   2. Add damping (α=0.1-0.3)")
        print("   3. Keep current regularization")
        print("   Time: 10-15 min to fix")
    
    # Theory-based alpha
    if A is not None and cond < 1e10:
        alpha_theory = 2.0 / (cond * np.linalg.norm(A))
        alpha_safe = 0.1 * alpha_theory
        print(f"\n   Theory-based damping:")
        print(f"     α_max (theory): {alpha_theory:.3e}")
        print(f"     α_safe (0.1×): {alpha_safe:.3e}")
        print(f"     Recommended: α={min(0.3, alpha_safe):.3e}")


if __name__ == "__main__":
    main()

