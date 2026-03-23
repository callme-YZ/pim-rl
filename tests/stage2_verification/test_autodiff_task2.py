"""
Test JAX Autodiff Correctness vs Finite Difference

Issue #24 Task 2: Validate gradients

Key insight: Need ε ≈ sqrt(machine_epsilon * H) ≈ 1e-3 for best FD accuracy

Author: 小P ⚛️
Date: 2026-03-24
"""

import sys
sys.path.insert(0, '../../src')

import jax.numpy as jnp
from jax import grad

from pytokmhd.geometry.toroidal import ToroidalGrid
from test_autodiff_hamiltonian import hamiltonian_jax


def test_autodiff_correctness():
    """Validate autodiff gradients vs finite difference"""
    print("=" * 60)
    print("TEST: Autodiff Correctness vs Finite Difference")
    print("=" * 60)
    
    # Grid
    grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
    
    r_grid = jnp.array(grid.r_grid)
    R_grid = jnp.array(grid.R_grid)
    dr = grid.dr
    dtheta = grid.dtheta
    
    # Test fields
    r = r_grid[:, 0:1]
    theta = grid.theta_grid[0:1, :]
    
    psi = jnp.array(0.1 * (r**2 + 0.1) * (jnp.sin(2*theta) + 1.5))
    phi = jnp.array(0.05 * (r + 0.05) * (jnp.cos(theta) + 1.2))
    
    def H_func(psi, phi):
        return hamiltonian_jax(psi, phi, r_grid, dr, dtheta, R_grid)
    
    H0 = H_func(psi, phi)
    print(f"\nHamiltonian H = {H0:.6e}")
    
    # Autodiff
    print("\nComputing autodiff gradients...")
    grad_H_psi = grad(H_func, argnums=0)(psi, phi)
    grad_H_phi = grad(H_func, argnums=1)(psi, phi)
    print("✅ Autodiff complete")
    
    # Finite difference with optimal epsilon
    # ε ≈ sqrt(machine_eps * H) ≈ sqrt(1e-16 * 1e-2) ≈ 1e-9
    # But H changes ~1e-7 per field change, so need ε ≈ 1e-3
    epsilon = 1e-3
    
    sample_points = [
        (8, 16),
        (16, 32),
        (24, 48),
    ]
    
    print(f"\nComputing FD at {len(sample_points)} points (ε={epsilon})...")
    
    results = []
    
    for i, j in sample_points:
        # ∇_ψ H
        psi_plus = psi.at[i, j].add(epsilon)
        psi_minus = psi.at[i, j].add(-epsilon)
        H_plus = H_func(psi_plus, phi)
        H_minus = H_func(psi_minus, phi)
        grad_psi_fd = (H_plus - H_minus) / (2 * epsilon)
        grad_psi_auto = grad_H_psi[i, j]
        
        # ∇_φ H
        phi_plus = phi.at[i, j].add(epsilon)
        phi_minus = phi.at[i, j].add(-epsilon)
        H_plus = H_func(psi, phi_plus)
        H_minus = H_func(psi, phi_minus)
        grad_phi_fd = (H_plus - H_minus) / (2 * epsilon)
        grad_phi_auto = grad_H_phi[i, j]
        
        results.append({
            'pos': (i, j),
            'psi_auto': grad_psi_auto,
            'psi_fd': grad_psi_fd,
            'phi_auto': grad_phi_auto,
            'phi_fd': grad_phi_fd,
        })
    
    print("✅ FD complete\n")
    
    # Display
    print("Comparison:")
    print("-" * 75)
    print("Position   Field     Autodiff        FD            Abs Diff    Rel Error")
    print("-" * 75)
    
    max_error = 0
    
    for r in results:
        i, j = r['pos']
        
        # ∇_ψ H
        abs_diff = abs(r['psi_auto'] - r['psi_fd'])
        rel_err = abs_diff / (abs(r['psi_fd']) + 1e-10)
        max_error = max(max_error, rel_err)
        print(f"({i:2d},{j:2d})   ∇_ψ H   {r['psi_auto']:+.6e}  {r['psi_fd']:+.6e}  {abs_diff:.3e}   {rel_err:.2%}")
        
        # ∇_φ H
        abs_diff = abs(r['phi_auto'] - r['phi_fd'])
        rel_err = abs_diff / (abs(r['phi_fd']) + 1e-10)
        max_error = max(max_error, rel_err)
        print(f"         ∇_φ H   {r['phi_auto']:+.6e}  {r['phi_fd']:+.6e}  {abs_diff:.3e}   {rel_err:.2%}")
    
    print("-" * 75)
    print(f"\nMax relative error: {max_error:.2%}")
    
    # Success criterion
    threshold = 0.02  # 2% (FD truncation error expected)
    
    if max_error < threshold:
        print(f"\n✅ TEST PASSED - All errors < {threshold:.1%}")
        print("\nConclusion:")
        print("  ✅ JAX autodiff produces correct gradients")
        print("  ✅ Agreement with finite difference within numerical error")
        return True
    else:
        print(f"\n❌ TEST FAILED - Max error {max_error:.2%} > {threshold:.1%}")
        return False


if __name__ == "__main__":
    success = test_autodiff_correctness()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ Task 2 Complete: Gradients validated vs FD")
        print("=" * 60)
    else:
        print("\n❌ Task 2 Failed")
        sys.exit(1)
