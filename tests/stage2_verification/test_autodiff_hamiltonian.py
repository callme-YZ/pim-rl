"""
Test JAX Autodiff for Hamiltonian Gradient

Issue #24 Task 1: Verify jax.grad(H) works end-to-end

Author: 小P ⚛️
Date: 2026-03-24
"""

import sys
sys.path.insert(0, '../../src')

import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit

from pytokmhd.geometry.toroidal import ToroidalGrid


# ============================================================
# Convert Hamiltonian to JAX
# ============================================================

def _compute_derivatives_jax(f, grid):
    """JAX version of derivative computation"""
    nr, ntheta = f.shape
    dr = grid.dr
    dtheta = grid.dtheta
    
    # Initialize
    df_dr = jnp.zeros_like(f)
    df_dtheta = jnp.zeros_like(f)
    
    # Radial derivatives (2nd order centered)
    df_dr = df_dr.at[1:-1, :].set((f[2:, :] - f[:-2, :]) / (2*dr))
    # Boundary: 2nd order one-sided
    df_dr = df_dr.at[0, :].set((-3*f[0, :] + 4*f[1, :] - f[2, :]) / (2*dr))
    df_dr = df_dr.at[-1, :].set((3*f[-1, :] - 4*f[-2, :] + f[-3, :]) / (2*dr))
    
    # Theta derivatives (periodic, 2nd order centered)
    df_dtheta = df_dtheta.at[:, 1:-1].set((f[:, 2:] - f[:, :-2]) / (2*dtheta))
    # Periodic BC
    df_dtheta = df_dtheta.at[:, 0].set((f[:, 1] - f[:, -1]) / (2*dtheta))
    df_dtheta = df_dtheta.at[:, -1].set((f[:, 0] - f[:, -2]) / (2*dtheta))
    
    return df_dr, df_dtheta


def hamiltonian_jax(psi, phi, r_grid, dr, dtheta, R_grid):
    """
    JAX-compatible Hamiltonian energy computation.
    
    H = ∫ [(1/2)|∇φ|² + (1/2)|∇ψ|²] dV
    
    where dV = r*R dr dθ * 2π
    """
    # Compute derivatives
    dpsi_dr, dpsi_dtheta = _compute_derivatives_jax(psi, 
                                                     type('Grid', (), {'dr': dr, 'dtheta': dtheta})())
    dphi_dr, dphi_dtheta = _compute_derivatives_jax(phi, 
                                                     type('Grid', (), {'dr': dr, 'dtheta': dtheta})())
    
    # |∇ψ|² = (∂ψ/∂r)² + (1/r²)(∂ψ/∂θ)²
    grad_psi_sq = dpsi_dr**2 + (dpsi_dtheta / r_grid)**2
    
    # |∇φ|² = (∂φ/∂r)² + (1/r²)(∂φ/∂θ)²
    grad_phi_sq = dphi_dr**2 + (dphi_dtheta / r_grid)**2
    
    # Energy density
    h = 0.5 * (grad_psi_sq + grad_phi_sq)
    
    # Volume element: r*R dr dθ
    jacobian = r_grid * R_grid
    
    # Integrate
    energy_2d = jnp.sum(h * jacobian) * dr * dtheta
    
    # Multiply by 2π (toroidal direction)
    H = 2 * jnp.pi * energy_2d
    
    return H


# ============================================================
# Test 1: Basic Autodiff
# ============================================================

def test_basic_autodiff():
    """Test that jax.grad(H) works without errors"""
    print("=" * 60)
    print("TEST 1: Basic Autodiff - Does jax.grad work?")
    print("=" * 60)
    
    # Grid
    grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
    
    # Grid arrays (convert to JAX)
    r_grid = jnp.array(grid.r_grid)
    R_grid = jnp.array(grid.R_grid)
    dr = grid.dr
    dtheta = grid.dtheta
    
    # Test fields
    psi = jnp.array(grid.r_grid**2 * jnp.sin(grid.theta_grid))
    phi = jnp.array(grid.r_grid * jnp.cos(grid.theta_grid))
    
    print(f"Grid: {grid.nr}×{grid.ntheta}")
    print(f"psi shape: {psi.shape}")
    print(f"phi shape: {phi.shape}\n")
    
    # Define H as function of state
    def H_func(psi, phi):
        return hamiltonian_jax(psi, phi, r_grid, dr, dtheta, R_grid)
    
    # Compute H
    H = H_func(psi, phi)
    print(f"Hamiltonian H = {H:.6e}\n")
    
    # Test autodiff for psi
    print("Computing ∇_ψ H (gradient w.r.t. psi)...")
    try:
        grad_H_psi = grad(H_func, argnums=0)(psi, phi)
        print(f"✅ jax.grad(H, psi) SUCCESS")
        print(f"   Shape: {grad_H_psi.shape}")
        print(f"   Range: [{jnp.min(grad_H_psi):.6e}, {jnp.max(grad_H_psi):.6e}]")
    except Exception as e:
        print(f"❌ jax.grad(H, psi) FAILED: {e}")
        return False
    
    # Test autodiff for phi
    print("\nComputing ∇_φ H (gradient w.r.t. phi)...")
    try:
        grad_H_phi = grad(H_func, argnums=1)(psi, phi)
        print(f"✅ jax.grad(H, phi) SUCCESS")
        print(f"   Shape: {grad_H_phi.shape}")
        print(f"   Range: [{jnp.min(grad_H_phi):.6e}, {jnp.max(grad_H_phi):.6e}]")
    except Exception as e:
        print(f"❌ jax.grad(H, phi) FAILED: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ TEST 1 PASSED - JAX autodiff works!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_basic_autodiff()
    
    if success:
        print("\n✅ Task 1 Complete: JAX autodiff verified")
    else:
        print("\n❌ Task 1 Failed: Autodiff errors detected")
        sys.exit(1)
