"""
Convert Solovev equilibrium to v2.0 toroidal format

Author: 小P ⚛️
Date: 2026-03-20

Purpose: 从equilibrium solver生成realistic B₀, P₀
"""

import jax.numpy as jnp
import numpy as np
from pytokeq.equilibrium.profiles.solovev import SolovevEquilibrium, make_standard_solovev


def solovev_to_toroidal(metric, grid_shape, scale_B=1.0, scale_P=1.0):
    """
    Convert Solovev equilibrium to v2.0 toroidal grid.
    
    Args:
        metric: BOUTMetric (R₀, a)
        grid_shape: (Nr, Ntheta, Nz)
        scale_B: Magnetic field scaling (normalize to ~1)
        scale_P: Pressure scaling
        
    Returns:
        B0_r, B0_theta, B0_z: Background magnetic field
        P0: Background pressure
        beta: Plasma beta profile
    """
    Nr, Ntheta, Nz = grid_shape
    R0 = metric.R0
    a = metric.a
    
    # Toroidal grid (in physical space)
    r = jnp.linspace(0.1 * a, 0.9 * a, Nr)  # Minor radius
    theta = jnp.linspace(0, 2*jnp.pi, Ntheta)  # Poloidal angle
    z_tor = jnp.linspace(0, 2*jnp.pi, Nz)  # Toroidal angle (periodic)
    
    # Convert to (R,Z) cylindrical
    r_grid = r[:, None]  # (Nr, 1)
    theta_grid = theta[None, :]  # (1, Ntheta)
    
    R_cyl = R0 + r_grid * jnp.cos(theta_grid)  # (Nr, Ntheta)
    Z_cyl = r_grid * jnp.sin(theta_grid)
    
    # Solovev equilibrium (analytical)
    sol = make_standard_solovev()
    
    # Magnetic field in cylindrical (R,Z,φ)
    B_R_cyl, B_Z_cyl, B_phi = sol.magnetic_field(np.array(R_cyl), 
                                                   np.array(Z_cyl))
    
    # Pressure
    psi_vals = sol.psi(np.array(R_cyl), np.array(Z_cyl))
    P_cyl = sol.pressure(psi_vals)
    
    # Convert to JAX
    B_R_cyl = jnp.array(B_R_cyl)
    B_Z_cyl = jnp.array(B_Z_cyl)
    B_phi = jnp.array(B_phi)
    P_cyl = jnp.array(P_cyl)
    
    # Transform to (r,θ,ζ) components
    # B_r = B_R cos(θ) + B_Z sin(θ)
    # B_theta = -B_R sin(θ) + B_Z cos(θ)
    # B_z = B_φ (toroidal direction)
    
    cos_theta = jnp.cos(theta_grid)  # (1, Ntheta)
    sin_theta = jnp.sin(theta_grid)
    
    B0_r = B_R_cyl * cos_theta + B_Z_cyl * sin_theta
    B0_theta = -B_R_cyl * sin_theta + B_Z_cyl * cos_theta
    B0_z = B_phi
    
    # Expand to 3D (same in all toroidal planes)
    B0_r = jnp.repeat(B0_r[:, :, None], Nz, axis=2)
    B0_theta = jnp.repeat(B0_theta[:, :, None], Nz, axis=2)
    B0_z = jnp.repeat(B0_z[:, :, None], Nz, axis=2)
    P0 = jnp.repeat(P_cyl[:, :, None], Nz, axis=2)
    
    # Normalize
    B_scale = float(jnp.max(jnp.abs(B0_z)))  # Normalize to dominant B_z
    B0_r = B0_r / B_scale * scale_B
    B0_theta = B0_theta / B_scale * scale_B
    B0_z = B0_z / B_scale * scale_B
    
    P_scale = float(jnp.max(P_cyl))
    P0 = P0 / P_scale * scale_P
    
    # Compute β
    B_squared = B0_r**2 + B0_theta**2 + B0_z**2
    beta = P0 / (B_squared / 2 + 1e-10)
    
    print("Equilibrium Conversion Summary")
    print("=" * 60)
    print(f"Geometry:")
    print(f"  R₀ = {R0:.2f} m, a = {a:.2f} m")
    print(f"  Grid: {Nr} × {Ntheta} × {Nz}")
    print()
    print(f"Normalization:")
    print(f"  B_scale (physical): {B_scale:.4f} T")
    print(f"  P_scale (physical): {P_scale:.4e} Pa")
    print(f"  Normalized to: B~{scale_B:.2f}, P~{scale_P:.2f}")
    print()
    print(f"Field magnitudes (normalized):")
    print(f"  |B_r|_max: {float(jnp.max(jnp.abs(B0_r))):.6f}")
    print(f"  |B_θ|_max: {float(jnp.max(jnp.abs(B0_theta))):.6f}")
    print(f"  |B_z|_max: {float(jnp.max(jnp.abs(B0_z))):.6f}")
    print(f"  <|B|²>^(1/2): {float(jnp.sqrt(jnp.mean(B_squared))):.6f}")
    print()
    print(f"Pressure (normalized):")
    print(f"  P_max: {float(jnp.max(P0)):.6f}")
    print(f"  <P>: {float(jnp.mean(P0)):.6f}")
    print()
    print(f"Plasma β:")
    print(f"  β_max: {float(jnp.max(beta)):.6f}")
    print(f"  <β>: {float(jnp.mean(beta)):.6f}")
    print()
    
    # Check if β is reasonable
    beta_mean = float(jnp.mean(beta))
    if 0.01 < beta_mean < 0.2:
        print(f"✅ β~{beta_mean:.3f} is tokamak-like (core ~0.02-0.05)")
    elif beta_mean < 0.01:
        print(f"⚠️ β~{beta_mean:.3f} is LOW (consider increasing scale_P)")
    else:
        print(f"⚠️ β~{beta_mean:.3f} is HIGH (consider decreasing scale_P)")
    
    return B0_r, B0_theta, B0_z, P0, beta


if __name__ == "__main__":
    from bout_metric import BOUTMetric
    
    # v2.0 geometry
    R0 = 6.2
    a = 2.0
    metric = BOUTMetric(R0, a)
    
    grid_shape = (16, 32, 16)
    
    # Convert with different scalings
    print("\n" + "=" * 60)
    print("Test 1: scale_B=1.0, scale_P=0.1")
    print("=" * 60)
    B0_r, B0_theta, B0_z, P0, beta = solovev_to_toroidal(
        metric, grid_shape, scale_B=1.0, scale_P=0.1
    )
    
    print("\n" + "=" * 60)
    print("Test 2: scale_B=1.0, scale_P=0.05")
    print("=" * 60)
    B0_r2, B0_theta2, B0_z2, P02, beta2 = solovev_to_toroidal(
        metric, grid_shape, scale_B=1.0, scale_P=0.05
    )
    
    print("\n" + "=" * 60)
    print("Recommendation")
    print("=" * 60)
    
    beta_mean1 = float(jnp.mean(beta))
    beta_mean2 = float(jnp.mean(beta2))
    
    if abs(beta_mean1 - 0.05) < abs(beta_mean2 - 0.05):
        print(f"✅ Use scale_P=0.1 (β={beta_mean1:.3f}, closer to target~0.05)")
    else:
        print(f"✅ Use scale_P=0.05 (β={beta_mean2:.3f}, closer to target~0.05)")
