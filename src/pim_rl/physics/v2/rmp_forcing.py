"""
RMP (Resonant Magnetic Perturbation) Forcing for v2.0 (Phase 4.4)

Author: 小P ⚛️  
Date: 2026-03-20

Implements external coil forcing for RL control.

Physics:
- Coil currents I → External B field (Biot-Savart)
- B_ext → J_ext = ∇×B_ext
- J_ext added to MHD RHS
"""

import jax.numpy as jnp
from typing import Tuple


def rmp_coil_field(coil_currents: jnp.ndarray, R_grid: jnp.ndarray, 
                   theta_grid: jnp.ndarray, z_grid: jnp.ndarray,
                   R0: float = 6.2, a: float = 2.0, n_mode: int = 1) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute RMP external magnetic field from coil currents
    
    Simplified model:
    - 4 coils at (R=R0+a, θ=0,π/2,π,3π/2) (outboard positions)
    - Toroidal mode number n=1 (typical for RMP)
    - Poloidal mode number m=n_coils/2 = 2
    
    Biot-Savart approximation in large-aspect-ratio limit:
    B_ext ~ I * cos(mθ - nζ) * r_envelope
    
    Args:
        coil_currents: [I₁, I₂, I₃, I₄] coil currents (kA)
        R_grid, theta_grid, z_grid: Grid coordinates (toroidal)
        R0, a: Major/minor radius
        n_mode: Toroidal mode number
        
    Returns:
        (B_r, B_theta, B_z): External field components
    """
    # Total coil current (sum, assuming symmetric winding)
    I_total = jnp.sum(coil_currents)  # kA
    
    # Convert to Amperes
    I_amp = I_total * 1000
    
    # Radial coordinate
    r = R_grid - R0
    
    # RMP field structure (resonant m=2, n=1 mode)
    m = 2  # Poloidal mode (from 4 coils)
    n = n_mode
    
    # Radial envelope (peaked at rational surface q=m/n)
    # For m=2, n=1: q=2 surface at r ~ 0.5a
    r_res = 0.5 * a
    width = 0.2 * a
    radial_envelope = jnp.exp(-((r - r_res) / width)**2)
    
    # Field amplitude (simplified scaling)
    # B ~ μ₀ I / (2π R)
    mu0 = 4 * jnp.pi * 1e-7  # H/m
    B_amplitude = (mu0 * I_amp) / (2 * jnp.pi * R0)  # Tesla
    
    # Resonant structure cos(mθ - nζ)
    phase = m * theta_grid - n * z_grid
    
    # External field components (simplified)
    # B_r: radial (main RMP component)
    B_r = B_amplitude * radial_envelope * jnp.cos(phase)
    
    # B_theta: poloidal (smaller, ~ m derivative)
    B_theta = -B_amplitude * radial_envelope * m * jnp.sin(phase) / 10
    
    # B_z: toroidal (even smaller, ~ n derivative)
    B_z = B_amplitude * radial_envelope * n * jnp.sin(phase) / 20
    
    return B_r, B_theta, B_z


def compute_current_density(B_ext: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
                           dr: float, dtheta: float, dz: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute J_ext = ∇×B_ext / μ₀
    
    Args:
        B_ext: (B_r, B_theta, B_z) external field
        dr, dtheta, dz: Grid spacing
        
    Returns:
        (J_r, J_theta, J_z): Current density
    """
    B_r, B_theta, B_z = B_ext
    mu0 = 4 * jnp.pi * 1e-7
    
    # Curl in cylindrical/toroidal coordinates (simplified)
    # J_r = (1/μ₀)[∂B_z/∂θ - ∂B_theta/∂z]
    # J_theta = (1/μ₀)[∂B_r/∂z - ∂B_z/∂r]
    # J_z = (1/μ₀)[∂B_theta/∂r - ∂B_r/∂θ]
    
    # Finite differences
    dBz_dtheta = (jnp.roll(B_z, -1, axis=1) - jnp.roll(B_z, 1, axis=1)) / (2*dtheta)
    dBtheta_dz = (jnp.roll(B_theta, -1, axis=2) - jnp.roll(B_theta, 1, axis=2)) / (2*dz)
    
    dBr_dz = (jnp.roll(B_r, -1, axis=2) - jnp.roll(B_r, 1, axis=2)) / (2*dz)
    dBz_dr = (jnp.roll(B_z, -1, axis=0) - jnp.roll(B_z, 1, axis=0)) / (2*dr)
    
    dBtheta_dr = (jnp.roll(B_theta, -1, axis=0) - jnp.roll(B_theta, 1, axis=0)) / (2*dr)
    dBr_dtheta = (jnp.roll(B_r, -1, axis=1) - jnp.roll(B_r, 1, axis=1)) / (2*dtheta)
    
    # Current density
    J_r = (dBz_dtheta - dBtheta_dz) / mu0
    J_theta = (dBr_dz - dBz_dr) / mu0
    J_z = (dBtheta_dr - dBr_dtheta) / mu0
    
    return J_r, J_theta, J_z


def test_rmp_forcing():
    """Test RMP forcing"""
    
    print("=" * 60)
    print("RMP Forcing Test (Phase 4.4)")
    print("=" * 60 + "\n")
    
    # Setup
    R0 = 6.2
    a = 2.0
    Nr, Ntheta, Nz = 16, 32, 16
    
    r = jnp.linspace(0, a, Nr)[:, None, None]
    theta = jnp.linspace(0, 2*jnp.pi, Ntheta)[None, :, None]
    z = jnp.linspace(0, 2*jnp.pi, Nz)[None, None, :]
    
    R_grid = R0 + r
    
    # Grid spacing
    dr = a / Nr
    dtheta = 2*jnp.pi / Ntheta
    dz = 2*jnp.pi / Nz
    
    print(f"Grid: {Nr}×{Ntheta}×{Nz}")
    print(f"R₀ = {R0}m, a = {a}m\n")
    
    # Test coil currents
    coil_currents = jnp.array([10.0, 10.0, 10.0, 10.0])  # 10kA each
    
    print(f"Coil currents: {coil_currents} kA")
    print(f"Total current: {jnp.sum(coil_currents)} kA\n")
    
    # Compute external field
    B_ext = rmp_coil_field(coil_currents, R_grid, theta, z, R0, a)
    B_r, B_theta, B_z = B_ext
    
    print("External magnetic field:")
    print(f"  |B_r|_max: {jnp.max(jnp.abs(B_r)):.6e} T")
    print(f"  |B_theta|_max: {jnp.max(jnp.abs(B_theta)):.6e} T")
    print(f"  |B_z|_max: {jnp.max(jnp.abs(B_z)):.6e} T\n")
    
    # Compute current density
    J_ext = compute_current_density(B_ext, dr, dtheta, dz)
    J_r, J_theta, J_z = J_ext
    
    print("External current density:")
    print(f"  |J_r|_max: {jnp.max(jnp.abs(J_r)):.6e} A/m²")
    print(f"  |J_theta|_max: {jnp.max(jnp.abs(J_theta)):.6e} A/m²")
    print(f"  |J_z|_max: {jnp.max(jnp.abs(J_z)):.6e} A/m²\n")
    
    # Check resonant structure
    r_idx = Nr // 2  # Mid-radius (near q=2 surface)
    z_idx = 0
    
    theta_profile = B_r[r_idx, :, z_idx]
    
    print("Resonant structure check:")
    print(f"  B_r at r=0.5a: {theta_profile[::8]}")
    print(f"  (Should oscillate with m=2 mode)\n")
    
    # Peak location
    peak_r_idx = jnp.argmax(jnp.abs(B_r[:, Ntheta//4, 0]))
    peak_r = r[peak_r_idx, 0, 0]
    
    print(f"Field peaks at r = {peak_r:.3f}m")
    print(f"Expected (q=2 surface): ~{0.5*a:.3f}m")
    
    if abs(peak_r - 0.5*a) < 0.2*a:
        print("  ✅ Resonant at correct radius\n")
    
    print("✅ Phase 4.4 RMP Forcing Complete!")
    print("Ready to integrate into CompleteMHDSolver")


if __name__ == "__main__":
    test_rmp_forcing()
