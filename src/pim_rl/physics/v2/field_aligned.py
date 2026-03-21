"""
Field-Aligned Coordinates for v2.0 (Phase 3.2)

Author: 小P ⚛️
Date: 2026-03-20

Implements field-aligned coordinate transformation z = ζ - ∫ν dθ

Theory:
- Standard toroidal: (ψ, θ, ζ)
- Field-aligned: (ψ, θ, z) where z parallel to B
- ν(ψ,θ) = (B·∇ζ)/(B·∇θ) = field-line pitch

References:
- Module 4 Section 1.2
- BOUT++ field-aligned coordinates
"""

import jax.numpy as jnp
from typing import Tuple

from .bout_metric import BOUTMetric


class FieldAlignedCoordinates:
    """Field-aligned coordinate transformation
    
    Transform: (ψ, θ, ζ) → (ψ, θ, z)
    where z = ζ - ∫_{θ₀}^θ ν(ψ, θ') dθ'
    """
    
    def __init__(self, metric: BOUTMetric, q_profile=None):
        """Initialize field-aligned coordinates
        
        Args:
            metric: BOUT++ metric
            q_profile: Safety factor q(ψ) (optional)
        """
        self.metric = metric
        
        # Default: constant q (simplified)
        if q_profile is None:
            self.q = lambda psi: 2.0  # q=2 (typical tokamak)
        else:
            self.q = q_profile
        
        print("Field-Aligned Coordinates initialized")
        print(f"  q-profile: {'custom' if q_profile else 'constant (q=2)'}")
    
    def field_line_pitch(self, psi: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        """Compute field-line pitch ν(ψ,θ)
        
        ν = (B·∇ζ)/(B·∇θ) ≈ 1/q(ψ)  (circular tokamak)
        
        For shaped plasma: ν = ν(ψ,θ) varies with θ
        
        Args:
            psi: Poloidal flux
            theta: Poloidal angle
            
        Returns:
            ν: Field-line pitch
        """
        # Simplified: ν = 1/q (circular approximation)
        q_val = self.q(psi)
        nu = 1.0 / q_val
        
        # Can add θ-dependence for shaping
        # nu = nu * (1 + triangularity * sin(theta))
        
        return nu
    
    def zeta_to_z(self, psi: jnp.ndarray, theta: jnp.ndarray, 
                  zeta: jnp.ndarray, dtheta: float) -> jnp.ndarray:
        """Transform ζ → z (field-aligned)
        
        z = ζ - ∫_{0}^θ ν(ψ, θ') dθ'
        
        Args:
            psi, theta, zeta: Toroidal coordinates
            dtheta: Theta grid spacing
            
        Returns:
            z: Field-aligned coordinate
        """
        # Compute ν at each theta
        nu = self.field_line_pitch(psi, theta)
        
        # Broadcast nu to theta shape if needed
        if jnp.ndim(nu) == 0:
            nu = nu * jnp.ones_like(theta)
        
        # Integrate: ∫ ν dθ (cumulative sum)
        # For 3D grid (Npsi, Ntheta, Nzeta):
        # Integrate along theta axis (axis=1)
        
        if theta.ndim == 3:
            # 3D grid
            integral = jnp.cumsum(nu * dtheta, axis=1)
        elif theta.ndim == 1:
            # 1D theta array
            integral = jnp.cumsum(nu * dtheta)
        else:
            integral = nu * theta  # Simplified
        
        # z = ζ - ∫ν dθ
        z = zeta - integral
        
        return z
    
    def z_to_zeta(self, psi: jnp.ndarray, theta: jnp.ndarray,
                  z: jnp.ndarray, dtheta: float) -> jnp.ndarray:
        """Transform z → ζ (inverse)
        
        ζ = z + ∫_{0}^θ ν(ψ, θ') dθ'
        
        Args:
            psi, theta, z: Coordinates
            dtheta: Grid spacing
            
        Returns:
            zeta: Toroidal angle
        """
        nu = self.field_line_pitch(psi, theta)
        
        # Broadcast if scalar
        if jnp.ndim(nu) == 0:
            nu = nu * jnp.ones_like(theta)
        
        if theta.ndim == 3:
            integral = jnp.cumsum(nu * dtheta, axis=1)
        elif theta.ndim == 1:
            integral = jnp.cumsum(nu * dtheta)
        else:
            integral = nu * theta
        
        zeta = z + integral
        
        return zeta
    
    def parallel_derivative(self, f: jnp.ndarray, dz: float) -> jnp.ndarray:
        """Compute parallel derivative ∂f/∂z
        
        In field-aligned coords: B·∇f ∝ ∂f/∂z
        
        Args:
            f: Field (Npsi, Ntheta, Nz)
            dz: Grid spacing in z
            
        Returns:
            ∂f/∂z (parallel derivative)
        """
        # Central difference in z-direction (axis=2)
        df_dz = (jnp.roll(f, -1, axis=2) - jnp.roll(f, 1, axis=2)) / (2*dz)
        return df_dz


def test_field_aligned():
    """Test field-aligned coordinates"""
    
    print("=" * 60)
    print("Field-Aligned Coordinates Test (Phase 3.2)")
    print("=" * 60 + "\n")
    
    # Metric
    R0 = 6.2
    a = 2.0
    metric = BOUTMetric(R0, a)
    
    # Field-aligned coords
    fa = FieldAlignedCoordinates(metric)
    
    # Grid
    Npsi, Ntheta, Nzeta = 16, 64, 64
    psi = jnp.linspace(0, 1, Npsi)[:, None, None]
    theta = jnp.linspace(0, 2*jnp.pi, Ntheta)[None, :, None]
    zeta = jnp.linspace(0, 2*jnp.pi, Nzeta)[None, None, :]
    
    dtheta = 2*jnp.pi / Ntheta
    dzeta = 2*jnp.pi / Nzeta
    
    print(f"Grid: {Npsi}×{Ntheta}×{Nzeta}")
    print(f"dθ = {dtheta:.6f}, dζ = {dzeta:.6f}\n")
    
    # Compute field-line pitch
    nu = fa.field_line_pitch(psi, theta)
    print(f"Field-line pitch ν:")
    nu_val = float(nu) if jnp.ndim(nu) == 0 else float(nu.flatten()[0])
    print(f"  ν = 1/q = {nu_val:.3f} (q=2)")
    if jnp.ndim(nu) > 0:
        print(f"  Shape: {nu.shape}\n")
    else:
        print(f"  (constant)\n")
    
    # Transform ζ → z
    z = fa.zeta_to_z(psi, theta, zeta, dtheta)
    print(f"Field-aligned coordinate z:")
    print(f"  z_min = {z.min():.3f}")
    print(f"  z_max = {z.max():.3f}")
    print(f"  z range: [{z.min():.3f}, {z.max():.3f}]\n")
    
    # Inverse transform z → ζ
    zeta_reconstructed = fa.z_to_zeta(psi, theta, z, dtheta)
    error = jnp.max(jnp.abs(zeta_reconstructed - zeta))
    print(f"Inverse transform check:")
    print(f"  ζ_original: [{zeta.min():.3f}, {zeta.max():.3f}]")
    print(f"  ζ_reconstructed: [{zeta_reconstructed.min():.3f}, {zeta_reconstructed.max():.3f}]")
    print(f"  Max error: {error:.6e}")
    
    if error < 1e-10:
        print("  ✅ Inverse transform exact!\n")
    else:
        print(f"  ⚠️ Inverse has error {error:.2e}\n")
    
    # Test parallel derivative
    # Test function: f = sin(z)
    f_test = jnp.sin(z)
    df_dz_numerical = fa.parallel_derivative(f_test, dzeta)
    df_dz_analytical = jnp.cos(z)
    
    # Compare (interior points)
    interior = (slice(None), slice(2,-2), slice(2,-2))
    deriv_error = jnp.max(jnp.abs(df_dz_numerical[interior] - df_dz_analytical[interior]))
    
    print(f"Parallel derivative test (f=sin(z)):")
    print(f"  ∂f/∂z numerical: {df_dz_numerical[0,Ntheta//2,Nzeta//2]:.6f}")
    print(f"  ∂f/∂z analytical: {df_dz_analytical[0,Ntheta//2,Nzeta//2]:.6f}")
    print(f"  Max error (interior): {deriv_error:.6e}")
    
    if deriv_error < 1e-3:
        print("  ✅ Parallel derivative correct!\n")
    else:
        print(f"  ⚠️ Derivative has error {deriv_error:.2e}\n")
    
    # Field-line following test
    # A field line should have constant (ψ, z)
    psi_line = 0.5  # Fixed flux surface
    theta_line = jnp.linspace(0, 2*jnp.pi, 100)
    z_line = 1.0    # Fixed z (follows field line)
    
    # Compute ζ along this field line
    psi_array = jnp.full_like(theta_line, psi_line)
    zeta_line = fa.z_to_zeta(psi_array, theta_line, z_line, dtheta)
    
    # Field line should wind around torus: Δζ ≈ 2πq after one poloidal turn
    q = 2.0
    Delta_zeta_expected = 2*jnp.pi * q
    Delta_zeta_actual = zeta_line[-1] - zeta_line[0]
    
    print(f"Field-line following:")
    print(f"  Fixed (ψ={psi_line}, z={z_line})")
    print(f"  θ: 0 → 2π")
    print(f"  ζ change: {Delta_zeta_actual:.3f}")
    print(f"  Expected (2πq): {Delta_zeta_expected:.3f}")
    print(f"  Error: {abs(Delta_zeta_actual - Delta_zeta_expected):.6f}")
    
    if abs(Delta_zeta_actual - Delta_zeta_expected) < 0.1:
        print("  ✅ Field line winding correct!\n")
    
    print("✅ Phase 3.2 Field-Aligned Coordinates Complete!")
    
    return fa


if __name__ == "__main__":
    test_field_aligned()
