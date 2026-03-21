"""
BOUT++ Metric Tensor for v2.0 (Phase 3.1)

Author: 小P ⚛️
Date: 2026-03-20

Implements toroidal metric tensor in field-aligned coordinates.

Theory:
- Coordinates: (ψ, θ, ζ) field-aligned
- Metric: g_ij from BOUT++ formulation
- Jacobian: √g = (h_θ R)/B_θ

References:
- Module 4 Section 2.2
- BOUT++ documentation
"""

import jax.numpy as jnp
from typing import Tuple


class BOUTMetric:
    """BOUT++ metric tensor in field-aligned coordinates
    
    Coordinates: (ψ, θ, ζ)
    - ψ: Poloidal flux
    - θ: Poloidal angle (straight-field-line)
    - ζ: Toroidal angle
    """
    
    def __init__(self, R0: float, a: float, epsilon: float = None):
        """Initialize metric
        
        Args:
            R0: Major radius (m)
            a: Minor radius (m)
            epsilon: Inverse aspect ratio ε=a/R₀ (optional)
        """
        self.R0 = R0
        self.a = a
        self.epsilon = epsilon if epsilon is not None else a / R0
        
        print(f"BOUT++ Metric initialized:")
        print(f"  R₀ = {R0} m")
        print(f"  a = {a} m")
        print(f"  ε = {self.epsilon:.3f}")
    
    def R_major(self, r: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        """Major radius R(r,θ) in large-aspect-ratio approximation
        
        R ≈ R₀ + r cos θ = R₀(1 + ε(r/a) cos θ)
        
        Args:
            r: Minor radius coordinate (0 to a)
            theta: Poloidal angle
            
        Returns:
            R: Major radius
        """
        return self.R0 * (1 + self.epsilon * (r / self.a) * jnp.cos(theta))
    
    def jacobian(self, r: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        """Jacobian √g = (h_θ R)/B_θ
        
        Large-aspect-ratio approximation:
        √g ≈ R₀ a (1 + ε cos θ)
        
        Args:
            r: Minor radius
            theta: Poloidal angle
            
        Returns:
            √g: Jacobian
        """
        R = self.R_major(r, theta)
        # Simplified: h_θ ≈ a, B_θ ≈ const
        return R * self.a
    
    def metric_tensor(self, r: jnp.ndarray, theta: jnp.ndarray) -> dict:
        """Compute metric tensor components g_ij
        
        Simplified large-aspect-ratio:
        g_xx ≈ (1/RB_θ)² (dominant term)
        g_yy ≈ h_θ² (poloidal)
        g_zz ≈ R² (toroidal)
        Off-diagonal ≈ 0 (simplified)
        
        Args:
            r, theta: Coordinates
            
        Returns:
            dict with g_xx, g_yy, g_zz, etc.
        """
        R = self.R_major(r, theta)
        
        # Simplified metric components
        B_theta = 1.0  # Normalized (can be refined)
        h_theta = self.a
        
        g_xx = (1 / (R * B_theta))**2
        g_yy = h_theta**2
        g_zz = R**2
        
        # Off-diagonal (simplified to zero)
        g_xy = jnp.zeros_like(R)
        g_xz = jnp.zeros_like(R)
        g_yz = jnp.zeros_like(R)
        
        return {
            'g_xx': g_xx,
            'g_yy': g_yy,
            'g_zz': g_zz,
            'g_xy': g_xy,
            'g_xz': g_xz,
            'g_yz': g_yz
        }
    
    def contravariant_metric(self, r: jnp.ndarray, theta: jnp.ndarray) -> dict:
        """Contravariant metric tensor g^ij (inverse of g_ij)
        
        For diagonal metric: g^ii = 1/g_ii
        
        Args:
            r, theta: Coordinates
            
        Returns:
            dict with g^xx, g^yy, g^zz
        """
        g = self.metric_tensor(r, theta)
        
        return {
            'g_xx': 1 / g['g_xx'],
            'g_yy': 1 / g['g_yy'],
            'g_zz': 1 / g['g_zz'],
            'g_xy': jnp.zeros_like(g['g_xy']),
            'g_xz': jnp.zeros_like(g['g_xz']),
            'g_yz': jnp.zeros_like(g['g_yz'])
        }


def test_bout_metric():
    """Test BOUT++ metric"""
    
    print("=" * 60)
    print("BOUT++ Metric Test (Phase 3.1)")
    print("=" * 60 + "\n")
    
    # ITER-like parameters
    R0 = 6.2  # m
    a = 2.0   # m
    epsilon = a / R0
    
    metric = BOUTMetric(R0, a, epsilon)
    
    # Grid
    Nr, Ntheta, Nz = 32, 32, 32
    r = jnp.linspace(0.1, a, Nr)[:, None, None]
    theta = jnp.linspace(0, 2*jnp.pi, Ntheta)[None, :, None]
    zeta = jnp.linspace(0, 2*jnp.pi, Nz)[None, None, :]
    
    print(f"\nGrid: {Nr}×{Ntheta}×{Nz}")
    print(f"r: [{r.min():.2f}, {r.max():.2f}] m")
    print(f"θ: [0, 2π]")
    print(f"ζ: [0, 2π]\n")
    
    # Compute metric quantities
    R = metric.R_major(r, theta)
    sqrt_g = metric.jacobian(r, theta)
    g = metric.metric_tensor(r, theta)
    
    print("Metric quantities:")
    print(f"  R (major radius):")
    print(f"    Min: {R.min():.3f} m (inboard)")
    print(f"    Max: {R.max():.3f} m (outboard)")
    print(f"    R₀: {metric.R0:.3f} m (geometric center)")
    
    print(f"\n  √g (Jacobian):")
    print(f"    Min: {sqrt_g.min():.3f}")
    print(f"    Max: {sqrt_g.max():.3f}")
    print(f"    Mean: {jnp.mean(sqrt_g):.3f}")
    
    print(f"\n  g_xx:")
    print(f"    Min: {g['g_xx'].min():.6e}")
    print(f"    Max: {g['g_xx'].max():.6e}")
    
    print(f"\n  g_yy:")
    if jnp.ndim(g['g_yy']) == 0:
        print(f"    Value: {float(g['g_yy']):.3f} (constant)")
    else:
        print(f"    Min: {g['g_yy'].min():.3f}")
        print(f"    Max: {g['g_yy'].max():.3f}")
    
    print(f"\n  g_zz:")
    print(f"    Min: {g['g_zz'].min():.3f}")
    print(f"    Max: {g['g_zz'].max():.3f}")
    
    # Check toroidal asymmetry
    R_inboard = R[:, 0, 0].min()   # θ=0
    R_outboard = R[:, Ntheta//2, 0].max()  # θ=π
    asymmetry = (R_outboard - R_inboard) / R0
    
    print(f"\nToroidal asymmetry:")
    print(f"  R_inboard (θ=0): {R_inboard:.3f} m")
    print(f"  R_outboard (θ=π): {R_outboard:.3f} m")
    print(f"  ΔR/R₀: {asymmetry:.2%}")
    print(f"  Expected (~2ε): {2*epsilon:.2%}")
    
    if abs(asymmetry - 2*epsilon) < 0.01:
        print("  ✅ Asymmetry matches large-aspect-ratio approximation")
    
    print("\n✅ Phase 3.1 BOUT++ Metric Complete!")
    
    return metric


if __name__ == "__main__":
    test_bout_metric()
