"""
Test with manufactured solution: verify matrix is computing correct Laplacian.
"""

import numpy as np
import sys
sys.path.insert(0, 'src')

from pytokmhd.solvers.poisson_3d_fixed import (
    solve_poisson_3d,
    build_2d_laplacian_matrix
)
from scipy.sparse.linalg import spsolve


class Grid3D:
    def __init__(self, nr=16, nθ=16, nζ=1, r_min=0.1, r_max=1.0):
        self.nr = nr
        self.nθ = nθ
        self.nζ = nζ
        self.r_min = r_min
        self.r_max = r_max
        
        self.dr = (r_max - r_min) / (nr - 1) if nr > 1 else 0.0
        self.dθ = 2 * np.pi / nθ
        self.dζ = 2 * np.pi / nζ
        
        self.r = np.linspace(r_min, r_max, nr)
        self.θ = np.linspace(0, 2*np.pi, nθ, endpoint=False)
        self.ζ = np.linspace(0, 2*np.pi, nζ, endpoint=False)
        
        self.Lr = r_max - r_min
        self.Lθ = 2 * np.pi
        self.Lζ = 2 * np.pi
    
    def meshgrid(self):
        r_3d, θ_3d, ζ_3d = np.meshgrid(self.r, self.θ, self.ζ, indexing='ij')
        return r_3d, θ_3d, ζ_3d


def test_constant_solution():
    """Test φ = const → ∇²φ = 0 (except at BC)."""
    print("=" * 60)
    print("Test: Constant Solution")
    print("=" * 60)
    
    grid = Grid3D(nr=8, nθ=8, nζ=1, r_min=0.1, r_max=1.0)
    nr, nθ = grid.nr, grid.nθ
    
    # Build matrix for k=0
    A = build_2d_laplacian_matrix(grid.r, grid.dr, grid.dθ, kz=0.0, bc='none', nθ=nθ)
    
    # Constant solution
    phi_const = 5.0 * np.ones((nr, nθ))
    phi_flat = phi_const.flatten(order='F')
    
    # Compute A*φ (should be ~0 for interior, but BC may affect)
    laplacian_flat = A.dot(phi_flat)
    laplacian = laplacian_flat.reshape((nr, nθ), order='F')
    
    # Interior points only (exclude r=0, r=a)
    interior_error = np.max(np.abs(laplacian[1:-1, :]))
    
    print(f"Max |∇²φ| in interior: {interior_error:.2e}")
    print(f"  (Should be ~0 for constant φ)")
    
    if interior_error < 1e-10:
        print("✅ PASSED")
    else:
        print(f"❌ FAILED: Interior error {interior_error:.2e} > 1e-10")
    
    print()


def test_pure_radial_solution():
    """Test φ = r² → ∇²φ = ∂²/∂r²(r²) + (1/r)∂/∂r(r²) = 2 + 2 = 4."""
    print("=" * 60)
    print("Test: Pure Radial Solution φ = r²")
    print("=" * 60)
    
    grid = Grid3D(nr=16, nθ=8, nζ=1, r_min=0.5, r_max=1.0)
    nr, nθ = grid.nr, grid.nθ
    r_2d, _, _ = grid.meshgrid()
    
    # Solution: φ = r²
    phi = r_2d[:, :, 0]**2
    
    # Expected Laplacian: ∇²(r²) = 4
    omega_expected = 4.0 * np.ones_like(phi)
    
    # Build matrix and compute numerical Laplacian
    A = build_2d_laplacian_matrix(grid.r, grid.dr, grid.dθ, kz=0.0, bc='none', nθ=nθ)
    
    phi_flat = phi.flatten(order='F')
    omega_flat = A.dot(phi_flat)
    omega_num = omega_flat.reshape((nr, nθ), order='F')
    
    # Compare (interior only)
    error = np.max(np.abs(omega_num[2:-2, :] - omega_expected[2:-2, :]))
    
    print(f"Expected ∇²φ = 4.0")
    print(f"Numerical ∇²φ range: [{omega_num[2:-2,:].min():.4f}, {omega_num[2:-2,:].max():.4f}]")
    print(f"Max error: {error:.2e}")
    
    if error < 0.1:  # Relaxed for FD error
        print("✅ PASSED")
    else:
        print(f"❌ FAILED: Error {error:.2e} > 0.1")
    
    print()


def test_poloidal_mode():
    """Test φ = cos(2θ) → ∇²φ = -(1/r²) * 4 * cos(2θ)."""
    print("=" * 60)
    print("Test: Poloidal Mode φ = cos(2θ)")
    print("=" * 60)
    
    grid = Grid3D(nr=8, nθ=32, nζ=1, r_min=0.5, r_max=1.0)
    nr, nθ = grid.nr, grid.nθ
    r_2d, θ_2d, _ = grid.meshgrid()
    
    # Solution: φ = cos(2θ)
    phi = np.cos(2 * θ_2d[:, :, 0])
    
    # Expected Laplacian: ∇²(cos(mθ)) = -(m²/r²) cos(mθ)
    # For m=2: ∇²φ = -(4/r²) cos(2θ)
    omega_expected = -(4.0 / r_2d[:, :, 0]**2) * np.cos(2 * θ_2d[:, :, 0])
    
    # Numerical Laplacian
    A = build_2d_laplacian_matrix(grid.r, grid.dr, grid.dθ, kz=0.0, bc='none', nθ=nθ)
    
    phi_flat = phi.flatten(order='F')
    omega_flat = A.dot(phi_flat)
    omega_num = omega_flat.reshape((nr, nθ), order='F')
    
    # Compare (interior only)
    error = np.max(np.abs(omega_num[2:-2, :] - omega_expected[2:-2, :]))
    
    print(f"Expected ∇²φ range: [{omega_expected[2:-2,:].min():.4f}, {omega_expected[2:-2,:].max():.4f}]")
    print(f"Numerical ∇²φ range: [{omega_num[2:-2,:].min():.4f}, {omega_num[2:-2,:].max():.4f}]")
    print(f"Max error: {error:.2e}")
    
    if error < 0.5:  # Relaxed
        print("✅ PASSED")
    else:
        print(f"❌ FAILED: Error {error:.2e} > 0.5")
    
    print()


if __name__ == "__main__":
    test_constant_solution()
    test_pure_radial_solution()
    test_poloidal_mode()
    
    print("=" * 60)
    print("Manufactured Solution Tests Complete")
    print("=" * 60)
