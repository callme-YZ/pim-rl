"""
Test the fixed 2D Poisson solver on a small grid.
"""

import numpy as np
import sys
sys.path.insert(0, 'src')

from pytokmhd.solvers.poisson_3d_fixed import (
    solve_poisson_3d,
    compute_laplacian_3d,
    build_radial_laplacian,
    build_poloidal_laplacian,
    build_2d_laplacian_matrix
)


class Grid3D:
    """Simple 3D cylindrical grid for testing."""
    def __init__(self, nr=8, nθ=8, nζ=8, r_min=0.0, r_max=1.0):
        self.nr = nr
        self.nθ = nθ
        self.nζ = nζ
        self.r_min = r_min
        self.r_max = r_max
        
        # Grid spacing
        self.dr = (r_max - r_min) / (nr - 1) if nr > 1 else 0.0
        self.dθ = 2 * np.pi / nθ
        self.dζ = 2 * np.pi / nζ
        
        # Coordinates
        self.r = np.linspace(r_min, r_max, nr)
        self.θ = np.linspace(0, 2*np.pi, nθ, endpoint=False)
        self.ζ = np.linspace(0, 2*np.pi, nζ, endpoint=False)
        
        # Domain lengths
        self.Lr = r_max - r_min
        self.Lθ = 2 * np.pi
        self.Lζ = 2 * np.pi
    
    def meshgrid(self):
        """Return (r, θ, ζ) meshgrid."""
        r_3d, θ_3d, ζ_3d = np.meshgrid(self.r, self.θ, self.ζ, indexing='ij')
        return r_3d, θ_3d, ζ_3d


def test_matrix_builders():
    """Test matrix construction on small grid."""
    print("=" * 60)
    print("Test 1: Matrix Builders")
    print("=" * 60)
    
    # Small grid
    grid = Grid3D(nr=4, nθ=8, nζ=1, r_max=1.0)
    
    print(f"Grid: nr={grid.nr}, nθ={grid.nθ}, nζ={grid.nζ}")
    print(f"dr={grid.dr:.4f}, dθ={grid.dθ:.4f}")
    
    # Test radial Laplacian
    print("\nBuilding radial Laplacian...")
    D_r = build_radial_laplacian(grid.r, grid.dr, kz=0.0, bc='dirichlet')
    print(f"D_r shape: {D_r.shape}")
    print(f"D_r sparsity: {D_r.nnz}/{D_r.shape[0]*D_r.shape[1]} "
          f"= {100*D_r.nnz/(D_r.shape[0]*D_r.shape[1]):.1f}%")
    print(f"D_r matrix:\n{D_r.toarray()}")
    
    # Test poloidal Laplacian
    print("\nBuilding poloidal Laplacian...")
    D_θ = build_poloidal_laplacian(grid.nθ, grid.dθ)
    print(f"D_θ shape: {D_θ.shape}")
    print(f"D_θ sparsity: {D_θ.nnz}/{D_θ.shape[0]*D_θ.shape[1]} "
          f"= {100*D_θ.nnz/(D_θ.shape[0]*D_θ.shape[1]):.1f}%")
    print(f"D_θ first 5 rows:\n{D_θ.toarray()[:5, :]}")
    
    # Test full 2D matrix
    print("\nBuilding full 2D Laplacian matrix...")
    A = build_2d_laplacian_matrix(grid.r, grid.dr, grid.dθ, kz=0.0, bc='dirichlet')
    print(f"A shape: {A.shape}")
    print(f"A sparsity: {A.nnz}/{A.shape[0]*A.shape[1]} "
          f"= {100*A.nnz/(A.shape[0]*A.shape[1]):.1f}%")
    
    # Check symmetry
    is_symmetric = np.allclose(A.toarray(), A.toarray().T)
    print(f"A is symmetric: {is_symmetric}")
    
    print("\n✅ Matrix builders work!\n")


def test_simple_analytical():
    """Test on simple analytical solution."""
    print("=" * 60)
    print("Test 2: Simple Analytical Solution")
    print("=" * 60)
    
    # Small grid for debugging
    grid = Grid3D(nr=8, nθ=8, nζ=8, r_max=1.0)
    r, θ, ζ = grid.meshgrid()
    
    # Simple solution: φ = r(1-r) sin(θ) cos(ζ)
    # Satisfies Dirichlet BC: φ(r=0) = φ(r=1) = 0
    def phi_exact_fn(r, θ, ζ):
        return r * (1 - r) * np.sin(θ) * np.cos(ζ)
    
    phi_exact = phi_exact_fn(r, θ, ζ)
    
    print(f"Grid: {grid.nr}×{grid.nθ}×{grid.nζ}")
    print(f"φ_exact range: [{phi_exact.min():.4f}, {phi_exact.max():.4f}]")
    
    # Compute RHS: ω = ∇²φ
    print("\nComputing ω = ∇²φ_exact...")
    omega = compute_laplacian_3d(phi_exact, grid)
    print(f"ω range: [{omega.min():.4f}, {omega.max():.4f}]")
    
    # Solve ∇²φ = ω
    print("\nSolving ∇²φ = ω...")
    phi_num = solve_poisson_3d(omega, grid, bc='dirichlet')
    print(f"φ_num range: [{phi_num.min():.4f}, {phi_num.max():.4f}]")
    
    # Check solution error
    solution_error = np.max(np.abs(phi_num - phi_exact))
    print(f"\nSolution error: {solution_error:.2e}")
    
    # Check residual
    lap_phi_num = compute_laplacian_3d(phi_num, grid)
    residual_error = np.max(np.abs(lap_phi_num - omega))
    print(f"Residual error: {residual_error:.2e}")
    
    # Check BC
    bc_error_r0 = np.max(np.abs(phi_num[0, :, :]))
    bc_error_ra = np.max(np.abs(phi_num[-1, :, :]))
    print(f"BC error at r=0: {bc_error_r0:.2e}")
    print(f"BC error at r=a: {bc_error_ra:.2e}")
    
    if solution_error < 1e-4:
        print("\n✅ Small grid test PASSED!\n")
    else:
        print(f"\n❌ Small grid test FAILED (error {solution_error:.2e} > 1e-4)\n")


def test_2d_limit():
    """Test 2D limit (nζ=1)."""
    print("=" * 60)
    print("Test 3: 2D Limit (nζ=1)")
    print("=" * 60)
    
    grid = Grid3D(nr=16, nθ=16, nζ=1, r_max=1.0)
    r, θ, ζ = grid.meshgrid()
    
    # 2D solution (independent of ζ)
    def phi_exact_fn(r, θ, ζ):
        return np.sin(np.pi * r) * np.cos(2 * θ)
    
    phi_exact = phi_exact_fn(r, θ, ζ)
    
    print(f"Grid: {grid.nr}×{grid.nθ}×{grid.nζ}")
    
    # Compute and solve
    omega = compute_laplacian_3d(phi_exact, grid)
    phi_num = solve_poisson_3d(omega, grid, bc='dirichlet')
    
    # Errors
    solution_error = np.max(np.abs(phi_num - phi_exact))
    lap_phi_num = compute_laplacian_3d(phi_num, grid)
    residual_error = np.max(np.abs(lap_phi_num - omega))
    
    print(f"Solution error: {solution_error:.2e}")
    print(f"Residual error: {residual_error:.2e}")
    
    if solution_error < 1e-4 and residual_error < 1e-4:
        print("\n✅ 2D limit test PASSED!\n")
    else:
        print(f"\n❌ 2D limit test FAILED\n")


if __name__ == "__main__":
    test_matrix_builders()
    test_simple_analytical()
    test_2d_limit()
    
    print("=" * 60)
    print("Small Grid Tests Complete")
    print("=" * 60)
