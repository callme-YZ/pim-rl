"""Debug the 2D matrix construction."""

import numpy as np
from scipy.sparse import diags, kron, eye, csr_matrix
import sys
sys.path.insert(0, 'src')


def build_radial_laplacian_debug(r, dr, kz, bc):
    """Debug version with print statements."""
    nr = len(r)
    r_safe = np.where(r > 1e-14, r, 1e-14)
    
    a = np.zeros(nr - 1)
    b = np.zeros(nr)
    c = np.zeros(nr - 1)
    
    # Interior points
    for i in range(1, nr - 1):
        a[i-1] = 1.0 / (dr**2) - 1.0 / (2.0 * r_safe[i] * dr)
        b[i] = -2.0 / (dr**2) - kz**2
        c[i] = 1.0 / (dr**2) + 1.0 / (2.0 * r_safe[i] * dr)
    
    print(f"Before BC:")
    print(f"  a (lower): {a}")
    print(f"  b (main):  {b}")
    print(f"  c (upper): {c}")
    
    # Boundary conditions
    if bc == 'dirichlet':
        # r=0: φ_0 = 0
        b[0] = 1.0
        c[0] = 0.0
        
        # r=a: φ_{nr-1} = 0
        a[-1] = 0.0
        b[-1] = 1.0
    
    print(f"After BC:")
    print(f"  a (lower): {a}")
    print(f"  b (main):  {b}")
    print(f"  c (upper): {c}")
    
    D_r = diags([a, b, c], offsets=[-1, 0, 1], shape=(nr, nr), format='csr')
    
    print(f"\nD_r matrix:")
    print(D_r.toarray())
    
    return D_r


def test_kronecker_product():
    """Test Kronecker product ordering."""
    print("=" * 60)
    print("Testing Kronecker Product Ordering")
    print("=" * 60)
    
    nr, nθ = 3, 4
    
    # Simple matrices
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    
    # Kronecker product
    K = np.kron(A, B)
    
    print(f"A ({A.shape}):")
    print(A)
    print(f"\nB ({B.shape}):")
    print(B)
    print(f"\nkron(A, B) ({K.shape}):")
    print(K)
    
    # Interpret indices
    print("\nIndex mapping (kron(A,B)[i,j]):")
    print("  i = i_A * nb + i_B")
    print("  j = j_A * nb + j_B")
    print("  Value = A[i_A, j_A] * B[i_B, j_B]")
    
    # Test with identity
    I_r = eye(nr, format='csr')
    I_θ = eye(nθ, format='csr')
    
    # D_r ⊗ I_θ: radial operator applied to each θ
    # Flattening order: if we flatten (r,θ) as r*nθ + θ (row-major)
    # then D_r ⊗ I_θ couples points with same θ, different r
    
    print(f"\n\nIdentity test:")
    print(f"I_r ({nr}×{nr}):")
    print(I_r.toarray())
    print(f"\nI_θ ({nθ}×{nθ}):")
    print(I_θ.toarray()[:5, :5])
    
    A_test = kron(I_r, I_θ, format='csr')
    print(f"\nkron(I_r, I_θ) ({A_test.shape}):")
    print(f"  Should be identity {nr*nθ}×{nr*nθ}")
    print(f"  Is identity: {np.allclose(A_test.toarray(), np.eye(nr*nθ))}")


def test_flatten_reshape():
    """Test flatten/reshape consistency."""
    print("\n" + "=" * 60)
    print("Testing Flatten/Reshape Order")
    print("=" * 60)
    
    nr, nθ = 3, 4
    
    # Create test array
    arr = np.arange(nr * nθ).reshape((nr, nθ))
    
    print(f"Array ({nr}×{nθ}):")
    print(arr)
    
    # Flatten row-major (C order)
    flat_C = arr.flatten(order='C')
    print(f"\nFlatten C-order (row-major): {flat_C}")
    print("  Interpretation: r=0 all θ, then r=1 all θ, ...")
    
    # Flatten column-major (F order)
    flat_F = arr.flatten(order='F')
    print(f"\nFlatten F-order (column-major): {flat_F}")
    print("  Interpretation: θ=0 all r, then θ=1 all r, ...")
    
    # Reshape back
    arr_C = flat_C.reshape((nr, nθ), order='C')
    arr_F = flat_F.reshape((nr, nθ), order='F')
    
    print(f"\nReshape C-order matches: {np.array_equal(arr, arr_C)}")
    print(f"Reshape F-order matches: {np.array_equal(arr, arr_F)}")
    
    # Conclusion
    print("\n⚠️  CRITICAL: Must use same order for flatten AND reshape!")
    print("    Our choice: F-order (θ fast, r slow)")


def test_bc_application():
    """Test how to apply BC to 2D matrix."""
    print("\n" + "=" * 60)
    print("Testing Boundary Condition Application")
    print("=" * 60)
    
    nr, nθ = 4, 8
    
    # BC: φ=0 at r=0 and r=a for all θ
    # In flattened F-order: θ changes fast, r slow
    # Indices [0, 1, ..., nθ-1] correspond to (r=0, θ=0..nθ-1)
    # Indices [(nr-1)*nθ, ..., nr*nθ-1] correspond to (r=a, θ=0..nθ-1)
    
    print(f"Grid: {nr}×{nθ} = {nr*nθ} total points")
    print(f"\nF-order (column-major) index mapping:")
    
    for idx_flat in [0, 1, nθ-1, nθ, nθ+1, (nr-1)*nθ, (nr-1)*nθ+1, nr*nθ-1]:
        r_idx = idx_flat // nθ
        θ_idx = idx_flat % nθ
        print(f"  flat_idx={idx_flat:2d} → (r={r_idx}, θ={θ_idx})")
    
    # BC rows to modify
    bc_rows_r0 = list(range(nθ))  # r=0
    bc_rows_ra = list(range((nr-1)*nθ, nr*nθ))  # r=a
    
    print(f"\nBC rows for r=0: {bc_rows_r0}")
    print(f"BC rows for r=a: {bc_rows_ra}")
    
    print("\n✅ BC application strategy:")
    print("  1. Build full 2D matrix A")
    print("  2. For each BC row i:")
    print("     - Set A[i, :] = 0")
    print("     - Set A[i, i] = 1")
    print("  3. Set RHS[i] = 0")


if __name__ == "__main__":
    test_kronecker_product()
    test_flatten_reshape()
    test_bc_application()
