"""
Numerical operators for Reduced MHD

Implements:
- Grad-Shafranov operator (Δ*)
- Poisson bracket ([f,g])
- Gradient, divergence operators

Author: 小P ⚛️
Date: 2026-03-11
"""

import numpy as np
from scipy.sparse import csr_matrix, diags
from typing import Tuple


def build_grad_shafranov_operator(R: np.ndarray, Z: np.ndarray) -> csr_matrix:
    """
    Build Grad-Shafranov operator matrix Δ* for toroidal geometry.
    
    Δ*ψ = ∂²ψ/∂R² - (1/R)∂ψ/∂R + ∂²ψ/∂Z²
    
    Discretized on uniform (R, Z) grid using 5-point stencil.
    
    Parameters
    ----------
    R : np.ndarray, shape (Nr,)
        Major radius grid points
    Z : np.ndarray, shape (Nz,)
        Vertical coordinate grid points
    
    Returns
    -------
    L : scipy.sparse.csr_matrix, shape (Nr*Nz, Nr*Nz)
        Sparse matrix such that L @ psi_flat = (Δ*ψ)_flat
        where psi_flat is ψ(R,Z) flattened in row-major (C) order
    
    Notes
    -----
    Boundary conditions are NOT enforced in this matrix.
    Apply boundary conditions separately after construction.
    
    Grid ordering: psi[i,j] → psi_flat[i*Nz + j]
    where i ∈ [0, Nr-1], j ∈ [0, Nz-1]
    
    Stencil at interior point (i,j):
        (Δ*ψ)ij = [ψi+1,j - 2ψij + ψi-1,j] / ΔR²
                 - [ψi+1,j - ψi-1,j] / (2*Ri*ΔR)
                 + [ψi,j+1 - 2ψij + ψi,j-1] / ΔZ²
    
    References
    ----------
    Grad & Shafranov (1958) - Original equilibrium equation
    M3D-C1 User's Guide Ch.8 - Discretization
    """
    Nr = len(R)
    Nz = len(Z)
    N = Nr * Nz
    
    # Grid spacing (uniform grid assumed)
    dR = R[1] - R[0]
    dZ = Z[1] - Z[0]
    
    # Check uniformity
    assert np.allclose(np.diff(R), dR, rtol=1e-10), "R grid must be uniform"
    assert np.allclose(np.diff(Z), dZ, rtol=1e-10), "Z grid must be uniform"
    
    # Build in COO format (triplet), then convert to CSR
    row_indices = []
    col_indices = []
    data = []
    
    def add_entry(row, col, value):
        """Helper to add matrix entry"""
        row_indices.append(row)
        col_indices.append(col)
        data.append(value)
    
    # Loop over all grid points
    for i in range(Nr):
        for j in range(Nz):
            # Flat index for current point
            idx = i * Nz + j
            
            # R coordinate at this point
            Ri = R[i]
            
            # Determine boundary status
            at_R_lower = (i == 0)
            at_R_upper = (i == Nr - 1)
            at_Z_lower = (j == 0)
            at_Z_upper = (j == Nz - 1)
            
            # === R-direction contributions ===
            if at_R_lower:
                # Forward difference: ∂²/∂R², ∂/∂R
                # 2nd order forward for 1st derivative
                c_R0 = 1.0/dR**2 + 3.0/(2*Ri*dR)
                add_entry(idx, idx, c_R0)
                
                if i+1 < Nr:
                    c_R1 = -2.0/dR**2 - 4.0/(2*Ri*dR)
                    add_entry(idx, (i+1)*Nz + j, c_R1)
                
                if i+2 < Nr:
                    c_R2 = 1.0/dR**2 + 1.0/(2*Ri*dR)
                    add_entry(idx, (i+2)*Nz + j, c_R2)
                    
            elif at_R_upper:
                # Backward difference
                c_R0 = 1.0/dR**2 - 3.0/(2*Ri*dR)
                add_entry(idx, idx, c_R0)
                
                if i-1 >= 0:
                    c_R1 = -2.0/dR**2 + 4.0/(2*Ri*dR)
                    add_entry(idx, (i-1)*Nz + j, c_R1)
                
                if i-2 >= 0:
                    c_R2 = 1.0/dR**2 - 1.0/(2*Ri*dR)
                    add_entry(idx, (i-2)*Nz + j, c_R2)
                    
            else:
                # Interior: central difference
                c_R0 = -2.0/dR**2
                add_entry(idx, idx, c_R0)
                
                c_Rp = 1.0/dR**2 - 1.0/(2*Ri*dR)
                add_entry(idx, (i+1)*Nz + j, c_Rp)
                
                c_Rm = 1.0/dR**2 + 1.0/(2*Ri*dR)
                add_entry(idx, (i-1)*Nz + j, c_Rm)
            
            # === Z-direction contributions ===
            if at_Z_lower:
                # Forward 2nd derivative
                c_Z0 = 1.0/dZ**2
                add_entry(idx, idx, c_Z0)
                
                if j+1 < Nz:
                    c_Z1 = -2.0/dZ**2
                    add_entry(idx, i*Nz + (j+1), c_Z1)
                
                if j+2 < Nz:
                    c_Z2 = 1.0/dZ**2
                    add_entry(idx, i*Nz + (j+2), c_Z2)
                    
            elif at_Z_upper:
                # Backward 2nd derivative
                c_Z0 = 1.0/dZ**2
                add_entry(idx, idx, c_Z0)
                
                if j-1 >= 0:
                    c_Z1 = -2.0/dZ**2
                    add_entry(idx, i*Nz + (j-1), c_Z1)
                
                if j-2 >= 0:
                    c_Z2 = 1.0/dZ**2
                    add_entry(idx, i*Nz + (j-2), c_Z2)
                    
            else:
                # Interior: central 2nd derivative
                c_Z0 = -2.0/dZ**2
                add_entry(idx, idx, c_Z0)
                
                c_Zp = 1.0/dZ**2
                add_entry(idx, i*Nz + (j+1), c_Zp)
                
                c_Zm = 1.0/dZ**2
                add_entry(idx, i*Nz + (j-1), c_Zm)
    
    # Convert to CSR format (efficient for matrix-vector multiply)
    L = csr_matrix((data, (row_indices, col_indices)), shape=(N, N))
    
    return L


def apply_grad_shafranov_operator(psi: np.ndarray, R: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    Apply Grad-Shafranov operator to ψ using finite differences.
    
    Direct computation without building matrix (for testing/validation).
    
    Parameters
    ----------
    psi : np.ndarray, shape (Nr, Nz)
        Poloidal flux field
    R : np.ndarray, shape (Nr,)
        Major radius grid
    Z : np.ndarray, shape (Nz,)
        Vertical grid
    
    Returns
    -------
    result : np.ndarray, shape (Nr, Nz)
        Δ*ψ computed via finite differences
    """
    Nr, Nz = psi.shape
    dR = R[1] - R[0]
    dZ = Z[1] - Z[0]
    
    result = np.zeros_like(psi)
    
    # Interior points only
    for i in range(1, Nr-1):
        for j in range(1, Nz-1):
            Ri = R[i]
            
            # ∂²ψ/∂R²
            d2psi_dR2 = (psi[i+1, j] - 2*psi[i, j] + psi[i-1, j]) / dR**2
            
            # ∂ψ/∂R (central difference)
            dpsi_dR = (psi[i+1, j] - psi[i-1, j]) / (2*dR)
            
            # ∂²ψ/∂Z²
            d2psi_dZ2 = (psi[i, j+1] - 2*psi[i, j] + psi[i, j-1]) / dZ**2
            
            # Δ*ψ = ∂²ψ/∂R² - (1/R)∂ψ/∂R + ∂²ψ/∂Z²
            result[i, j] = d2psi_dR2 - dpsi_dR / Ri + d2psi_dZ2
    
    return result


def compute_poisson_bracket_arakawa(f: np.ndarray, g: np.ndarray, 
                                     dR: float, dZ: float) -> np.ndarray:
    """
    Compute Poisson bracket [f,g] using Arakawa (1966) scheme.
    
    [f,g] = ∂f/∂R ∂g/∂Z - ∂f/∂Z ∂g/∂R
    
    Arakawa scheme conserves energy and enstrophy (9-point stencil).
    
    Parameters
    ----------
    f : np.ndarray, shape (Nr, Nz)
        First field
    g : np.ndarray, shape (Nr, Nz)
        Second field
    dR : float
        Grid spacing in R
    dZ : float
        Grid spacing in Z
    
    Returns
    -------
    result : np.ndarray, shape (Nr, Nz)
        [f,g] at interior points (boundaries set to 0)
    
    Notes
    -----
    This is the J++ form from Arakawa (1966), Eq. 2.16.
    It is 2nd order accurate and conserves both energy and enstrophy.
    
    References
    ----------
    Arakawa, A. (1966). Computational design for long-term numerical
    integration of the equations of fluid motion. J. Comp. Phys. 1, 119-143.
    """
    Nr, Nz = f.shape
    result = np.zeros_like(f)
    
    # Interior points only (need ±1 neighbors)
    i = slice(1, Nr-1)
    j = slice(1, Nz-1)
    
    # Shortcuts for neighbor access
    # f at (i±1, j±1)
    f_ip_j = f[2:, 1:-1]    # f[i+1, j]
    f_im_j = f[:-2, 1:-1]   # f[i-1, j]
    f_i_jp = f[1:-1, 2:]    # f[i, j+1]
    f_i_jm = f[1:-1, :-2]   # f[i, j-1]
    f_ip_jp = f[2:, 2:]     # f[i+1, j+1]
    f_ip_jm = f[2:, :-2]    # f[i+1, j-1]
    f_im_jp = f[:-2, 2:]    # f[i-1, j+1]
    f_im_jm = f[:-2, :-2]   # f[i-1, j-1]
    
    # g at (i±1, j±1)
    g_ip_j = g[2:, 1:-1]
    g_im_j = g[:-2, 1:-1]
    g_i_jp = g[1:-1, 2:]
    g_i_jm = g[1:-1, :-2]
    g_ip_jp = g[2:, 2:]
    g_ip_jm = g[2:, :-2]
    g_im_jp = g[:-2, 2:]
    g_im_jm = g[:-2, :-2]
    
    # Arakawa J++ scheme (Eq. 2.16 from paper)
    # [f,g] = (1/12ΔRΔZ) × [J1 + J2 + J3]
    
    # J1: Simple centered differences
    J1 = ((f_ip_j - f_im_j) * (g_i_jp - g_i_jm)
          - (f_i_jp - f_i_jm) * (g_ip_j - g_im_j))
    
    # J2: Upper diagonal
    J2 = (f_ip_j * (g_ip_jp - g_ip_jm)
          - f_im_j * (g_im_jp - g_im_jm)
          - f_i_jp * (g_ip_jp - g_im_jp)
          + f_i_jm * (g_ip_jm - g_im_jm))
    
    # J3: Lower diagonal
    J3 = (f_ip_jp * (g_i_jp - g_ip_j)
          - f_im_jm * (g_im_j - g_i_jm)
          - f_im_jp * (g_i_jp - g_im_j)
          + f_ip_jm * (g_ip_j - g_i_jm))
    
    # Combine with normalization
    result[1:-1, 1:-1] = (J1 + J2 + J3) / (12 * dR * dZ)
    
    return result


# Convenience function for backward compatibility
def poisson_bracket(f: np.ndarray, g: np.ndarray, 
                   dR: float, dZ: float) -> np.ndarray:
    """Alias for compute_poisson_bracket_arakawa"""
    return compute_poisson_bracket_arakawa(f, g, dR, dZ)
