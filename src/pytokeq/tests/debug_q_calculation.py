#!/usr/bin/env python3
"""
Debug q-profile calculation

Check intermediate values
"""

import sys
sys.path.insert(0, '..')

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from equilibrium.picard_gs_solver import (
    Grid, CoilSet, Constraints,
    solve_picard_free_boundary
)
from equilibrium.m3dc1_profile import M3DC1Profile
from equilibrium.q_profile import QCalculator


# Setup grid
R_min, R_max = 1.0, 2.0
Z_min, Z_max = -0.5, 0.5
nx, ny = 128, 128

R_1d = np.linspace(R_min, R_max, nx)
Z_1d = np.linspace(Z_min, Z_max, ny)

grid = Grid.from_1d(R_1d, Z_1d)

# Profile
profile = M3DC1Profile()

# No coils
coils = CoilSet(R=np.array([]), Z=np.array([]), I=np.array([]))
constraints = Constraints(xpoint=[], isoflux=[])

# Solve
print("Solving equilibrium...")
result = solve_picard_free_boundary(
    profile=profile,
    grid=grid,
    coils=coils,
    constraints=constraints,
    max_outer=30,
    tol_psi=1e-5,
    damping=0.5
)

print(f"Converged: {result.converged}")

# Extract fields
psi = result.psi
R, Z = grid.R, grid.Z

# Check psi field
print(f"\nψ field:")
print(f"  Shape: {psi.shape}")
print(f"  Min: {psi.min():.6e}")
print(f"  Max: {psi.max():.6e}")
print(f"  Mean: {psi.mean():.6e}")

# Find axis
psi_abs = np.abs(psi)
idx = np.unravel_index(np.argmax(psi_abs), psi.shape)
i_axis, j_axis = idx

print(f"\nMagnetic axis (from psi max):")
print(f"  Index: ({i_axis}, {j_axis})")
print(f"  R_axis: {R_1d[i_axis]:.4f} m")
print(f"  Z_axis: {Z_1d[j_axis]:.4f} m")
print(f"  ψ_axis: {psi[i_axis, j_axis]:.6e}")

# Edge value
boundary_vals = np.concatenate([
    psi[0, :], psi[-1, :], psi[:, 0], psi[:, -1]
])
psi_edge = np.min(boundary_vals)
print(f"  ψ_edge: {psi_edge:.6e}")

# Check gradients
dpsi_dR = np.gradient(psi, grid.dR, axis=0)
dpsi_dZ = np.gradient(psi, grid.dZ, axis=1)

print(f"\n∂ψ/∂R:")
print(f"  Shape: {dpsi_dR.shape}")
print(f"  Min: {dpsi_dR.min():.6e}")
print(f"  Max: {dpsi_dR.max():.6e}")
print(f"  At axis: {dpsi_dR[i_axis, j_axis]:.6e}")

print(f"\n∂ψ/∂Z:")
print(f"  Shape: {dpsi_dZ.shape}")
print(f"  Min: {dpsi_dZ.min():.6e}")
print(f"  Max: {dpsi_dZ.max():.6e}")
print(f"  At axis: {dpsi_dZ[i_axis, j_axis]:.6e}")

# Check B fields
B_R = -dpsi_dZ / R
B_Z = dpsi_dR / R

print(f"\nB_R = -(1/R) ∂ψ/∂Z:")
print(f"  Min: {B_R.min():.6e}")
print(f"  Max: {B_R.max():.6e}")
print(f"  At axis: {B_R[i_axis, j_axis]:.6e}")

print(f"\nB_Z = (1/R) ∂ψ/∂R:")
print(f"  Min: {B_Z.min():.6e}")
print(f"  Max: {B_Z.max():.6e}")
print(f"  At axis: {B_Z[i_axis, j_axis]:.6e}")

B_theta = np.sqrt(B_R**2 + B_Z**2)

print(f"\nB_θ = sqrt(B_R² + B_Z²):")
print(f"  Min: {B_theta.min():.6e}")
print(f"  Max: {B_theta.max():.6e}")
print(f"  At axis: {B_theta[i_axis, j_axis]:.6e}")

# Check F(psi)
print(f"\nF(ψ) function:")
psi_norm_test = np.array([0.0, 0.5, 1.0])
for pn in psi_norm_test:
    F = profile.Fpol(pn)
    print(f"  F({pn:.1f}) = {F:.4f} T·m")

# Manually compute q at axis
print(f"\n" + "="*70)
print("Manual q calculation at axis")
print("="*70)

# At axis: B_theta → 0, need to use limit
# q ≈ r·B_phi / (R·B_theta) ≈ F / (R·B_theta)
# Near axis: use r = sqrt((R-R0)^2 + (Z-Z0)^2)

# Sample a small flux surface (psi_norm = 0.1)
psi_norm_test = 0.1
psi_target = psi_edge + psi_norm_test * (psi[i_axis, j_axis] - psi_edge)

print(f"\nTest at ψ_norm = {psi_norm_test}")
print(f"  ψ_target = {psi_target:.6e}")

# Find one point on this surface manually
# Use ray shooting
R_axis = R_1d[i_axis]
Z_axis = Z_1d[j_axis]

theta_test = 0.0  # Horizontal ray
R_end = R_max
Z_end = Z_axis

n_ray = 100
R_ray = np.linspace(R_axis, R_end, n_ray)
Z_ray = np.linspace(Z_axis, Z_end, n_ray)

# Interpolate psi along ray
from scipy.interpolate import RectBivariateSpline
psi_interp = RectBivariateSpline(R_1d, Z_1d, psi)
psi_ray = psi_interp(R_ray, Z_ray, grid=False)

print(f"\nRay psi values (first 10):")
for i in range(min(10, len(psi_ray))):
    print(f"  R={R_ray[i]:.4f}, Z={Z_ray[i]:.4f}, ψ={psi_ray[i]:.6e}")

# Find where psi crosses target
ind = np.argmax(psi_ray > psi_target)
if ind == 0 and psi_ray[0] <= psi_target:
    print(f"ERROR: psi_ray[0]={psi_ray[0]:.6e} <= target={psi_target:.6e}")
    print(f"Ray didn't cross target surface!")
else:
    print(f"\nCrossing found at index {ind}")
    print(f"  psi_ray[{ind-1}] = {psi_ray[ind-1]:.6e}")
    print(f"  psi_ray[{ind}] = {psi_ray[ind]:.6e}")
    print(f"  Target = {psi_target:.6e}")
    
    # Linear interpolation
    f = (psi_ray[ind] - psi_target) / (psi_ray[ind] - psi_ray[ind-1])
    R_cross = (1-f) * R_ray[ind] + f * R_ray[ind-1]
    Z_cross = (1-f) * Z_ray[ind] + f * Z_ray[ind-1]
    
    print(f"\n  Interpolation f = {f:.4f}")
    print(f"  R_cross = {R_cross:.4f} m")
    print(f"  Z_cross = {Z_cross:.4f} m")
    
    # Evaluate B at this point
    BR_cross = float(psi_interp(R_cross, Z_cross, dx=0, dy=1, grid=False)) * (-1/R_cross)
    BZ_cross = float(psi_interp(R_cross, Z_cross, dx=1, dy=0, grid=False)) * (1/R_cross)
    Btheta_cross = np.sqrt(BR_cross**2 + BZ_cross**2)
    
    print(f"\n  B_R = {BR_cross:.6e} T")
    print(f"  B_Z = {BZ_cross:.6e} T")
    print(f"  B_θ = {Btheta_cross:.6e} T")
    
    # Toroidal field
    F = profile.Fpol(psi_norm_test)
    print(f"\n  F(ψ_norm={psi_norm_test}) = {F:.4f} T·m")
    
    # q at this point
    q_local = F / (R_cross**2 * Btheta_cross)
    print(f"\n  q_local = F/(R²B_θ) = {q_local:.4f}")
    
    # Expected q from profile
    q_expected = profile.q_profile(np.array([psi_norm_test]))[0]
    print(f"  q_expected (from profile) = {q_expected:.4f}")
    print(f"  Ratio q_local/q_expected = {q_local/q_expected:.4f}")
