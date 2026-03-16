#!/usr/bin/env python3
"""
Debug q-profile calculation v2

Use FluxSurfaceTracer correctly
"""

import sys
sys.path.insert(0, '..')

import numpy as np

from equilibrium.picard_gs_solver import (
    Grid, CoilSet, Constraints,
    solve_picard_free_boundary
)
from equilibrium.m3dc1_profile import M3DC1Profile
from equilibrium.flux_surface_tracer import FluxSurfaceTracer


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

# Create tracer
tracer = FluxSurfaceTracer(psi, R_1d, Z_1d)

print(f"\nFluxSurfaceTracer:")
print(f"  R_axis = {tracer.R_axis:.4f} m")
print(f"  Z_axis = {tracer.Z_axis:.4f} m")
print(f"  ψ_axis = {tracer.psi_axis:.6e}")
print(f"  ψ_edge = {tracer.psi_edge:.6e}")

# Test normalize/denormalize
psi_norm_test = np.array([0.0, 0.1, 0.5, 0.9, 1.0])

print(f"\nNormalization test:")
print(f"  psi_norm  ->  psi_absolute")
for pn in psi_norm_test:
    psi_abs = tracer.denormalize_psi(pn)
    psi_norm_back = tracer.normalize_psi(psi_abs)
    print(f"  {pn:.1f}         {psi_abs:.6e}  (back: {psi_norm_back:.4f})")

# Now test find_surface_points
print(f"\n" + "="*70)
print("Test find_surface_points at psi_norm=0.1")
print("="*70)

psi_norm = 0.1
psi_target = tracer.denormalize_psi(psi_norm)

print(f"\nTarget:")
print(f"  psi_norm = {psi_norm}")
print(f"  psi_target = {psi_target:.6e}")

R_surf, Z_surf = tracer.find_surface_points(psi_target, ntheta=64)

print(f"\nFound {len(R_surf)} points on surface")

if len(R_surf) > 0:
    print(f"\nFirst 5 points:")
    for i in range(min(5, len(R_surf))):
        print(f"  R={R_surf[i]:.4f}, Z={Z_surf[i]:.4f}")
    
    # Check psi values at these points
    from scipy.interpolate import RectBivariateSpline
    psi_interp = RectBivariateSpline(R_1d, Z_1d, psi)
    
    psi_surf = psi_interp(R_surf, Z_surf, grid=False)
    psi_norm_surf = tracer.normalize_psi(psi_surf)
    
    print(f"\nψ values at surface points:")
    print(f"  Mean ψ_norm = {psi_norm_surf.mean():.6f} (target: {psi_norm:.6f})")
    print(f"  Std  ψ_norm = {psi_norm_surf.std():.6e}")
    print(f"  Min  ψ_norm = {psi_norm_surf.min():.6f}")
    print(f"  Max  ψ_norm = {psi_norm_surf.max():.6f}")
    
    # Check if we're on the right surface
    error = abs(psi_norm_surf.mean() - psi_norm)
    print(f"\n  Error: {error:.6e}")
    
    if error < 0.01:
        print("  ✅ Surface location correct!")
    else:
        print(f"  ❌ Surface location wrong! (error={error:.2%})")

# Now test q calculation
print(f"\n" + "="*70)
print("Test q calculation")
print("="*70)

from equilibrium.q_profile import QCalculator

# Setup B field functions
dpsi_dR = np.gradient(psi, grid.dR, axis=0)
dpsi_dZ = np.gradient(psi, grid.dZ, axis=1)

R, Z = grid.R, grid.Z

from scipy.interpolate import RegularGridInterpolator

# B_R = -(1/R) ∂ψ/∂Z
B_R_grid = -dpsi_dZ / R
# B_Z = (1/R) ∂ψ/∂R
B_Z_grid = dpsi_dR / R

def Br_func(R_pts, Z_pts):
    interp = RegularGridInterpolator(
        (R_1d, Z_1d), B_R_grid.T,
        bounds_error=False, fill_value=0
    )
    return interp(np.column_stack([R_pts, Z_pts]))

def Bz_func(R_pts, Z_pts):
    interp = RegularGridInterpolator(
        (R_1d, Z_1d), B_Z_grid.T,
        bounds_error=False, fill_value=0
    )
    return interp(np.column_stack([R_pts, Z_pts]))

def fpol(psi_norm):
    return profile.Fpol(psi_norm)

# Create calculator
calc = QCalculator(psi, R_1d, Z_1d, fpol, Br_func, Bz_func)

# Compute q at psi_norm=0.1
q_test = calc.compute_q_single(0.1, ntheta=64)
q_expected = profile.q_profile(np.array([0.1]))[0]

print(f"\nq at psi_norm=0.1:")
print(f"  Computed: {q_test:.4f}")
print(f"  Expected: {q_expected:.4f}")
print(f"  Ratio:    {q_test/q_expected:.4f}")

# Compute full profile
psi_norm_arr, q_arr = calc.compute_q_profile(npsi=20, ntheta=64, extrapolate=True)

print(f"\nFull q-profile (first/last 5 points):")
print(f"  psi_norm    q_computed    q_expected    ratio")
for i in [0, 1, 2, 3, 4, 15, 16, 17, 18, 19]:
    pn = psi_norm_arr[i]
    qc = q_arr[i]
    qe = profile.q_profile(np.array([pn]))[0]
    ratio = qc / qe if qe > 0 else np.inf
    print(f"  {pn:8.4f}    {qc:10.4f}    {qe:10.4f}    {ratio:8.4f}")
