#!/usr/bin/env python3
"""
Debug q积分 - 详细输出每一步

Author: 小P ⚛️
"""

import sys
sys.path.insert(0, 'src')

import numpy as np

from pytokeq.equilibrium.solver.picard_gs_solver import Grid, CoilSet, Constraints, solve_picard_free_boundary
from pytokeq.equilibrium.profiles.m3dc1_profile import M3DC1Profile
from pytokeq.equilibrium.diagnostics.q_profile import QCalculator

print("="*70)
print("Debug q积分详细过程")
print("="*70)

# 简单setup
R_min, R_max = 1.0, 2.0
Z_min, Z_max = -0.5, 0.5
nx, ny = 64, 64  # 更小grid加速

R_1d = np.linspace(R_min, R_max, nx)
Z_1d = np.linspace(Z_min, Z_max, ny)

grid = Grid.from_1d(R_1d, Z_1d)
profile = M3DC1Profile()

coils = CoilSet(R=np.array([]), Z=np.array([]), I=np.array([]))
constraints = Constraints(xpoint=[], isoflux=[])

result = solve_picard_free_boundary(
    profile=profile,
    grid=grid,
    coils=coils,
    constraints=constraints,
    max_outer=50,
    tol_psi=1e-5,
    damping=0.3
)

print(f"Equilibrium: converged={result.converged}, niter={result.niter}")

psi = result.psi
R, Z = grid.R, grid.Z

def fpol(psi_norm):
    return profile.Fpol(psi_norm)

dpsi_dR = np.gradient(psi, grid.dR, axis=0)
dpsi_dZ = np.gradient(psi, grid.dZ, axis=1)

def Br_func(R_pts, Z_pts):
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(
        (R_1d, Z_1d), (-dpsi_dZ / R).T,
        bounds_error=False, fill_value=0
    )
    return interp(np.column_stack([R_pts, Z_pts]))

def Bz_func(R_pts, Z_pts):
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(
        (R_1d, Z_1d), (dpsi_dR / R).T,
        bounds_error=False, fill_value=0
    )
    return interp(np.column_stack([R_pts, Z_pts]))

calc = QCalculator(psi, R_1d, Z_1d, fpol, Br_func, Bz_func)

# 在psi_norm=0.5计算q (详细输出)
psi_norm_test = 0.5
psi_target = calc.tracer.denormalize_psi(psi_norm_test)

print(f"\n计算 q(ψ_norm={psi_norm_test}):")
print(f"  ψ_target: {psi_target:.6f}")

# Get surface
R_surf, Z_surf = calc.tracer.find_surface_points(psi_target, ntheta=64)
print(f"  Surface points: {len(R_surf)}")
print(f"  R range: [{R_surf.min():.4f}, {R_surf.max():.4f}]")
print(f"  Z range: [{Z_surf.min():.4f}, {Z_surf.max():.4f}]")

# F(ψ)
F = fpol(psi_norm_test)
print(f"  F(ψ): {F:.4f}")

# Poloidal field
Br = Br_func(R_surf, Z_surf)
Bz = Bz_func(R_surf, Z_surf)
Btheta = np.sqrt(Br**2 + Bz**2)

print(f"  Br range: [{Br.min():.6f}, {Br.max():.6f}]")
print(f"  Bz range: [{Bz.min():.6f}, {Bz.max():.6f}]")
print(f"  Bθ range: [{Btheta.min():.6f}, {Btheta.max():.6f}]")

# Integrand
qint = F / (R_surf**2 * Btheta)
print(f"  Integrand range: [{qint.min():.6f}, {qint.max():.6f}]")
print(f"  Integrand mean: {qint.mean():.6f}")

# Arc length
dR = np.roll(R_surf, -1) - np.roll(R_surf, 1)
dZ = np.roll(Z_surf, -1) - np.roll(Z_surf, 1)
dR /= 2.0
dZ /= 2.0
dl = np.sqrt(dR**2 + dZ**2)

print(f"  dl range: [{dl.min():.6f}, {dl.max():.6f}]")
print(f"  ∑dl: {dl.sum():.6f}")

# Integral
integral = np.sum(qint * dl)
q = integral / (2 * np.pi)

print(f"\n结果:")
print(f"  ∮ (F/R²Bθ) dl: {integral:.6f}")
print(f"  q = integral/(2π): {q:.6f}")

# Sanity check
print(f"\n理论估算 (粗略):")
R_mean = R_surf.mean()
Btheta_mean = Btheta.mean()
circumference = dl.sum()

q_naive = F / (R_mean**2 * Btheta_mean) * circumference / (2*np.pi)
print(f"  F/(R²Bθ)_mean × L/(2π): {q_naive:.6f}")

# Check if units make sense
print(f"\nUnits check:")
print(f"  [F]: T·m = {F:.2f}")
print(f"  [R²]: m² ~ {R_mean**2:.2f}")
print(f"  [Bθ]: T ~ {Btheta_mean:.6f}")
print(f"  [dl]: m, ∑dl ~ {dl.sum():.2f}")
print(f"  Expected q ~ F/(R²·Bθ)·L/(2π) ~ {F/(R_mean**2 * Btheta_mean) * dl.sum()/(2*np.pi):.2f}")

print("="*70)
