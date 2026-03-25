#!/usr/bin/env python3
"""
Issue #12 Level 1 Diagnostic: 验证测试设置

检查:
1. M3D-C1平衡是否正确收敛
2. fpol/Br/Bz函数是否正确
3. 网格设置是否合理

Author: 小P ⚛️
Date: 2026-03-25
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pytokeq.equilibrium.solver.picard_gs_solver import (
    Grid, CoilSet, Constraints,
    solve_picard_free_boundary
)
from pytokeq.equilibrium.profiles.m3dc1_profile import M3DC1Profile

print("="*70)
print("Issue #12 Level 1: 测试设置诊断")
print("="*70)

# 使用与test相同的设置
R_min, R_max = 1.0, 2.0
Z_min, Z_max = -0.5, 0.5
nx, ny = 128, 128

R_1d = np.linspace(R_min, R_max, nx)
Z_1d = np.linspace(Z_min, Z_max, ny)

grid = Grid.from_1d(R_1d, Z_1d)
profile = M3DC1Profile()

print(f"\n网格设置:")
print(f"  R: [{R_min}, {R_max}] m, {nx} points")
print(f"  Z: [{Z_min}, {Z_max}] m, {ny} points")
print(f"  dR: {grid.dR:.4f} m")
print(f"  dZ: {grid.dZ:.4f} m")

# 无线圈(fixed boundary)
coils = CoilSet(R=np.array([]), Z=np.array([]), I=np.array([]))
constraints = Constraints(xpoint=[], isoflux=[])

print(f"\nProfile: {profile}")

# 求解平衡
print(f"\n{'='*70}")
print("求解M3D-C1平衡...")
print("="*70)

result = solve_picard_free_boundary(
    profile=profile,
    grid=grid,
    coils=coils,
    constraints=constraints,
    max_outer=100,  # 增加迭代次数
    tol_psi=1e-6,   # 更严格的tolerance
    damping=0.3     # 降低damping避免震荡
)

# 检查收敛性
print(f"\n{'='*70}")
print("收敛性检查:")
print("="*70)

if result.converged:
    print(f"✅ 平衡收敛")
    print(f"   迭代次数: {result.niter}")
    print(f"   最终残差: {result.residuals[-1]:.3e}")
else:
    print(f"❌ 平衡未收敛!")
    print(f"   迭代次数: {result.niter} (max: 100)")
    print(f"   最终残差: {result.residuals[-1]:.3e} (tol: 1e-6)")
    print(f"   **这可能导致q-profile计算错误!**")

# 检查残差历史
print(f"\n残差历史 (后10次迭代):")
for i, res in enumerate(result.residuals[-10:], start=len(result.residuals)-10):
    status = "✓" if i == 0 or res < result.residuals[i-1] else "↑"
    print(f"  {status} Iter {i:3d}: {res:.3e}")

# 检查psi分布
psi = result.psi
print(f"\nψ分布:")
print(f"  范围: [{psi.min():.6f}, {psi.max():.6f}]")
print(f"  是否有NaN: {np.any(np.isnan(psi))}")
print(f"  是否有Inf: {np.any(np.isinf(psi))}")

# 找磁轴
i_max, j_max = np.unravel_index(np.argmax(psi), psi.shape)
R_axis = R_1d[i_max]
Z_axis = Z_1d[j_max]
psi_axis = psi.max()

print(f"\n磁轴位置:")
print(f"  R_axis: {R_axis:.4f} m (grid index: {i_max})")
print(f"  Z_axis: {Z_axis:.4f} m (grid index: {j_max})")
print(f"  ψ_axis: {psi_axis:.6f}")

# 检查fpol函数
print(f"\n{'='*70}")
print("检查F(ψ)函数:")
print("="*70)

psi_norm_test = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
fpol_values = np.array([profile.Fpol(p) for p in psi_norm_test])

print(f"{'psi_norm':>10} {'F(psi)':>10}")
for pn, fv in zip(psi_norm_test, fpol_values):
    print(f"{pn:10.2f} {fv:10.4f}")

if np.any(fpol_values <= 0):
    print(f"⚠️  F(ψ)有负值或零! 这会导致q计算错误")
else:
    print(f"✅ F(ψ)都为正")

# 检查梯度计算
print(f"\n{'='*70}")
print("检查∇ψ计算:")
print("="*70)

dpsi_dR = np.gradient(psi, grid.dR, axis=0)
dpsi_dZ = np.gradient(psi, grid.dZ, axis=1)

print(f"∂ψ/∂R:")
print(f"  范围: [{dpsi_dR.min():.6f}, {dpsi_dR.max():.6f}]")
print(f"  NaN: {np.any(np.isnan(dpsi_dR))}")

print(f"∂ψ/∂Z:")
print(f"  范围: [{dpsi_dZ.min():.6f}, {dpsi_dZ.max():.6f}]")
print(f"  NaN: {np.any(np.isnan(dpsi_dZ))}")

# Bθ = |∇ψ|/R
R, Z = grid.R, grid.Z
Btheta = np.sqrt(dpsi_dR**2 + dpsi_dZ**2) / R

print(f"\nBθ (poloidal field):")
print(f"  范围: [{Btheta.min():.6f}, {Btheta.max():.6f}]")
print(f"  在磁轴: {Btheta[i_max, j_max]:.6f}")

# 检查q的分母是否接近零
denominator = R**2 * Btheta
print(f"\nq积分分母 (R² Bθ):")
print(f"  范围: [{denominator.min():.6f}, {denominator.max():.6f}]")

if np.any(denominator < 1e-10):
    print(f"⚠️  分母接近零! 可能导致q发散")
    n_small = np.sum(denominator < 1e-10)
    print(f"   {n_small} 个点 < 1e-10")
else:
    print(f"✅ 分母都足够大")

# 可视化
print(f"\n{'='*70}")
print("生成诊断图...")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# ψ等高线
axes[0,0].contour(R_1d, Z_1d, psi.T, levels=20, colors='blue')
axes[0,0].scatter([R_axis], [Z_axis], color='red', s=100, marker='x', label='Axis')
axes[0,0].set_xlabel('R (m)')
axes[0,0].set_ylabel('Z (m)')
axes[0,0].set_title('ψ Contours')
axes[0,0].legend()
axes[0,0].set_aspect('equal')

# ∂ψ/∂R
im1 = axes[0,1].pcolormesh(R_1d, Z_1d, dpsi_dR.T, shading='auto', cmap='RdBu_r')
axes[0,1].scatter([R_axis], [Z_axis], color='black', s=50, marker='x')
axes[0,1].set_xlabel('R (m)')
axes[0,1].set_ylabel('Z (m)')
axes[0,1].set_title('∂ψ/∂R')
plt.colorbar(im1, ax=axes[0,1])

# ∂ψ/∂Z
im2 = axes[0,2].pcolormesh(R_1d, Z_1d, dpsi_dZ.T, shading='auto', cmap='RdBu_r')
axes[0,2].scatter([R_axis], [Z_axis], color='black', s=50, marker='x')
axes[0,2].set_xlabel('R (m)')
axes[0,2].set_ylabel('Z (m)')
axes[0,2].set_title('∂ψ/∂Z')
plt.colorbar(im2, ax=axes[0,2])

# Bθ
im3 = axes[1,0].pcolormesh(R_1d, Z_1d, Btheta.T, shading='auto', cmap='viridis')
axes[1,0].scatter([R_axis], [Z_axis], color='red', s=50, marker='x')
axes[1,0].set_xlabel('R (m)')
axes[1,0].set_ylabel('Z (m)')
axes[1,0].set_title('Bθ (poloidal field)')
plt.colorbar(im3, ax=axes[1,0])

# 收敛历史
axes[1,1].semilogy(result.residuals, 'b-')
axes[1,1].axhline(y=1e-6, color='r', linestyle='--', label='Target tol')
axes[1,1].set_xlabel('Iteration')
axes[1,1].set_ylabel('Residual')
axes[1,1].set_title('Convergence History')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

# F(ψ)函数
psi_norm_plot = np.linspace(0, 1, 100)
fpol_plot = np.array([profile.Fpol(p) for p in psi_norm_plot])
axes[1,2].plot(psi_norm_plot, fpol_plot, 'b-')
axes[1,2].set_xlabel('ψ_norm')
axes[1,2].set_ylabel('F(ψ)')
axes[1,2].set_title('Toroidal Field Function')
axes[1,2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/issue12_level1_equilibrium.png', dpi=150)
print(f"✅ 保存到: /tmp/issue12_level1_equilibrium.png")

# Summary
print(f"\n{'='*70}")
print("Level 1 诊断总结:")
print("="*70)

issues = []

if not result.converged:
    issues.append("❌ 平衡未收敛")
    
if np.any(fpol_values <= 0):
    issues.append("❌ F(ψ)有非正值")
    
if np.any(denominator < 1e-10):
    issues.append("⚠️  q积分分母接近零")

if len(issues) == 0:
    print("✅ 测试设置看起来正常")
    print("   问题可能在q计算算法本身 → 进入Level 2")
else:
    print("发现问题:")
    for issue in issues:
        print(f"  {issue}")
    print("\n需要先修复这些问题!")

print("="*70)
