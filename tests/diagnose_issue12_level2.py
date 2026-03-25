#!/usr/bin/env python3
"""
Issue #12 Level 2: 检查q计算算法

用解析解验证QCalculator是否正确:
1. 圆形截面 (q已知)
2. 检查flux surface tracer
3. 检查积分算法

Author: 小P ⚛️
Date: 2026-03-25
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pytokeq.equilibrium.diagnostics.q_profile import QCalculator

print("="*70)
print("Issue #12 Level 2: q计算算法验证")
print("="*70)

# Case 1: 圆形截面, constant q
# Theory: q = r B_z / (R_0 B_θ)
# For circular: q ~ (r/R_0) * (B_z / B_θ)

print("\n" + "="*70)
print("Case 1: 圆形截面 (q应为常数)")
print("="*70)

# 设置
R0 = 1.5  # Major radius
a = 0.4   # Minor radius
q0 = 2.0  # Target q

nr, nz = 65, 65
R_1d = np.linspace(R0 - a - 0.1, R0 + a + 0.1, nr)
Z_1d = np.linspace(-a - 0.1, a + 0.1, nz)

R, Z = np.meshgrid(R_1d, Z_1d, indexing='ij')

# 圆形flux surfaces
r = np.sqrt((R - R0)**2 + Z**2)
psi = np.maximum(1.0 - (r/a)**2, 0)  # parabolic

# For circular geometry:
# B_θ ∝ r (linear with minor radius)
# B_z = constant
# q = (r/R0) * (B_z/B_θ) ≈ constant for small ε

# 定义fields
# ψ = ψ0 (1 - r²/a²)
# B_θ = |∇ψ|/R = (2ψ0/a²) r / R

# For constant q, need F(ψ) such that:
# q = F/(R²B_θ) integrated over surface = constant

# Simplest: F = const
F0 = R0 * 1.0  # Bφ = F/R ≈ B0 at R0

def fpol_circular(psi_norm):
    """Constant F"""
    return F0

# Compute gradients
dpsi_dR = np.gradient(psi, R_1d[1] - R_1d[0], axis=0)
dpsi_dZ = np.gradient(psi, Z_1d[1] - Z_1d[0], axis=1)

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

# Create calculator
calc = QCalculator(psi, R_1d, Z_1d, fpol_circular, Br_func, Bz_func)

print(f"\n设置:")
print(f"  R0: {R0} m")
print(f"  a: {a} m")
print(f"  F(ψ): {F0:.2f} (constant)")

# Compute q profile
try:
    psi_norm, q = calc.compute_q_profile(npsi=20, ntheta=64, extrapolate=False)
    
    print(f"\nq-profile结果:")
    print(f"{'psi_norm':>10} {'q':>10}")
    for i in range(min(10, len(psi_norm))):
        print(f"{psi_norm[i]:10.3f} {q[i]:10.3f}")
    
    # Check if q is approximately constant
    q_mean = np.mean(q[np.isfinite(q)])
    q_std = np.std(q[np.isfinite(q)])
    q_variation = q_std / q_mean if q_mean > 0 else np.inf
    
    print(f"\nq统计:")
    print(f"  平均值: {q_mean:.3f}")
    print(f"  标准差: {q_std:.3f}")
    print(f"  变化率: {q_variation*100:.1f}%")
    
    if q_variation < 0.2:
        print(f"✅ q近似常数 (变化<20%)")
    else:
        print(f"⚠️  q变化较大 ({q_variation*100:.1f}%)")
    
    # Check for unphysical values
    if np.any(q > 100):
        print(f"❌ q有异常大值 (max={q.max():.1f})")
    else:
        print(f"✅ q值在合理范围")
    
except Exception as e:
    print(f"❌ q计算失败: {e}")
    import traceback
    traceback.print_exc()

# Visualization
print(f"\n生成诊断图...")

fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# ψ contours
axes[0,0].contour(R_1d, Z_1d, psi.T, levels=20, colors='blue')
circle = plt.Circle((R0, 0), a, fill=False, color='red', linestyle='--', label='LCFS')
axes[0,0].add_patch(circle)
axes[0,0].set_xlabel('R (m)')
axes[0,0].set_ylabel('Z (m)')
axes[0,0].set_title('ψ Contours (Circular)')
axes[0,0].legend()
axes[0,0].set_aspect('equal')

# Bθ
Btheta = np.sqrt(dpsi_dR**2 + dpsi_dZ**2) / R
im = axes[0,1].pcolormesh(R_1d, Z_1d, Btheta.T, shading='auto', cmap='viridis')
axes[0,1].set_xlabel('R (m)')
axes[0,1].set_ylabel('Z (m)')
axes[0,1].set_title('Bθ')
plt.colorbar(im, ax=axes[0,1])

# q profile
if 'q' in locals():
    axes[1,0].plot(psi_norm, q, 'bo-', label='Computed')
    axes[1,0].axhline(y=q_mean, color='r', linestyle='--', label=f'Mean={q_mean:.2f}')
    axes[1,0].set_xlabel('ψ_norm')
    axes[1,0].set_ylabel('q')
    axes[1,0].set_title('q Profile')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/issue12_level2_circular.png', dpi=150)
print(f"✅ 保存到: /tmp/issue12_level2_circular.png")

# Summary
print(f"\n" + "="*70)
print("Level 2 诊断总结:")
print("="*70)

if 'q' in locals():
    if q_variation < 0.2 and np.all(q < 100):
        print("✅ q计算器在圆形case工作正常")
        print("   问题可能在M3D-C1特定geometry → Level 3")
    else:
        print("❌ q计算器在圆形case已有问题!")
        print(f"   需要debug QCalculator算法")
else:
    print("❌ q计算失败,需要debug QCalculator")

print("="*70)
