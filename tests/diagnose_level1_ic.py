#!/usr/bin/env python3
"""
Level 1: Verify Tearing IC Properties

Check if IC satisfies theoretical requirements for tearing instability.

Author: 小P ⚛️
Date: 2026-03-25
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pim_rl.physics.v2.tearing_ic import create_tearing_ic

print("="*60)
print("Level 1: Tearing IC Verification")
print("="*60)

# Create IC
nr, ntheta = 32, 64
psi, phi = create_tearing_ic(nr, ntheta, lam=0.1, m=2, eps=0.05, eta=0.05)

# Theory: Tearing instability requires:
# 1. Current sheet (J_z reversal)
# 2. Magnetic shear (q varies with r)
# 3. Rational surface (q=m/n somewhere)

print("\n1. Current Sheet Check")
print("-"*60)

# J_z ≈ ∇²φ (in 2D)
# Or approximately: J_z ~ ∂phi/∂r at r=r0
dr = 1.0 / nr
dphi_dr = np.gradient(phi, dr, axis=0)

# Find current sheet location (where dphi/dr changes sign)
r_grid = np.linspace(0, 1, nr)
dphi_dr_r = dphi_dr[:, 0]  # Take theta=0 slice

sign_changes = np.where(np.diff(np.sign(dphi_dr_r)))[0]
if len(sign_changes) > 0:
    r_sheet = r_grid[sign_changes[0]]
    print(f"✅ Current sheet found at r ≈ {r_sheet:.3f}")
    print(f"   dphi/dr changes sign: {dphi_dr_r[sign_changes[0]]:.6f} → {dphi_dr_r[sign_changes[0]+1]:.6f}")
else:
    print(f"❌ No current sheet found!")
    print(f"   dphi/dr range: [{dphi_dr_r.min():.6f}, {dphi_dr_r.max():.6f}]")

print(f"\nCurrent profile:")
print(f"  J_z ~ dphi/dr")
print(f"  min(dphi/dr): {dphi_dr.min():.6f}")
print(f"  max(dphi/dr): {dphi_dr.max():.6f}")
print(f"  Reversal? {dphi_dr.min() < 0 and dphi_dr.max() > 0}")

# 2. Safety factor q(r)
print("\n2. Safety Factor q(r)")
print("-"*60)

# q ~ r Bz / Bθ ~ r * ∂psi/∂r (approximately)
dpsi_dr = np.gradient(psi, dr, axis=0)
# Avoid r=0
r_grid_safe = r_grid[1:]
dpsi_dr_safe = dpsi_dr[1:, 0]
q_profile = r_grid_safe * dpsi_dr_safe

print(f"q(r) profile:")
print(f"  q range: [{q_profile.min():.3f}, {q_profile.max():.3f}]")
print(f"  q(r=0.5) ≈ {q_profile[nr//2]:.3f}")

# For m=2 tearing, need q≈2 somewhere
q_target = 2.0
q_diff = np.abs(q_profile - q_target)
if q_diff.min() < 0.5:
    i_res = np.argmin(q_diff)
    r_res = r_grid_safe[i_res]
    q_res = q_profile[i_res]
    print(f"✅ Rational surface q≈{q_target} found at r≈{r_res:.3f} (q={q_res:.3f})")
else:
    print(f"⚠️  No rational surface q≈{q_target} found")
    print(f"   Closest q: {q_profile[np.argmin(q_diff)]:.3f} at r={r_grid_safe[np.argmin(q_diff)]:.3f}")

# 3. Perturbation structure
print("\n3. Perturbation Structure (m=2)")
print("-"*60)

# FFT in theta direction
psi_fft = np.fft.fft(psi, axis=1)
phi_fft = np.fft.fft(phi, axis=1)

# m=2 mode amplitude
m = 2
psi_m2 = np.abs(psi_fft[:, m])
phi_m2 = np.abs(phi_fft[:, m])

print(f"m={m} mode amplitudes:")
print(f"  max|ψ_m={m}|: {psi_m2.max():.6f}")
print(f"  max|φ_m={m}|: {phi_m2.max():.6f}")

# Check if m=2 is dominant
total_modes = np.abs(psi_fft).sum(axis=1).mean()
m2_fraction = psi_m2.mean() / (total_modes / ntheta)
print(f"  m={m} fraction of total: {m2_fraction:.3f}")

if m2_fraction > 0.1:
    print(f"✅ m={m} mode is significant")
else:
    print(f"⚠️  m={m} mode is weak (fraction={m2_fraction:.3f})")

# 4. Theoretical growth rate estimate
print("\n4. Theoretical Growth Rate Estimate")
print("-"*60)

# Furth-Killeen-Rosenbluth formula (FKR 1963):
# γ ~ η^(3/5) * (stuff)^(2/5)
# For rough estimate: γ ~ 10-100 for typical parameters

eta = 0.05
L = 0.1  # Current sheet width
V_A = 1.0  # Alfvén speed (order of magnitude)

# Resistive time
tau_R = L**2 / eta
# Alfvén time  
tau_A = L / V_A

# FKR scaling
gamma_FKR = (tau_R / tau_A)**(-2/5) / tau_A
gamma_FKR_approx = (eta / L**2)**(3/5) * (V_A / L)**(2/5)

print(f"Parameters:")
print(f"  η: {eta}")
print(f"  L (sheet width): {L}")
print(f"  V_A (Alfvén speed): {V_A}")
print(f"\nTime scales:")
print(f"  τ_R (resistive): {tau_R:.3f}")
print(f"  τ_A (Alfvén): {tau_A:.3f}")
print(f"\nFKR scaling estimate:")
print(f"  γ ~ {gamma_FKR_approx:.2f}")
print(f"  (Very rough - actual formula has geometric factors)")

# 5. Visualization
print("\n5. Creating Diagnostic Plots")
print("-"*60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Row 1: Equilibrium
axes[0,0].contourf(psi, levels=20, cmap='RdBu_r')
axes[0,0].set_title('ψ(r,θ)')
axes[0,0].set_xlabel('θ index')
axes[0,0].set_ylabel('r index')

axes[0,1].contourf(phi, levels=20, cmap='RdBu_r')
axes[0,1].set_title('φ(r,θ)')  
axes[0,1].set_xlabel('θ index')
axes[0,1].set_ylabel('r index')

axes[0,2].plot(r_grid_safe, q_profile)
axes[0,2].axhline(y=2, color='r', linestyle='--', label='q=2 (m=2 resonance)')
axes[0,2].set_xlabel('r')
axes[0,2].set_ylabel('q(r)')
axes[0,2].set_title('Safety Factor Profile')
axes[0,2].legend()
axes[0,2].grid(True)

# Row 2: Perturbation analysis
axes[1,0].plot(r_grid, dphi_dr[:, 0])
axes[1,0].axhline(y=0, color='k', linestyle='--')
axes[1,0].set_xlabel('r')
axes[1,0].set_ylabel('dφ/dr (~ J_z)')
axes[1,0].set_title('Current Profile (theta=0 slice)')
axes[1,0].grid(True)

axes[1,1].plot(r_grid, psi_m2, label='|ψ_m=2|')
axes[1,1].plot(r_grid, phi_m2, label='|φ_m=2|')
axes[1,1].set_xlabel('r')
axes[1,1].set_ylabel('Mode amplitude')
axes[1,1].set_title('m=2 Mode Structure')
axes[1,1].legend()
axes[1,1].grid(True)

# Mode spectrum
mode_spectrum = np.abs(psi_fft[nr//2, :ntheta//2])
axes[1,2].bar(range(len(mode_spectrum)), mode_spectrum)
axes[1,2].set_xlabel('m (mode number)')
axes[1,2].set_ylabel('|ψ_m|')
axes[1,2].set_title('Mode Spectrum at r=0.5')
axes[1,2].axvline(x=2, color='r', linestyle='--', label='m=2')
axes[1,2].legend()

plt.tight_layout()
plt.savefig('/tmp/tearing_ic_diagnosis.png', dpi=150)
print(f"Saved diagnostic plot to: /tmp/tearing_ic_diagnosis.png")

print("\n" + "="*60)
print("Level 1 Summary:")
print("="*60)

# Checklist
checks = []
checks.append(("Current sheet (J reversal)", dphi_dr.min() < 0 and dphi_dr.max() > 0))
checks.append(("Rational surface q≈2", q_diff.min() < 0.5))
checks.append(("m=2 mode present", m2_fraction > 0.1))

all_pass = all(c[1] for c in checks)

for check_name, passed in checks:
    status = "✅" if passed else "❌"
    print(f"{status} {check_name}")

if all_pass:
    print("\n✅ IC satisfies tearing instability requirements")
    print("   Problem likely in solver, not IC")
else:
    print("\n❌ IC may not be properly configured for tearing")
    print("   Need to fix IC before testing solver")
