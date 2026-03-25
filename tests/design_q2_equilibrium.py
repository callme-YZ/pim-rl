#!/usr/bin/env python3
"""
Design equilibrium with q≈2 rational surface for tearing mode.

Goal: Find psi(r) such that q(r) = r * (dpsi/dr) / psi has q≈2 somewhere.

Author: 小P ⚛️
Date: 2026-03-25
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("="*60)
print("Design q=2 Equilibrium for Tearing Mode")
print("="*60)

# Grid
nr = 64
r = np.linspace(0.01, 1.0, nr)  # Avoid r=0

# Approach: Use modified Harris sheet that has q≈2
# Harris sheet + linear ramp to ensure q=2 somewhere

# Combined equilibrium: Harris sheet (current reversal) + correction for q
def psi_combined(r, r0=0.5, lam=0.1, B0=1.0, q_correction=1.2):
    """
    Modified Harris sheet with q≈2 at resonance.
    
    psi = psi_harris + linear correction
    """
    # Harris component (current sheet)
    x = (r - r0) / lam
    psi_harris = B0 * lam * np.log(np.cosh(x))
    
    # Linear correction to boost q
    psi_linear = q_correction * r**2
    
    return psi_harris + psi_linear

psi = psi_combined(r)

# Compute dpsi/dr and q
dpsi_dr = np.gradient(psi, r)
q_vals = r * dpsi_dr / psi

q_target = 2.0
i_res = np.argmin(np.abs(q_vals - q_target))
r_res = r[i_res]
q_res = q_vals[i_res]

print(f"\nCombined Equilibrium:")
print(f"  q(r=0.5) ≈ {q_vals[nr//2]:.3f}")
print(f"  Resonant surface q={q_res:.3f} at r={r_res:.3f}")

print(f"\nψ(r) profile:")
print(f"  ψ(r=0.01) = {psi[0]:.6f}")
print(f"  ψ(r=0.5) = {psi[nr//2]:.6f}")
print(f"  ψ(r=1.0) = {psi[-1]:.6f}")

# Already have q_vals from above

# Current density (proportional to Laplacian of psi)
# J_z ~ -∇²ψ ≈ -(1/r) d/dr(r dpsi/dr)
d2psi_dr2 = np.gradient(dpsi_dr, r)
J_z = -(dpsi_dr / r + d2psi_dr2)

print(f"\nCurrent density:")
print(f"  J_z range: [{J_z.min():.6f}, {J_z.max():.6f}]")
print(f"  J_z reversal? {J_z.min() < 0 and J_z.max() > 0}")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# q profile
axes[0,0].plot(r, q_vals, 'b-', linewidth=2, label='q(r)')
axes[0,0].axhline(y=2, color='k', linestyle=':', label='q=2')
axes[0,0].axvline(x=r_res, color='k', linestyle=':', alpha=0.5)
axes[0,0].scatter([r_res], [q_res], color='red', s=100, zorder=10, label=f'Resonance at r={r_res:.3f}')
axes[0,0].set_xlabel('r')
axes[0,0].set_ylabel('q')
axes[0,0].set_title('Safety Factor Profile')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# psi profile
axes[0,1].plot(r, psi, 'b-', linewidth=2)
axes[0,1].axvline(x=r_res, color='r', linestyle='--', alpha=0.5, label=f'q=2 at r={r_res:.3f}')
axes[0,1].set_xlabel('r')
axes[0,1].set_ylabel('ψ(r)')
axes[0,1].set_title('Poloidal Flux')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# dpsi/dr (Btheta)
axes[1,0].plot(r, dpsi_dr, 'b-', linewidth=2)
axes[1,0].axvline(x=r_res, color='r', linestyle='--', alpha=0.5)
axes[1,0].set_xlabel('r')
axes[1,0].set_ylabel('dψ/dr ~ B_θ')
axes[1,0].set_title('Poloidal Field')
axes[1,0].grid(True, alpha=0.3)

# Current density
axes[1,1].plot(r, J_z, 'b-', linewidth=2)
axes[1,1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
axes[1,1].axvline(x=r_res, color='r', linestyle='--', alpha=0.5, label=f'q=2 at r={r_res:.3f}')
axes[1,1].set_xlabel('r')
axes[1,1].set_ylabel('J_z')
axes[1,1].set_title('Current Density')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/q2_equilibrium_design.png', dpi=150)
print(f"\n✅ Saved design plot to: /tmp/q2_equilibrium_design.png")

# Export parameters for implementation
print("\n" + "="*60)
print("Implementation Parameters (Modified Harris Sheet):")
print("="*60)
print("""
def psi_q2_equilibrium(r, r0=0.5, lam=0.1, B0=1.0, q_correction=1.2):
    '''
    Modified Harris sheet with q≈2 rational surface.
    
    Combines:
    - Harris sheet (current reversal)
    - Parabolic correction (boosts q)
    
    Result: q≈2 at r≈0.5-0.6, with current sheet
    '''
    x = (r - r0) / lam
    psi_harris = B0 * lam * np.log(np.cosh(x))
    psi_linear = q_correction * r**2
    return psi_harris + psi_linear
""")
print(f"Resonant surface: r≈{r_res:.3f}, q≈{q_res:.3f}")
print("Copy this function to tearing_ic.py ✅")
