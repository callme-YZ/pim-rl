"""
Analyze convergence results from spatial convergence study.

Author: 小P ⚛️
Date: 2026-03-24
"""

import numpy as np
import matplotlib.pyplot as plt


def load_and_analyze():
    """Load and analyze convergence results"""
    
    data = np.load('convergence_results.npz', allow_pickle=True)
    
    grids = data['grids']
    gamma = data['growth_rates']
    energies = data['energies']
    
    print("=" * 70)
    print("Convergence Results Analysis")
    print("=" * 70)
    print()
    
    # Grid information
    print("Grids tested:")
    for i, grid in enumerate(grids):
        print(f"  [{i+1}] {grid[0]}×{grid[1]}×{grid[2]}")
    print()
    
    # Growth rates
    print("Growth rates:")
    for i, (grid, g) in enumerate(zip(grids, gamma)):
        nr = grid[0]
        print(f"  Grid {nr:3d}: γ = {g:+.6f}")
    print()
    
    # Convergence metrics
    print("Convergence metrics:")
    
    # Relative changes
    for i in range(1, len(gamma)):
        rel_change = abs(gamma[i] - gamma[i-1]) / abs(gamma[i])
        print(f"  Grid {i} → {i+1}: {rel_change:.2%} relative change")
    
    # Richardson extrapolation
    if len(gamma) >= 3:
        g1, g2, g3 = gamma[-3:]
        gamma_inf = (4*g3 - g2) / 3
        print(f"\n  Richardson extrapolation: γ∞ ≈ {gamma_inf:.6f}")
        print(f"  Finest grid error: {abs(g3 - gamma_inf):.2%}")
    
    # Convergence order estimate
    if len(gamma) >= 3:
        n1, n2, n3 = grids[-3][0], grids[-2][0], grids[-1][0]
        e1 = abs(gamma[0] - gamma_inf)
        e2 = abs(gamma[1] - gamma_inf)
        e3 = abs(gamma[2] - gamma_inf)
        
        # p ≈ log(e1/e2) / log(n2/n1)
        if e1 > 1e-10 and e2 > 1e-10:
            order_12 = np.log(e1/e2) / np.log(n2/n1)
            order_23 = np.log(e2/e3) / np.log(n3/n2)
            print(f"\n  Convergence order estimate:")
            print(f"    Grid 1→2: p ≈ {order_12:.2f}")
            print(f"    Grid 2→3: p ≈ {order_23:.2f}")
            print(f"    Expected: p = 2 (2nd order)")
    
    print()
    print("=" * 70)
    print("Interpretation")
    print("=" * 70)
    
    # Sign of growth rate
    if gamma[-1] < 0:
        print("✅ Negative γ: Mode decays (resistive diffusion dominant)")
        print("   This is expected with η=0.01 and no external drive")
    else:
        print("⚠️  Positive γ: Mode grows (check physics)")
    
    # Convergence quality
    final_change = abs(gamma[-1] - gamma[-2]) / abs(gamma[-1])
    if final_change < 0.05:
        print(f"✅ Good convergence: {final_change:.2%} change on finest grid")
        print(f"   Converged value: γ ≈ {gamma[-1]:.4f}")
    else:
        print(f"⚠️  Marginal convergence: {final_change:.2%} change")
        print("   May need finer grid for publication quality")
    
    # Energy conservation
    print()
    print("Energy conservation:")
    for i, (grid, E) in enumerate(zip(grids, energies)):
        nr = grid[0]
        print(f"  Grid {nr:3d}: H_final = {E:+.6e}")
    
    print()
    print("=" * 70)


if __name__ == "__main__":
    load_and_analyze()
