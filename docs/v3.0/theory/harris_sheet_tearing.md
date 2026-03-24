# Harris Sheet Tearing Mode - Theory & Derivation

**Author:** 小P ⚛️  
**Date:** 2026-03-24  
**Purpose:** Design unstable IC for Issue #29  

---

## Physical Setup

### Harris Sheet Equilibrium

**Configuration:**
- 1D current sheet centered at r = r₀
- Anti-parallel magnetic field across sheet
- Pressure balance maintains equilibrium

**Magnetic field (2D MHD, ignorin toroidal effects):**
```
B_r = 0
B_θ = B₀ × tanh((r - r₀)/λ)
```

Where:
- B₀ = characteristic field strength
- λ = sheet half-width
- r₀ = sheet center (choose r₀ = 0.5 for mid-radius)

**Current density:**
```
J_z = (1/μ₀) × ∇×B
    = (B₀/μ₀λ) × sech²((r-r₀)/λ)
```

Peaked at r = r₀ with width ~ λ.

---

## Reduced MHD Formulation

### Flux Function ψ

**Define poloidal flux:**
```
B_r = -(1/r) × ∂ψ/∂θ
B_θ = ∂ψ/∂r
```

**For Harris sheet:**
```
B_θ = B₀ tanh((r-r₀)/λ)

→ ψ(r) = ∫ B_θ dr
        = B₀λ × ln(cosh((r-r₀)/λ)) + const
```

**Current:**
```
J_z = -∇²ψ = -(B₀/λ) × sech²((r-r₀)/λ)
```

(Negative because current flows in -z direction)

---

## Tearing Mode Instability

### Linear Perturbation

**Add small m=1, n=1 mode:**
```
ψ = ψ₀(r) + ε × ψ₁(r) × sin(θ)
φ = ε × φ₁(r) × cos(θ)
```

Where:
- ψ₀(r) = Harris sheet equilibrium
- ε << 1 (small amplitude)
- Assume exponential growth: ~ exp(γt)

### Tearing Mode Equation

**Linearized reduced MHD (from Furth-Killeen-Rosenbluth 1963):**

**Outer region (r ≠ r_s):**
```
∇²ψ₁ - (m/r)² ψ₁ = 0  (ideal MHD)
```

**Inner region (r ≈ r_s, resonant surface):**
```
γ × ∇²ψ₁ = η × ∇⁴ψ₁ + [resistive + reconnection terms]
```

**Resonant surface:** Where q(r_s) = m/n = 1
- For Harris sheet, this can be at r_s = r₀ (current peak)

---

## Growth Rate Estimate

### Classic Tearing Formula (Furth-Killeen-Rosenbluth)

**For Δ' > 0 (tearing-unstable):**
```
γ = c × η^(3/5) × τ_A^(-1) × (Δ')^(4/5)
```

Where:
- η = resistivity
- τ_A = Alfvén time = a/v_A
- Δ' = tearing stability parameter (jump in ∇ψ' at r_s)
- c ~ O(1) numerical constant

**For Harris sheet:**
- Δ' ~ 1/λ (steeper sheet → larger Δ')
- τ_A ~ 1 (normalized units)

**Simplified estimate:**
```
γ ≈ η^(3/5) / λ^(4/5)
```

### Parameter Selection for Observable Growth

**Target: γ ~ 1-10 s⁻¹ for 10-100% growth in 0.1s**

**Choose:**
- λ = 0.1 (10% of domain)
- η = 1e-2 (enhanced resistivity)

**Predicted growth rate:**
```
γ ≈ (1e-2)^0.6 / (0.1)^0.8
  ≈ 0.04 / 0.16
  ≈ 0.25 s⁻¹
```

**Growth in 0.1s:**
```
A(t=0.1) / A(0) = exp(γ × 0.1)
                ≈ exp(0.025)
                ≈ 1.025 (2.5% growth)
```

**TOO SLOW!** Need higher growth rate.

### Revised Parameters

**Try larger η:**
- η = 0.1 (very large, unphysical but for testing)

```
γ ≈ (0.1)^0.6 / (0.1)^0.8
  ≈ 0.25 / 0.16
  ≈ 1.6 s⁻¹
```

**Growth in 0.1s:**
```
exp(1.6 × 0.1) ≈ exp(0.16) ≈ 1.17 (17% growth) ✓
```

**OR narrower sheet:**
- λ = 0.05, η = 1e-2

```
γ ≈ 0.04 / (0.05)^0.8
  ≈ 0.04 / 0.09
  ≈ 0.44 s⁻¹
```

```
exp(0.44 × 0.1) ≈ 1.045 (4.5% growth)
```

Still marginal.

**BEST OPTION:** Use η = 0.05, λ = 0.1
```
γ ≈ (0.05)^0.6 / 0.16
  ≈ 0.13 / 0.16
  ≈ 0.8 s⁻¹

exp(0.8 × 0.1) ≈ 1.08 (8% growth) ✓
```

---

## Recommended Parameters

**Equilibrium:**
```python
r₀ = 0.5      # Sheet center (mid-radius)
λ = 0.1       # Sheet width (10% of domain)
B₀ = 1.0      # Field strength (normalized)
```

**Physics:**
```python
η = 0.05      # Resistivity (enhanced for observable growth)
ε = 0.01      # Perturbation amplitude (small, linear regime)
```

**Expected behavior:**
```
Growth rate: γ ≈ 0.8 s⁻¹
0.1s growth: ~8%
1.0s growth: ~120% (clear instability!)
```

**For faster growth (if needed):**
- Increase η → 0.1 gives γ ~ 1.6 s⁻¹ (17% in 0.1s)
- Decrease λ → 0.05 gives γ ~ 1.8 s⁻¹ (20% in 0.1s)

---

## Implementation Details

### Equilibrium Functions

**Poloidal flux:**
```python
def psi_harris(r, r0=0.5, lam=0.1, B0=1.0):
    """Harris sheet equilibrium."""
    x = (r - r0) / lam
    return B0 * lam * np.log(np.cosh(x))
```

**Current density:**
```python
def J_harris(r, r0=0.5, lam=0.1, B0=1.0):
    """Current density (J_z)."""
    x = (r - r0) / lam
    return -(B0 / lam) * (1 / np.cosh(x))**2
```

**Safety factor (approximate):**
```python
def q_harris(r, r0=0.5, lam=0.1, B0=1.0, R0=3.0):
    """Safety factor q(r) = r B_phi / (R0 B_theta)."""
    B_theta = B0 * np.tanh((r - r0) / lam)
    # Assume B_phi ~ const for simplicity
    B_phi = 1.0
    return r * B_phi / (R0 * B_theta + 1e-10)  # Avoid division by zero
```

### Tearing Perturbation

**Form (Furth-Killeen-Rosenbluth eigenfunction):**
```python
def psi_tearing(r, theta, r0=0.5, lam=0.1, eps=0.01):
    """Tearing mode perturbation (m=1, n=1)."""
    # Radial structure (Gaussian centered at r0)
    radial = np.exp(-((r - r0) / (2*lam))**2)
    
    # Poloidal structure
    poloidal = np.sin(theta)
    
    return eps * radial * poloidal

def phi_tearing(r, theta, r0=0.5, lam=0.1, eps=0.01):
    """Stream function perturbation."""
    # Similar structure but cos for vorticity
    radial = np.exp(-((r - r0) / (2*lam))**2)
    poloidal = np.cos(theta)
    
    return eps * radial * poloidal
```

### Combined Initial Condition

```python
def create_tearing_ic(grid, r0=0.5, lam=0.1, eta=0.05, eps=0.01):
    """
    Create Harris sheet + tearing perturbation.
    
    Returns
    -------
    psi, phi : (nr, ntheta) arrays
    """
    r = grid.r[:, None]
    theta = grid.theta[None, :]
    
    # Equilibrium
    psi_eq = psi_harris(r, r0, lam)
    
    # Perturbation
    psi_pert = psi_tearing(r, theta, r0, lam, eps)
    phi_pert = phi_tearing(r, theta, r0, lam, eps)
    
    # Total
    psi = psi_eq + psi_pert
    phi = phi_pert  # No equilibrium flow
    
    return psi, phi
```

---

## Validation Tests

### Test 1: Equilibrium Force Balance

**Check:** ∇p = J × B

For Harris sheet (pressure-free):
```
∇p = 0 (assumed)
J × B = J_z × B_θ  (should be balanced by tension)
```

**Acceptance:** Error < 1% of peak force

### Test 2: Growth Rate Measurement

**Procedure:**
1. Initialize with IC
2. Run no-control for 1000 steps (0.1s)
3. Track m=1 amplitude: A(t)
4. Fit: A(t) = A₀ exp(γt)
5. Compare γ_measured vs γ_theory

**Acceptance:** |γ_measured - γ_theory| < 20%

### Test 3: Observable Growth

**Check:** A(0.1s) / A(0) > 1.05 (>5% growth)

**If fails:**
- Increase η (try 0.1)
- Decrease λ (try 0.05)
- Check for numerical damping

---

## References

**Tearing mode theory:**
1. Furth, Killeen, Rosenbluth (1963) "Finite-Resistivity Instabilities of a Sheet Pinch", *Phys. Fluids* **6**, 459
   - Original tearing mode analysis
   - Growth rate formula: γ ~ η^(3/5)

2. Biskamp (1993) *Nonlinear Magnetohydrodynamics*, Cambridge
   - Chapter 5: Tearing modes
   - Detailed derivation of Δ'

**Harris sheet:**
3. Harris (1962) "On a Plasma Sheath Separating Regions of Oppositely Directed Magnetic Field", *Nuovo Cimento* **23**, 115
   - Original equilibrium
   
4. Loureiro & Uzdensky (2015) "Magnetic Reconnection: Recent Theoretical and Observational Results", *Plasma Phys. Control. Fusion* **58**, 014021
   - Modern review including fast reconnection

**Numerical implementation:**
5. Numata, Howes, Dorland, Loureiro (2011) "Gyrokinetic simulations of the tearing instability", *Phys. Plasmas* **18**, 112106
   - Numerical methods for tearing modes

---

## Summary

**Parameters chosen:**
- r₀ = 0.5, λ = 0.1, B₀ = 1.0
- η = 0.05, ε = 0.01

**Expected:**
- Growth rate: γ ≈ 0.8 s⁻¹
- 0.1s evolution: 8% amplitude increase
- Observable, controllable instability ✓

**Next steps:**
- Implement in code (Phase 2)
- Validate growth (Phase 3)
- Integrate with Issue #28 experiments

---

**小P签字:** Theory complete ⚛️  
**Status:** Ready for Phase 2 (implementation)
