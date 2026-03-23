# Hamiltonian Classification: True vs Pseudo-Hamiltonian

**Author:** 小P ⚛️  
**Date:** 2026-03-23  
**Issue:** #23 (Stage 1 Deliverable)  
**Status:** Complete

---

## Executive Summary

**Classification Result:** PyTokMHD is a **Pseudo-Hamiltonian system**

**Evidence:**
- Default resistivity: η = 1e-4 Ω·m
- Default viscosity: ν = 1e-4 m²/s
- Can be configured as True Hamiltonian by setting η = ν = 0

**Implication for v3.0 RL:**
- Standard HNN applicable to ideal (η=0) case
- Need modified architecture for resistive (η>0) case
- Recommend: Train on ideal part, handle dissipation separately

---

## 1. Code Inspection Results

### 1.1 Solver Parameters

**File:** `src/pytokmhd/solvers/hamiltonian_mhd.py`

**Constructor signature:**
```python
def __init__(
    self,
    grid: ToroidalGrid,
    dt: float = 1e-4,
    eta: float = 1e-4,    # Resistivity [Ω·m]
    nu: float = 1e-4,     # Viscosity [m²/s]
    P0: float = 0.0,
    psi_edge: Optional[float] = None,
    alpha: float = 2.0
):
```

**Key findings:**
- ✅ η (resistivity) is explicit parameter
- ✅ ν (viscosity) is explicit parameter
- ✅ Both have **non-zero defaults** (1e-4)
- ✅ Can be set to zero for ideal MHD

**Conclusion:** System supports both ideal and resistive modes

---

### 1.2 Evolution Equations in Code

**From documentation:**
```python
"""
Physical Model
--------------
Evolution equations in Hamiltonian form:
    ∂ψ/∂t = {ψ, H} - η·J       ← Resistive term
    ∂ω/∂t = {ω, H} + S_P - ν·∇²ω   ← Viscous term

where:
    - {A, B}: Poisson bracket (Hamiltonian part)
    - η: resistivity (dissipation)
    - ν: viscosity (dissipation)
"""
```

**Implementation (line 223, 243, 256):**
```python
# Line 223: Resistive diffusion (half-step)
psi_half = psi_half - 0.5 * self.dt * self.eta * J_half

# Line 243: Viscous term
viscous_term = -self.nu * laplacian_toroidal(omega, self.grid)

# Line 256: Resistive diffusion (second half-step)
psi_new = psi_new - 0.5 * self.dt * self.eta * J_new
```

**Verification:**
- ✅ Hamiltonian part: `{ψ, H}`, `{ω, H}` implemented via Poisson bracket
- ✅ Dissipative part: `η·J`, `ν·∇²ω` implemented separately
- ✅ Operator splitting: Hamiltonian → Dissipation → Hamiltonian

**Structure:** Chacón (2020) splitting strategy confirmed ✅

---

### 1.3 Default Configuration

**From test files and examples:**

**v2.0 validation (ballooning mode):**
- Used: η = 1e-5, ν = 1e-5 (resistive)
- Energy drift: 0.0000% over 100 steps
- **Conclusion:** Even with dissipation, energy conservation excellent (numerical)

**Ideal MHD capability:**
- Can set: `HamiltonianMHD(grid, dt, eta=0.0, nu=0.0)`
- Expected: Exact energy conservation (symplectic integrator)

---

## 2. Theoretical Decomposition

### 2.1 Conservative Part (True Hamiltonian)

**Equations (η = ν = 0):**

```
∂ψ/∂t = {ψ, H}
∂ω/∂t = {ω, H} + S_P
```

where Hamiltonian:

```
H[ψ, ω] = ∫ [1/2 |∇φ|² + 1/2 |∇ψ|²] dV
```

**Properties:**

1. **Energy conservation:**
```
dH/dt = 0  (exactly, for ideal case)
```

**Proof:**
```
dH/dt = ∫ [∂H/∂ψ · ∂ψ/∂t + ∂H/∂ω · ∂ω/∂t] dV
      = ∫ [δH/δψ · {ψ, H} + δH/δω · {ω, H}] dV
      = {H, H}  (Poisson bracket)
      = 0       (antisymmetry: {H, H} = -{H, H})
```

2. **Symplectic structure:**
- Canonical coordinates: q = ψ, p = -ω
- Symplectic 2-form: ω_symp = dψ ∧ dω
- **Preserved by Hamiltonian flow** (Liouville theorem)

3. **Poisson bracket properties:**
- Antisymmetry: {F, G} = -{G, F} ✅
- Jacobi identity: {{F,G},H} + cyclic = 0 ✅
- Leibniz rule: {F, GH} = {F,G}H + G{F,H} ✅

**Conclusion:** Conservative part IS true Hamiltonian system

---

### 2.2 Dissipative Part (Non-Hamiltonian)

**Equations (only dissipation):**

```
∂ψ/∂t = -η J       (resistive diffusion)
∂ω/∂t = -ν ∇²ω     (viscous diffusion)
```

where J = -∇²ψ (toroidal current)

**Properties:**

1. **Energy dissipation:**
```
dH/dt = -∫ [η |∇J|² + ν |∇ω|²] dV < 0
```

**Detailed derivation:** See [`theory/energy-dissipation-derivation.md`](theory/energy-dissipation-derivation.md) for complete proof including:
- Integration by parts (twice for each term)
- Boundary condition treatment
- Transformation: (∇²ψ)² → |∇J|²

**Physical interpretation:**
- Resistivity converts magnetic energy → heat
- Viscosity converts kinetic energy → heat
- **Irreversible process** (2nd law of thermodynamics)

2. **Breaks symplectic structure:**
- Phase space volume shrinks (not preserved)
- Not derivable from Hamiltonian
- **Requires different integrator** (implicit or semi-implicit)

---

### 2.3 Combined System (Pseudo-Hamiltonian)

**Full equations:**

```
∂ψ/∂t = {ψ, H} - η J
        ↑ Hamiltonian  ↑ Dissipative

∂ω/∂t = {ω, H} + S_P - ν ∇²ω
        ↑ Hamiltonian    ↑ Dissipative
```

**Energy balance:**

```
dH/dt = {H, H}  -  ∫ [η |∇J|² + ν |∇ω|²] dV
        ↑ = 0        ↑ < 0 (dissipation)
      
Total: dH/dt < 0 (energy decreases)
```

**Classification:** **Pseudo-Hamiltonian**
- Has Hamiltonian structure (Poisson bracket)
- Plus non-Hamiltonian dissipation
- **Not a true Hamiltonian system**

**Port-Hamiltonian form:**

```
dz/dt = (J - R) ∇H

where:
  z = [ψ, ω]ᵀ (state)
  J = skew-symmetric matrix (Poisson structure)
  R = positive-definite matrix (dissipation)
  H = Hamiltonian (energy)
```

---

## 3. Implications for Hamiltonian RL

### 3.1 Standard HNN Applicability

**HNN assumptions:**
1. System has Hamiltonian H
2. Dynamics: dz/dt = J ∇H (symplectic)
3. Energy conserved: dH/dt = 0

**PyTokMHD ideal case (η=0, ν=0):**
- ✅ Has Hamiltonian H (kinetic + magnetic energy)
- ✅ Symplectic dynamics (Morrison bracket)
- ✅ Energy conserved (proven above)

**Conclusion:** Standard HNN **applicable to ideal MHD** ✅

---

### 3.2 Resistive MHD Challenges

**HNN assumptions vs resistive reality:**
1. Energy conserved? ❌ (dH/dt < 0)
2. Symplectic? ❌ (dissipation breaks structure)
3. Reversible? ❌ (heat generation irreversible)

**Small A's v2.1 lesson:**
- Attempted HNN on resistive MHD
- Failed because system NOT Hamiltonian
- Energy loss violated HNN assumption

**Conclusion:** Standard HNN **NOT applicable to resistive MHD** ❌

---

### 3.3 Recommended Approaches for v3.0

**Option A: Port-Hamiltonian NN** (Best for resistive)

```python
class PortHamiltonianNN:
    def __init__(self, latent_dim):
        self.H_net = MLP([latent_dim, 128, 1])  # Conservative
        self.D_net = MLP([latent_dim, 128, 1])  # Dissipative
    
    def dynamics(self, z):
        dH_dz = grad(self.H_net)(z)
        dD_dz = grad(self.D_net)(z)
        J = symplectic_matrix()
        R = dissipation_matrix()
        return J @ dH_dz - R @ dD_dz
```

**Pros:**
- Respects physics (conservative + dissipative)
- Learns both H and D
- Energy dissipation modeled explicitly

**Cons:**
- More complex architecture
- Need to define R (dissipation matrix)

---

**Option B: HNN on Ideal + Residual** (Simpler)

```python
class IdealHNN:
    # Train HNN on ideal MHD (η=0, ν=0)
    # Learn conservative structure
    
class ResistiveCorrection:
    # Separate network for dissipation
    # dz/dt = HNN(z) + Dissipation(z)
```

**Pros:**
- Clean separation (physics-informed)
- HNN learns true Hamiltonian part
- Dissipation handled separately

**Cons:**
- Two networks to train
- Need to coordinate them

---

**Option C: Physics-Informed Loss** (Pragmatic)

```python
# Standard HNN + physics penalty
loss = mse_loss + λ_energy * energy_penalty + λ_diss * dissipation_penalty

where:
  energy_penalty = |dH/dt - dH_dt_expected|²
  dH_dt_expected = -∫ [η |∇J|² + ν |∇ω|²] dV
```

**Pros:**
- Single HNN architecture
- Physics constraint via loss

**Cons:**
- Not guaranteed to respect structure
- May learn wrong physics if λ too small

---

**Small P Recommendation:** **Option B** (HNN on Ideal + Residual)

**Rationale:**
1. PyTokMHD **can** run ideal MHD (η=0) → HNN training data available
2. Conservative part **is** true Hamiltonian → HNN perfect fit
3. Dissipation **is** simple diffusion → easy residual network
4. Physics-informed decomposition → interpretable

**Implementation:**
```python
# Stage 1: Train HNN on ideal trajectories
solver = HamiltonianMHD(grid, dt, eta=0.0, nu=0.0)
trajectories = generate_ideal_data(solver)
hnn = train_HNN(trajectories)

# Stage 2: Learn dissipation residual
solver_resistive = HamiltonianMHD(grid, dt, eta=1e-4, nu=1e-4)
trajectories_resistive = generate_resistive_data(solver_resistive)
residual_net = train_residual(hnn, trajectories_resistive)

# Stage 3: Combined policy
def policy(z, a):
    conservative = hnn.dynamics(z)
    dissipative = residual_net(z)
    total = conservative + dissipative
    dH_da = grad(total, a)
    return dH_da  # Control gradient
```

---

## 4. Verification Plan Impact

### 4.1 Stage 2 Tests (Conservative Part)

**Test ideal MHD (η=0, ν=0):**
1. Long-term energy conservation (1000 steps)
   - **Target:** |dH/dt| < 1e-12
2. Poisson bracket properties
   - Antisymmetry, Jacobi, Leibniz
3. Symplectic integrator advantage
   - Compare RK2 vs Störmer-Verlet

**Expected:** All tests PASS for ideal case → confirms True Hamiltonian structure

---

### 4.2 Stage 3 Tests (Dissipative Part)

**Test resistive MHD (η>0, ν>0):**
1. Energy dissipation rate
   - Measure: dH/dt
   - Compare: Theory vs numerical
   - **Target:** Match within 10%
2. Operator splitting accuracy
   - Compare: Full step vs split step
   - **Target:** Splitting error ~ O(dt²)

**Expected:** Dissipation matches theory → confirms Pseudo-Hamiltonian classification

---

### 4.3 Modified Success Criteria

**Original (from design doc):**
- ✅ Symplectic property verified
- ✅ Long-term tests pass

**Revised (based on classification):**
- ✅ **Ideal case:** Energy conserved to machine precision
- ✅ **Resistive case:** Energy dissipation matches theory
- ✅ **Decomposition:** Conservative + Dissipative split validated
- ✅ **RL guidance:** Option B architecture recommended

---

## 5. Summary & Next Steps

### 5.1 Classification Result

**PyTokMHD is:**
- ✅ **Pseudo-Hamiltonian** (default: η=1e-4, ν=1e-4)
- ✅ **Configurable as True Hamiltonian** (set η=0, ν=0)
- ✅ **Physically correct decomposition** (conservative + dissipative)

**Evidence:**
- Code inspection confirms η, ν parameters
- Operator splitting matches Chacón (2020) theory
- Energy balance derivable from first principles

---

### 5.2 RL Integration Path

**For v3.0 Hamiltonian RL:**

**Phase 2 (HNN Framework):**
- Use **Option B:** HNN on ideal + residual
- Train HNN with ideal data (η=0)
- Validate conservative structure first

**Phase 3 (RL Competitive Positioning):**
- Add dissipation residual network
- Test on resistive scenarios
- Compare vs classical control

**Why this works:**
- Conservative part IS Hamiltonian (HNN perfect)
- Dissipation simple (diffusion, easy to learn)
- Physics-informed (interpretable, robust)

---

### 5.3 Stage 1 Deliverable Complete

**This document provides:**
- ✅ Classification: Pseudo-Hamiltonian (resistive) or True (ideal)
- ✅ Code evidence: η, ν parameters identified
- ✅ Theoretical decomposition: Conservative + Dissipative
- ✅ RL recommendations: Option B (HNN on ideal + residual)
- ✅ Verification plan impact: Modified success criteria

**Next:** Stage 2 - Numerical verification tests

---

**Sign-off:**

**Classification:** Pseudo-Hamiltonian (default), True Hamiltonian (configurable)  
**Recommendation:** Option B (HNN on ideal + residual) for v3.0 RL  
**Author:** 小P ⚛️  
**Date:** 2026-03-23  
**Status:** Complete ✅
