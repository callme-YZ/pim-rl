# Issue #26 Design Decision: Conversion Strategy

**Date:** 2026-03-24  
**Author:** 小P ⚛️ (physics), 小A 🤖 (RL integration)  
**Status:** Approved by YZ

---

## Problem

Issue #26 requires integrating:
- **CompleteMHDSolver:** Uses (z⁺, z⁻) Elsasser variables
- **HamiltonianObservation (Issue #25):** Expects (ψ, φ) MHD primitives

**Question:** How to bridge the two representations?

---

## Options Considered

### Option A: Avoid Poisson Inversion

**Idea:** Compute observation directly from (v, B) = derived from (z⁺, z⁻)

**Pros:**
- No Poisson solver needed
- Simpler, no BC issues
- Faster

**Cons:**
- ❌ Helicity K = ∫ ψ·B dV **requires ψ itself** (not just B = ∇²ψ)
- ❌ Issue #24 HamiltonianGradientComputer API expects (ψ, φ)
- ❌ Would break validated Issue #25 observation
- ❌ Requires reimplementing observation formulas

**Verdict:** ❌ **Rejected** (breaks existing validated API)

---

### Option B: Conversion via Poisson Solver ✅

**Idea:** 
- Evolution: (z⁺, z⁻) → (z⁺, z⁻) (no Poisson)
- Observation: (z⁺, z⁻) → (ψ, φ) → compute (uses Poisson)

**Pros:**
- ✅ Keeps Issue #25 observation API unchanged
- ✅ Reuses validated Issue #24 Hamiltonian computation
- ✅ Evolution still structure-preserving (Morrison bracket)
- ✅ Poisson solver only called 1×/RL step (acceptable cost)

**Cons:**
- Need Poisson solver (but already validated)
- Need to handle boundary conditions correctly

**Verdict:** ✅ **APPROVED** (小P + 小A + YZ)

---

## Selected Solution: Option B

### Architecture

```
┌─────────────────────────────────────────────┐
│ RL Environment (HamiltonianMHDEnv)         │
│                                             │
│  Initialize: (ψ, φ)                        │
│      ↓                                      │
│  Forward: (ψ, φ) → (v, B) → (z⁺, z⁻)      │
│      (via laplacian, 1× only)              │
│                                             │
│  ┌────────────────────────────────┐        │
│  │ Physics Evolution Loop         │        │
│  │ (many substeps per RL step)    │        │
│  │                                │        │
│  │ (z⁺, z⁻) → CompleteMHDSolver  │        │
│  │             ↓                  │        │
│  │         (z⁺, z⁻)_new           │        │
│  │                                │        │
│  │ NO POISSON SOLVER NEEDED ✅    │        │
│  └────────────────────────────────┘        │
│      ↓                                      │
│  Inverse: (z⁺, z⁻) → (v, B) → (ψ, φ)      │
│      (via Poisson solver, 1×/RL step)      │
│      ↓                                      │
│  Observation: compute from (ψ, φ)          │
│      (Issue #25 API, validated)            │
└─────────────────────────────────────────────┘
```

### Key Insight

**Evolution does NOT need (ψ, φ):**
- Morrison bracket formulation computes {F, G} directly
- Elsasser variables (z⁺, z⁻) are self-contained
- **Only observation needs (ψ, φ)**

**Cost analysis:**
- Evolution: N substeps × 0 Poisson solves = 0
- Observation: 1 RL step × 2 Poisson solves = ~1 second
- **Acceptable for RL** (10-50ms typical env step budget includes physics)

---

## Implementation Details

### Wrapper Class

```python
class ElsasserMHDSolver:
    """
    Wrapper bridging (z⁺, z⁻) evolution and (ψ, φ) observation.
    
    Internal: CompleteMHDSolver (Elsasser formulation)
    External: (ψ, φ) interface for Issue #25 observation
    """
    
    def __init__(self, solver, grid):
        self.solver = solver  # CompleteMHDSolver
        self.grid = grid      # ToroidalGrid for Poisson solver
        
        # State storage
        self._state_els = None  # ElsasserState (z⁺, z⁻, P)
        self._psi_prev = None   # For BC
        self._phi_prev = None   # For BC
    
    def initialize(self, psi, phi):
        """
        Initialize from (ψ, φ).
        
        Forward conversion: (ψ, φ) → (z⁺, z⁻)
        Uses laplacian (not Poisson).
        """
        v = laplacian_toroidal(phi, self.grid)
        B = laplacian_toroidal(psi, self.grid)
        
        z_plus = v + B
        z_minus = v - B
        
        self._state_els = ElsasserState(z_plus, z_minus, 0)
        
        # Store for BC
        self._psi_prev = psi
        self._phi_prev = phi
    
    def step(self, dt):
        """
        Evolve physics (no Poisson solver).
        
        (z⁺, z⁻) → CompleteMHDSolver → (z⁺, z⁻)_new
        """
        self._state_els = self.solver.step(self._state_els, dt)
    
    def get_mhd_state(self):
        """
        Convert to (ψ, φ) for observation.
        
        Inverse conversion: (z⁺, z⁻) → (ψ, φ)
        Uses Poisson solver with BC from previous state.
        
        Returns
        -------
        psi, phi : jnp.ndarray
            MHD primitives for observation
        """
        # Extract v, B
        v = (self._state_els.z_plus + self._state_els.z_minus) / 2
        B = (self._state_els.z_plus - self._state_els.z_minus) / 2
        
        # Boundary conditions from previous (ψ, φ)
        if self._psi_prev is not None:
            psi_bnd = self._psi_prev[-1, :]
            phi_bnd = self._phi_prev[-1, :]
        else:
            # First call: zero BC
            psi_bnd = np.zeros(self.grid.ntheta)
            phi_bnd = np.zeros(self.grid.ntheta)
        
        # Poisson inversion
        v_np = np.array(v)
        B_np = np.array(B)
        
        phi_np, info_phi = solve_poisson_toroidal(v_np, self.grid, phi_bnd)
        psi_np, info_psi = solve_poisson_toroidal(B_np, self.grid, psi_bnd)
        
        if info_phi != 0 or info_psi != 0:
            print(f"Warning: Poisson solve convergence issue")
        
        # Convert to JAX
        phi = jnp.array(phi_np)
        psi = jnp.array(psi_np)
        
        # Store for next BC
        self._psi_prev = psi
        self._phi_prev = phi
        
        return psi, phi
```

---

## Boundary Condition Strategy

**Problem:** Poisson solver needs BC: φ(r=a, θ) = ?

**Solution:** Use previous (ψ, φ) values at boundary

**Why it works:**
1. Evolution is smooth → boundary changes slowly
2. BC from t → good approximation for BC at t+dt
3. Poisson solver enforces BC → solution consistent

**Validation:**
- Round-trip test: (ψ, φ) → (z⁺, z⁻) → (ψ, φ)
- Expected error: <5% (vs 100% with zero BC)

---

## Testing Plan

### Phase 1 Tests (小P)

1. **Round-trip with BC storage:**
   - Initialize: (ψ, φ)_0
   - Convert: → (z⁺, z⁻)
   - Invert: → (ψ, φ)_recovered
   - Check: ||(ψ, φ)_recovered - (ψ, φ)_0|| < 5%

2. **Evolution stability:**
   - Initialize from realistic (ψ, φ)
   - Evolve 100 steps
   - Check: no NaN/Inf, energy bounded

3. **Observation consistency:**
   - Get (ψ, φ) from wrapper
   - Compute observation (Issue #25 API)
   - Verify: all 23D components finite

### Phase 2 Tests (小A)

1. **HamiltonianMHDEnv integration:**
   - Replace dummy solver with wrapper
   - Test reset(), step()
   - Verify: observation shape (23,)

2. **PPO smoke test:**
   - 10 episodes training
   - Check: policy improves, no crashes

---

## Performance Analysis

**Evolution (N substeps):**
- Laplacian: N × ~0.5ms = 0.5N ms
- Morrison bracket: N × ~1ms = N ms
- **No Poisson solver** ✅
- **Total: ~1.5N ms**

**Observation (1× per RL step):**
- Poisson solve ×2: ~0.4s (GMRES iterations)
- Observation compute: ~1.5ms (Issue #25)
- **Total: ~0.4s**

**RL step budget:**
- Typical: 10-50ms for simple envs
- Physics-heavy: 0.5-5s acceptable
- **Our cost: ~0.4s + 1.5N ms** ✅ Acceptable

---

## Risks and Mitigations

**Risk 1: BC mismatch accumulates**
- Symptom: Observation drifts over time
- Mitigation: Validate round-trip every K steps
- Fallback: Re-initialize from checkpoints

**Risk 2: Poisson solver slow**
- Symptom: Observation bottleneck
- Mitigation: Already validated (10/10 tests, ~0.2s/solve)
- Optimization: Cache Poisson matrix (future)

**Risk 3: API incompatibility**
- Symptom: Issue #25 observation breaks
- Mitigation: Keep exact API (ψ, φ) unchanged
- Validation: Rerun Issue #25 tests

---

## Alternative Rejected

**Why not recompute observation from (v, B)?**

**Helicity blocker:**
```python
# Definition
K = ∫ A·B dV  (magnetic helicity)

# In 2D toroidal
K ≈ ∫ ψ·B dV  where B = ∇²ψ

# Problem
# Given: B (from z⁺, z⁻)
# Need: ψ such that ∇²ψ = B
# → This IS the Poisson problem!
```

**Verdict:** Can't avoid Poisson solver for helicity ✅

---

## Decision Rationale

**Why Option B wins:**

1. **Preserves validated work:**
   - Issue #24: HamiltonianGradientComputer ✅
   - Issue #25: 23D observation ✅

2. **Structure-preserving evolution:**
   - Morrison bracket (Elsasser) ✅
   - No (ψ, φ) round-trip in physics loop ✅

3. **Acceptable cost:**
   - Poisson solver: 0.4s/RL step
   - vs RL training: hours
   - Negligible overhead ✅

4. **Robust:**
   - BC from previous state (smooth evolution)
   - Validated Poisson solver (10/10 tests)
   - Clear failure modes (convergence, BC drift)

---

## Approval

**小P (physics):** ✅ Approved  
**小A (RL):** ✅ Approved  
**YZ (决策):** ✅ Approved (2026-03-24 11:14)

**Implementation:** Proceed with Phase 1

---

**Next:** 小P implements wrapper, 小A integrates to HamiltonianMHDEnv
