# Issue #29: Tearing Mode Initial Condition Design

**Owner:** 小P ⚛️  
**Status:** Design phase  
**Priority:** Critical (blocks Issue #28)  
**Date:** 2026-03-24 16:00  

---

## Problem Statement

**Current Issue #28 IC produces decay, not instability:**
- Generic m=1 perturbation: ψ = ε·r(1-r)·sin(θ)
- No current gradient, no resonant surface
- Natural decay via resistive diffusion
- **Cannot test tearing mode control** ❌

**Need:** Proper tearing-unstable initial condition with observable growth in 0.1s episode.

---

## Physics Requirements ⚛️

### 1. Equilibrium with Current Profile

**Must have:**
- Non-zero current density: J₀(r) = ∇²ψ₀ ≠ 0
- Current gradient: dJ₀/dr ≠ 0 (driving instability)
- Force-balanced: ∇p = J × B

**Candidate: Linear current profile**
```
J₀(r) = J₀ × (1 - r/a)  # Decreasing outward
ψ₀(r) = -J₀/4 × (r² - r⁴/a²)  # Integrated
```

### 2. Resonant Surface

**Tearing requires:**
- Safety factor: q(r) = r·B_φ / (R₀·B_θ)
- Resonant condition: q(r_s) = m/n
- For m=1, n=1: need q(r_s) = 1

**With linear current:**
```
q(r) ≈ const / J₀(r) ~ 1/(1-r/a)
```
- q → ∞ at r=a (edge)
- Can choose J₀ such that q(r_s) = 1 at desired r_s

### 3. Tearing-Unstable Perturbation

**Classic tearing structure:**
```
δψ = ε × f(r) × sin(θ - nφ)
δφ = ε × g(r) × cos(θ - nφ)
```

**Radial structure near resonance:**
- Inner region (r < r_s): logarithmic reconnection
- Outer region (r > r_s): ideal MHD response
- Matching condition gives Δ' (tearing parameter)

**For observable growth:**
- Need Δ' > 0 (unstable)
- Growth rate: γ ~ η^(3/5) × Δ'^(4/5)
- Target: γ ~ 1-10 s⁻¹ for 10-100% growth in 0.1s

---

## Design Approach

### Option A: Harris Sheet (Simplest) ⚡

**Equilibrium:**
```python
# Current sheet at r = 0.5
J₀ = J_max × sech²((r - 0.5)/λ)
ψ₀ = -J_max × λ² × [tanh((r-0.5)/λ) + const]
```

**Advantages:**
- ✅ Well-known unstable configuration
- ✅ Analytical growth rate: γ ~ √(k·v_A·η)
- ✅ Simple to implement

**Disadvantages:**
- ⚠️ Not realistic tokamak profile
- ⚠️ May need tuning for observable growth

**小P initial choice:** Start with this for speed ⚡

### Option B: Tokamak-Like q Profile (Realistic)

**Equilibrium:**
```python
q(r) = q₀ + (q_a - q₀) × r²/a²  # Parabolic q
# Invert to get J(r)
# Add pressure gradient
```

**Advantages:**
- ✅ Realistic tokamak geometry
- ✅ Well-studied in literature
- ✅ Multiple resonant surfaces possible

**Disadvantages:**
- ⚠️ More complex numerically
- ⚠️ Requires pressure equilibrium solver
- ⚠️ 1-2 day implementation

**小P decision:** Save for v3.1+ if needed

---

## Implementation Plan

### Phase 1: Theory & Parameters (1-2h)

**Tasks:**
1. ✅ Harris sheet equilibrium derivation
2. ✅ Tearing mode dispersion relation
3. ✅ Parameter selection for γ ~ 1-10 s⁻¹
4. ✅ Document in theory/

**Deliverables:**
- `theory/harris_sheet_tearing.md` (derivation)
- Parameter table (J_max, λ, η, ε)

### Phase 2: Implementation (1-2h)

**Tasks:**
1. ✅ Implement equilibrium generator
2. ✅ Add tearing perturbation
3. ✅ Create IC function for experiments
4. ✅ Integration tests

**Files:**
- `src/pim_rl/physics/v2/tearing_ic.py` (new)
- `tests/test_tearing_ic.py` (new)

**API:**
```python
def create_tearing_ic(
    grid: ToroidalGrid,
    amplitude: float = 0.01,
    current_width: float = 0.1,
    target_growth_rate: float = 1.0  # s⁻¹
) -> Tuple[Array, Array]:
    """
    Create Harris sheet equilibrium + tearing perturbation.
    
    Returns
    -------
    psi, phi : Arrays (nr, ntheta)
        Initial conditions with tearing instability
    """
```

### Phase 3: Validation (2h)

**Tests:**
1. ✅ Equilibrium force balance: ∇p = J×B within tolerance
2. ✅ Growth rate measurement: run no-control, fit exponential
3. ✅ Growth matches theory: |γ_measured - γ_theory| < 20%
4. ✅ Observable in 0.1s: amplitude increases 10-100%

**Acceptance criteria:**
- Equilibrium error < 1%
- Growth rate matches theory ±20%
- 0.1s episode shows clear growth (>10%)

**Deliverables:**
- Validation plots (equilibrium, growth)
- Test report with measurements

---

## Technical Considerations ⚛️

### Numerical Stability

**Potential issues:**
- Current sheet steep gradient → grid resolution
- Tearing growth → may hit solver limits

**Mitigations:**
- Use tanh profile (smooth)
- Limit amplitude ε < 0.01
- Test with current grid (32×64)
- Monitor ∇·B, energy conservation

### Parameter Tuning

**Key parameters to optimize:**

| Parameter | Range | Target |
|-----------|-------|--------|
| J_max | 1-10 | Set resonant surface location |
| λ (width) | 0.05-0.2 | Smooth enough for grid |
| η | 1e-3 to 1e-2 | Growth rate ~ 1-10 s⁻¹ |
| ε | 0.001-0.01 | Small enough for linear regime |

**Tuning strategy:**
1. Fix J_max, λ (equilibrium shape)
2. Vary η to hit target growth rate
3. Verify with test run

### Compatibility with Existing Code

**No changes needed to:**
- ✅ CompleteMHDSolver (uses any ψ, φ IC)
- ✅ ElsasserMHDSolver (wrapper)
- ✅ HamiltonianMHDEnv (just passes IC)
- ✅ Observation computer (works with any state)

**Only modify:**
- Experiment scripts: replace `reset_tearing_mode()` with new IC
- Add new IC generator module

---

## Timeline & Milestones

**Total estimate:** 4-6 hours

**Breakdown:**
- Phase 1 (Theory): 1-2h
  - [ ] Derive Harris sheet
  - [ ] Calculate growth rate
  - [ ] Select parameters
  
- Phase 2 (Code): 1-2h
  - [ ] Implement IC generator
  - [ ] Add to experiment script
  - [ ] Unit tests
  
- Phase 3 (Validate): 2h
  - [ ] Force balance test
  - [ ] Growth rate measurement
  - [ ] 0.1s evolution test

**Checkpoints:**
- After Phase 1: Review parameters with YZ
- After Phase 2: Code review (小A可以review structure)
- After Phase 3: Validation report to YZ

---

## Success Criteria

**Issue #29 complete when:**

1. ✅ **Theory documented**
   - Harris sheet derivation clear
   - Growth rate formula derived
   - Parameters justified

2. ✅ **Implementation tested**
   - IC generator works
   - Equilibrium force-balanced
   - Integration with env successful

3. ✅ **Physics validated**
   - Tearing mode grows (not decays!)
   - Growth rate ~ theory (±20%)
   - Observable in 0.1s (10-100% amplitude increase)

4. ✅ **Unblocks Issue #28**
   - 小A can run baselines with new IC
   - Results show instability suppression
   - RL can be compared to classical control

---

## Risks & Mitigation

**Risk 1: Growth too slow**
- Impact: Can't see in 0.1s episode
- Mitigation: Increase η to 1e-2 if needed
- Fallback: Extend episode to 1s (but slower experiments)

**Risk 2: Numerical instability**
- Impact: Solver diverges
- Mitigation: Start with small ε, increase gradually
- Fallback: Reduce current gradient (smoother profile)

**Risk 3: Implementation complexity**
- Impact: Takes >6h
- Mitigation: Start with simplest (Harris sheet)
- Fallback: Ask YZ for time extension

---

## Dependencies

**Requires (all available):**
- ✅ CompleteMHDSolver (Phase 2)
- ✅ Grid infrastructure (Phase 1)
- ✅ Physics validation tools (Issue #26)

**Blocks:**
- Issue #28 baseline experiments (waiting on IC)

**Enables:**
- Proper tearing mode control tests
- Realistic RL training scenarios
- v3.0 completion with correct physics

---

## References

**Classical tearing mode theory:**
1. Furth, Killeen, Rosenbluth (1963) - Original tearing mode paper
2. Biskamp (1993) - Nonlinear MHD book, Ch. 5
3. White (2001) - Tokamak stability theory

**Harris sheet:**
1. Harris (1962) - Original equilibrium
2. Daughton et al. (2006) - PIC simulations
3. Numata & Loureiro (2015) - Collisionless tearing

**Implementation notes:**
- Will cite specific formulas in theory doc
- May use FKR dispersion relation for growth rate

---

**小P签字:** Ready to execute ⚛️  
**YZ approval:** Awaiting confirmation to proceed 🎯

**Next step:** Begin Phase 1 (theory derivation) immediately upon approval.
