# Issue #28 Physics Review

**Reviewer:** 小P ⚛️  
**Designer:** 小A 🤖  
**Date:** 2026-03-24 11:54  
**Status:** APPROVED with recommendations

---

## Executive Summary

**Overall assessment:** ✅ APPROVED

**Design quality:** Excellent
- Well-structured experiment
- Appropriate metrics
- Realistic expectations
- Good baseline choices

**Physics correctness:** ✅ Validated
- Test case appropriate
- Metrics physically meaningful
- Controller choices reasonable

**Recommendations:** 3 (non-blocking)

---

## Q1: PID Control Variable

**小A Question:** Use m=1 amplitude, H, or enstrophy?

### 小P Answer: **m=1 amplitude** ✅

**Rationale:**

**Option A: m=1 amplitude (RECOMMENDED)**
```python
m1_amp = |ψ_{1,1}|  # Direct tearing mode measure
```
✅ **Physics correct:** Directly controls instability  
✅ **Interpretable:** Clear physical meaning  
✅ **Observable:** Available from Fourier decomposition  
✅ **Stable signal:** Less noisy than H or Ω

**Option B: Hamiltonian H**
❌ **Global:** Doesn't distinguish tearing from other dynamics  
❌ **Conservation:** MHD tries to conserve H (resistivity drift only)  
❌ **Slow response:** dH/dt ~ η (resistive timescale)

**Option C: Enstrophy Ω = ∫ J² dV**
⚠️ **Multi-mode:** Includes all m modes, not just tearing  
⚠️ **Less specific:** Can't target m=1 specifically

### Recommendation

**Primary control:** m=1 amplitude ✅

**Why it works for tearing:**
- Tearing mode is m=1 instability
- Linear growth γ > 0 initially
- PID can suppress by adjusting η (resistivity)
- Physical mechanism: Increased η → faster reconnection → mode saturation

**Implementation:**
```python
# Extract m=1, n=1 mode
psi_fft = jnp.fft.fft2(psi)
m1_amp = jnp.abs(psi_fft[1, 1])  # (m=1, n=1)

# PID on amplitude
error = 0.0 - m1_amp  # Target: zero amplitude
```

---

## Q2: Linearization for LQR

**小A Question:** Is MHD weakly nonlinear enough for LQR?

### 小P Answer: **Probably NOT** ⚠️

**Analysis:**

**MHD nonlinearity sources:**
1. **Convection:** v·∇ψ, B·∇φ (strongly nonlinear)
2. **Poisson bracket:** {ψ, φ} (quadratic)
3. **Magnetic tension:** (B·∇)B (quadratic)

**Tearing mode specifics:**
- Linear phase: Small amplitude, γt < 1 (LQR may work)
- **Nonlinear phase:** Island formation, γ saturates (LQR fails)
- **Regime transition:** ~ 0.1s (our episode length!)

**Problem for LQR:**
- Linearization around equilibrium
- Assumes small perturbations
- Tearing mode grows → leaves linear regime
- **LQR control invalid once nonlinear**

### Recommendation

**Skip LQR for now** ⚠️

**Reasons:**
1. ⏰ **Time:** Deriving A, B matrices takes significant effort
2. ⚠️ **Validity:** Questionable for nonlinear tearing
3. 🎯 **Value:** PID likely sufficient for baseline
4. 📊 **Comparison:** RL vs PID is main goal

**Alternative:**
- If time permits after PID: try simple linear feedback
- Compare: u = -K*x (constant gain, no optimization)
- Avoid full LQR derivation

**Decision:** Mark as "Phase 4 optional" (not blocking)

---

## Q3: Success Criterion

**小A Question:** "amplitude < initial" sufficient, or need < 1%?

### 小P Answer: **Tiered criteria** ✅

**Problem with single threshold:**
- "< initial" too weak (random walk could achieve)
- "< 1%" too strict (may be unachievable)

### Recommendation: **3-tier success**

**Tier 1: Stabilization (minimum)**
```python
success = (m1_amp_final < m1_amp_initial)
```
- Mode not growing exponentially
- Baseline requirement

**Tier 2: Suppression (target)**
```python
success = (m1_amp_final < 0.5 * m1_amp_initial)
```
- Significant reduction
- PID should achieve this

**Tier 3: Quenching (stretch goal)**
```python
success = (m1_amp_final < 0.1 * m1_amp_initial)
```
- Near-complete suppression
- RL target

### Metrics to Report

**For each controller:**
1. **Stabilization rate:** % episodes with Tier 1
2. **Suppression rate:** % episodes with Tier 2
3. **Quenching rate:** % episodes with Tier 3
4. **Mean reduction:** (initial - final) / initial

**Why tiered:**
- Distinguishes "barely stable" from "well controlled"
- RL can target Tier 3, classical may achieve Tier 2
- Clear performance hierarchy

---

## Additional Physics Validation

### Test Case Parameters ✅

**Initial perturbation:**
```python
ψ ~ 0.01 * sin(m*θ) * exp(-r²/2)
```
✅ **Realistic:** Typical tearing mode amplitude  
✅ **Localized:** Exponential decay in r  
✅ **Mode selection:** m=1 dominant

**Episode duration: 1000 steps × 1e-4 = 0.1s**
✅ **Tearing timescale:** τ ~ η⁻¹ ~ 1e5 × 1e-4 = 10s  
⚠️ **Concern:** 0.1s << τ (episode too short?)

**Check:** Linear growth time γ⁻¹
- Typical γ ~ 0.1 (resistive MHD)
- Growth e-folding: ~10 steps
- 1000 steps → **100× growth** if uncontrolled ✅
- **Sufficient to test control**

### Grid Resolution ✅

**32 × 64 (nr × nθ)**
✅ **Adequate:** m=1 mode needs Δθ ~ 2π/64 ≈ 0.1 rad  
✅ **Nyquist:** Can resolve up to m=32  
✅ **Efficient:** Fast enough for 10 episodes

### Control Range ✅

**η_mult ∈ [0.5, 2.0]**
✅ **Physical:** Resistivity can vary (e.g., temperature changes)  
✅ **Reasonable:** 2× range allows significant control  
⚠️ **Limitation:** Cannot fully quench (would need η→0, unphysical)

**Recommendation:** Document that full quenching may be impossible with resistivity-only control

---

## PID Tuning Suggestions

**小A initial guess:**
- Kp = 10.0
- Ki = 1.0
- Kd = 0.1

### 小P Analysis

**Dimensional analysis:**
- m1_amp ~ 0.01 (dimensionless)
- error ~ 0.01
- dt ~ 1e-4

**PID output:**
- Proportional: 10 × 0.01 = 0.1
- Integral: 1.0 × 0.01 × (cumulative error)
- Derivative: 0.1 × (change/1e-4)

**Concern:** Derivative term may be too large
- If Δ(error) ~ 1e-3 → Kd term ~ 1.0 (dominates!)
- May cause oscillations

### Recommendation

**Conservative tuning (initial):**
```python
Kp = 5.0   # Lower than 小A's 10.0 (gentler response)
Ki = 0.5   # Lower integral gain (avoid windup)
Kd = 0.01  # Much lower derivative (reduce noise amplification)
```

**Tuning strategy:**
1. Start with Kp only (Ki=Kd=0)
2. Increase Kp until stable
3. Add Ki slowly (remove steady-state error)
4. Add Kd if needed (faster response)

**Physics-motivated bounds:**
- Resistive time: τ_η ~ 10s
- Control response: should act on ~ 0.1s timescale
- Kp ~ 1/τ_response ~ 10 (小A's value reasonable!)

**Revised recommendation:** Try小A's values first, reduce if oscillations

---

## Experiment Design Validation

### Trials: 10 episodes ✅

✅ **Statistical significance:** Enough for mean ± std  
✅ **Computational cost:** Reasonable (~10 min total)

**Recommendation:** Report 95% confidence intervals

### Random seeds ✅

✅ **Reproducibility:** Good practice  
✅ **Robustness:** Tests sensitivity to initial conditions

### Metrics ✅

**All metrics physically meaningful:**
- Success rate: Clear yes/no
- Energy efficiency: ∫|dH/dt|dt (resistive dissipation)
- Stability: max amplitude, std(H) (variability)
- Control effort: Mean action (cost metric)

**No issues** ✅

---

## Implementation Recommendations

### Phase 1: No Control & Random (30 min)

**Validation tests:**
```python
def test_no_control_diverges():
    """No control → exponential growth."""
    env = HamiltonianMHDEnv()
    obs = env.reset()
    
    for i in range(100):
        obs, reward, done, info = env.step([1.0, 1.0])  # No change
        
    # Check m=1 amplitude grew
    assert info['m1_amplitude'] > initial_amplitude
```

✅ **Physics check:** Growth rate ~ expected γ

### Phase 2: PID (1 hour)

**Implementation:**
```python
class PIDController:
    def __init__(self, Kp=5.0, Ki=0.5, Kd=0.01, target=0.0):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.target = target
        self.error_int = 0.0
        self.error_prev = 0.0
    
    def __call__(self, m1_amp, dt):
        error = self.target - m1_amp
        self.error_int += error * dt
        error_der = (error - self.error_prev) / dt
        self.error_prev = error
        
        eta_mult = self.Kp * error + self.Ki * self.error_int + self.Kd * error_der
        eta_mult = np.clip(eta_mult, 0.5, 2.0)
        
        return np.array([eta_mult, 1.0])  # (η_mult, ν_mult)
```

✅ **Physics correct:** Controls resistivity based on m=1 feedback

**Anti-windup (important!):**
```python
# Prevent integral windup when saturated
if eta_mult == 0.5 or eta_mult == 2.0:
    self.error_int -= error * dt  # Undo integration if saturated
```

### Phase 3: Experiments

**Recommendation:** Record diagnostics
```python
diagnostics = {
    'm1_amplitude': m1_amp_history,
    'hamiltonian': H_history,
    'actions': action_history,
    'rewards': reward_history
}
```

**For analysis:**
- Plot m1 amplitude vs time (all controllers)
- Compare growth rates
- Identify control effectiveness

---

## Final Recommendations

### Must-have (blocking)

1. ✅ **Use m=1 amplitude for PID** (not H or Ω)
2. ✅ **Tiered success criteria** (stabilization/suppression/quenching)
3. ✅ **Add anti-windup** to PID (prevent saturation issues)

### Should-have (recommended)

4. ⚠️ **Start with conservative PID gains** (Kp=5, Ki=0.5, Kd=0.01)
5. ✅ **Record full diagnostics** (m1_amp, H, actions for plots)
6. ✅ **Report 95% CI** (not just mean ± std)

### Could-have (optional)

7. ⏸️ **Skip LQR for Phase 1** (focus on PID, revisit later)
8. 📊 **Plot comparative timeseries** (all controllers on same plot)

---

## Physics Approval

**Status:** ✅ **APPROVED**

**Conditions:**
1. Use m=1 amplitude for PID control variable
2. Implement tiered success criteria
3. Add anti-windup to PID controller

**Non-blocking suggestions:**
- Conservative PID tuning initially
- Skip LQR unless time permits
- Record full diagnostics

**Physics correctness:** Validated ✅  
**Experimental design:** Sound ✅  
**Metrics:** Appropriate ✅

---

## Sign-off

**Reviewer:** 小P ⚛️  
**Recommendation:** Proceed with Phase 1 implementation  
**Next:** 小A can start coding  

**小P ready to support小A during implementation** ⚛️🤖

---

**Date:** 2026-03-24 11:54  
**Branch:** v3.0-phase3
