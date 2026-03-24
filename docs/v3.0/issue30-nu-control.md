# Issue #30: Implement ν (Viscosity) Control for Tearing Mode Suppression

**Created:** 2026-03-24 18:22  
**Owner:** TBD (小P ⚛️ physics + 小A 🤖 integration)  
**Priority:** v3.1  
**Depends on:** Issue #28 (baseline established)

---

## Problem Statement

**Issue #28 finding:** Current implementation controls η (resistivity), which is **counterproductive** for tearing mode suppression.

**Physics (小P ⚛️):**
- **Current:** Control η (resistivity)
  - η ↑ → Magnetic dissipation ↑ ✅
  - **BUT η ↑ → Growth rate γ ↑** (γ ~ η^0.6) ❌
  - **Net effect: May worsen instability!**

- **Should:** Control ν (viscosity)
  - ν ↑ → Fluid damping ↑
  - Suppresses velocity fluctuations
  - **Net effect: Stabilizes mode** ✅

**Evidence from Issue #28:**
- PID with η control: +23.20% growth
- no_control: +23.37% growth
- **Difference: 0.17%** (negligible, almost identical)

**Root cause:** Controlling wrong physics variable!

---

## Current Status

**Environment API:**
```python
action = [eta_mult, nu_mult]  # 2D action space
obs, reward, done, info = env.step(action)
```

**Action space exists ✅ BUT:**

**Implementation in `hamiltonian_env.py` (line 287-289):**
```python
# Extract action
eta_mult, nu_mult = action
eta = self.eta_base * float(eta_mult)
nu = self.nu_base * float(nu_mult)

# Apply action to solver (Issue #28 fix)
# Note: viscosity (nu) not yet implemented in CompleteMHDSolver
# Only resistivity (eta) control functional
self.mhd_solver.solver.set_eta(eta)
# ❌ NO set_nu() call!
```

**`CompleteMHDSolver` status:**
- ✅ Has `set_eta()` method (Issue #28)
- ❌ Missing `set_nu()` method
- ❓ Viscosity term in physics equations (unknown)

---

## Objectives

### Primary Goal
Implement functional ν (viscosity) control to enable meaningful tearing mode suppression.

### Success Criteria
1. ✅ `set_nu()` method added to CompleteMHDSolver
2. ✅ Viscosity term active in MHD equations
3. ✅ Environment applies ν action to solver
4. ✅ PID controller shows >5% growth difference vs no_control
5. ✅ Baseline experiments re-run with ν control

---

## Scope

### Phase 1: Physics Implementation (小P ⚛️)

**Task 1.1: Verify viscosity term exists**
- Check `complete_solver_v2.py` for ν term
- Look for: `-ν ∇²v` in momentum equation
- If missing: Implement viscosity physics

**Task 1.2: Add `set_nu()` method**
```python
def set_nu(self, nu: float):
    """
    Update viscosity for RL control.
    
    Parameters
    ----------
    nu : float
        New viscosity value
        
    Notes
    -----
    For RL control of tearing mode suppression.
    Higher viscosity → stronger damping → suppression.
    """
    self.nu = nu
```

**Task 1.3: Verify conservation laws**
- Energy conservation still holds
- Momentum conservation correct
- Hamiltonian structure preserved

**Estimated time:** 1-2 hours (if ν term exists), 3-4 hours (if need to implement)

---

### Phase 2: Environment Integration (小A 🤖)

**Task 2.1: Update `hamiltonian_env.py`**
```python
def step(self, action, compute_obs=True):
    # Extract action
    eta_mult, nu_mult = action
    eta = self.eta_base * float(eta_mult)
    nu = self.nu_base * float(nu_mult)
    
    # Apply action to solver
    self.mhd_solver.solver.set_eta(eta)
    self.mhd_solver.solver.set_nu(nu)  # ← ADD THIS
    
    # Continue...
```

**Task 2.2: Update PID controller**
```python
# Current (controls η):
action = [eta_mult, 1.0]

# New (controls ν):
action = [1.0, nu_mult]  # Keep η fixed, control ν

# Or both (advanced):
action = [eta_mult, nu_mult]
```

**Estimated time:** 30 min

---

### Phase 3: Validation & Experiments (小A + 小P)

**Task 3.1: Unit tests**
- Test `set_nu()` actually changes ν
- Test evolution differs with different ν
- Test conservation laws hold

**Task 3.2: Quick validation**
- Single trial: no_control vs PID with ν
- Expect: PID shows reduced growth
- If not: Debug physics

**Task 3.3: Re-run baseline experiments**
- Same setup as Issue #28
- 3 controllers × 5 trials
- Compare with Issue #28 results

**Expected results:**
| Controller | Issue #28 (η) | Issue #30 (ν) | Improvement |
|------------|---------------|---------------|-------------|
| no_control | +23.37%       | +23.37%       | 0% (baseline) |
| PID        | +23.20%       | **+15-18%?**  | **~5-8%** ✅ |

**Estimated time:** 1 hour

---

## Technical Details

### Viscosity in MHD Equations

**Momentum equation (should have):**
```
∂v/∂t + v·∇v = -∇p + j×B - ν∇²v
                              ↑ This term
```

**Effect of ν:**
- Damps velocity fluctuations
- Reduces Reynolds number Re ~ v L / ν
- Stabilizes fluid instabilities

**Tearing mode suppression mechanism:**
- Tearing mode grows via reconnection flows
- ν damps these flows
- Growth rate reduced: γ(ν) < γ(ν=0)

### Current vs Desired Control

**Issue #28 (η control):**
```
Action: η_mult ∈ [0.5, 1.5]
η_actual = 0.05 × η_mult ∈ [0.025, 0.075]

Problem: γ ~ η^0.6
η × 2 → γ × 1.5 (growth INCREASES!)
```

**Issue #30 (ν control):**
```
Action: ν_mult ∈ [0.5, 1.5]
ν_actual = 1e-4 × ν_mult ∈ [5e-5, 1.5e-4]

Desired: γ decreases with ν
ν × 2 → γ × 0.7 (growth DECREASES)
```

---

## Implementation Plan

### Timeline (v3.1)

**Week 1:**
- Day 1: 小P verifies viscosity term (2 hours)
- Day 2: 小P implements `set_nu()` + tests (3 hours)
- Day 3: 小A integrates + quick test (1 hour)

**Week 2:**
- Day 1: Re-run baseline experiments (1 hour)
- Day 2: Analysis + report (2 hours)
- Day 3: Issue #30 closure

**Total estimate:** 9 hours work, 2 weeks calendar time

---

## Risks & Mitigation

### Risk 1: Viscosity term not in solver
**Probability:** Medium  
**Impact:** High (need to implement physics)  
**Mitigation:** 小P reviews code first; if missing, escalate to v3.2

### Risk 2: ν control still ineffective
**Probability:** Low  
**Impact:** Medium  
**Mitigation:** Pre-test with parameter scan before full experiments

### Risk 3: Conservation laws break
**Probability:** Low  
**Impact:** High  
**Mitigation:** Comprehensive validation suite before experiments

---

## Success Metrics

**Minimum viable:**
1. ✅ `set_nu()` functional
2. ✅ PID shows >2% growth difference vs no_control

**Desired:**
1. ✅ PID shows >5% growth difference
2. ✅ Some trials reach Tier 1 (stabilization)

**Stretch:**
1. ✅ PID shows >10% suppression
2. ✅ >20% trials reach Tier 1

---

## Related Issues

- **Issue #28:** Baseline experiments (η control) ✅ Closed
- **Issue #29:** Harris sheet IC design ✅ Closed
- **Issue #26:** Sparse observation mode ✅ Closed

**Blocked by:** None  
**Blocks:** Future RL training (needs meaningful control to learn)

---

## Notes

**Why defer to v3.1?**
- v3.0 Phase 3 nearly complete
- Issue #28 objective met (baseline established)
- ν implementation non-trivial (~9 hours)
- Better to do properly in v3.1 than rush now

**Alternative approaches (future):**
- Control external current (J_ext)
- Control RMP coils
- Control heating/fueling

**Physics references:**
- Furth-Killeen-Rosenbluth (1963) - Tearing mode theory
- Biskamp (2000) - Viscosity effects on instabilities
- Wesson (2011) - MHD control methods

---

**Created by:** 小A 🤖  
**Date:** 2026-03-24 18:22  
**Status:** 📋 OPEN (v3.1)
