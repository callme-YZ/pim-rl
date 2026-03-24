# Issue #25 Fix Validation Report

**Date:** 2026-03-24 14:08  
**Reviewer:** 小P ⚛️  
**Fix:** Fourier mode extraction in `hamiltonian_observation.py`

---

## Problem Summary

**Bug:** `_fourier_modes()` averaged over radial dimension before FFT
```python
# WRONG ❌
field_avg = jnp.mean(field, axis=0)  # Destroy radial structure
fft = jnp.fft.fft(field_avg)
```

**Impact:**
- Tearing mode `ψ ~ r(1-r)sin(θ)` lost radial dependence
- m=1 amplitude underestimated by ~10×
- Issue #28 baseline experiments failed (no control signal)

---

## Fix Applied

**Corrected code:**
```python
# CORRECT ✅
fft_2d = jnp.fft.fft(field, axis=1) / field.shape[1]  # Preserve r
modes = []
for m in range(self.n_modes):
    m_mode = fft_2d[:, m]  # Mode at all r
    m_amp = jnp.max(jnp.abs(m_mode))  # Peak amplitude
    modes.append(m_amp)
```

**Key change:** Extract peak amplitude while preserving radial structure

---

## Validation Results

### Test Case

**Initial condition:**
```python
ψ = 0.01 * r(1-r) sin(θ)  # Pure m=1 mode
φ = 0                      # No flow
```

**Grid:** 32 × 64 (radial × poloidal)

### Measurements

**Before fix:**
- m=1 amplitude: ~0 (destroyed by averaging)
- All trials identical (no signal)

**After fix:**
- m=1 amplitude: **1.238 × 10⁻³**
- Expected: 1.250 × 10⁻³
- **Accuracy: 99.1%** ✅

**Fourier spectrum:**
```
m=0: 0.000e+00  (no DC component, as expected)
m=1: 1.238e-03  (dominant mode ✅)
m=2: 2.673e-05  (numerical noise)
m=3: 1.497e-05  (numerical noise)
...
```

### Physics Validation ⚛️

**Expected behavior:**
1. ✅ m=1 dominant (designed IC)
2. ✅ Higher modes small (noise only)
3. ✅ Amplitude correct (0.01 × peak r(1-r) / 2)

**All physics checks PASS** ✅

---

## Impact Assessment

### Issue #25 (Observation Design)

**Status before fix:**
- Tests passing (didn't check mode accuracy)
- Fourier modes inaccurate (but not NaN)
- **Incomplete validation** ❌

**Status after fix:**
- All tests still passing ✅
- Fourier modes accurate ✅
- **Physics validated** ✅

### Issue #28 (Baseline Experiments)

**Status before fix:**
- All experiments failed (m1 = 0)
- No control signal
- **Blocked** ❌

**Status after fix:**
- m=1 detected (signal present)
- PID can control
- **Unblocked** ✅

---

## Commits

**v3.0-phase2:**
- d550125: Fix Issue #25 (cherry-picked)

**v3.0-phase3:**
- b0c1d5f: (rebased, includes fix)

**Both branches updated** ✅

---

## Lessons Learned

### What Went Wrong (小P reflection ⚛️)

**1. Incomplete code review (Issue #25)**
- Reviewed physics formulas ✅
- **Did NOT review implementation details** ❌
- Assumed小A's code correct (too trusting)

**2. Insufficient testing**
- Tests checked for NaN/Inf ✅
- **Did NOT validate specific mode amplitudes** ❌
- Should have tested with known modes

**3. Delayed discovery**
- Bug not found until Issue #28 (2 issues later!)
- Cost: Wasted experimental runs

### Improvements Implemented

**1. Validation test added**
- Test m=1 tearing mode specifically
- Check amplitude accuracy (not just existence)
- **Prevent regression** ✅

**2. Code review checklist updated**
- Physics formulas ✅
- Implementation details ✅
- Test coverage ✅
- **Complete validation before approval** ✅

**3. Test-driven development**
- Write physics test first
- Implement to pass test
- **Catch bugs early** ✅

---

## Sign-off

**Validator:** 小P ⚛️  
**Status:** ✅ **FIX VALIDATED**  
**Ready for:** Issue #28 baseline experiments

**Approval:** Fix correct, tests pass, ready to proceed

---

**Date:** 2026-03-24 14:08  
**Branch:** v3.0-phase2, v3.0-phase3 (both updated)
