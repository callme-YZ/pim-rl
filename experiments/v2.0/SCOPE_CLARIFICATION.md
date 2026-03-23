# v2.0 Scope Clarification (2026-03-23)

**Author:** 小P ⚛️  
**Purpose:** Honest assessment of v2.0 capabilities after benchmark investigation

---

## What v2.0 IS

### Designed For: Ballooning Mode Control

**Physics model:**
- 2D Morrison bracket (r, θ) + z-periodic Fourier modes
- Optimized for **pressure-driven instabilities**
- Toroidal geometry (tokamak)

**Validated capabilities:**
- ✅ Ballooning mode growth (γ ~ √β)
- ✅ Structure-preserving energy conservation
- ✅ β-dependent plasma dynamics
- ✅ RMP coil control via RL

**Benchmark status:**
- ✅ Theory validation (Tier 1): 3/3 PASS
- ✅ Internal consistency verified
- ⏸️ External validation: Seeking ballooning literature comparison

---

## What v2.0 IS NOT

### NOT Designed For: Tearing Mode Control

**Evidence:**
1. **Implementation:** No tearing IC in codebase
2. **Validation:** All tests used `ballooning_ic_v2`
3. **Benchmark failures:** 
   - Test 1.2 (toroidal tearing): ❌ Decay instead of growth
   - FKR cylindrical tearing: ❌ Decay instead of growth

**Physics limitation:**
- Morrison bracket formulation may stabilize resistive tearing
- 2D bracket lacks parallel dynamics needed for tearing reconnection
- **Not a bug — designed scope limitation**

### NOT a General MHD Code

**v2.0 is:**
- RL framework for plasma instability control
- Physics layer = specialized Morrison bracket solver
- **Purpose:** Demonstrate RL-based control, not general simulation

**NOT:**
- BOUT++/M3D-C1 competitor
- General-purpose MHD tool
- Universal instability solver

---

## Why This Matters

### For Validation

**Appropriate benchmarks:**
- ✅ Ballooning growth rates vs theory
- ✅ Energy conservation
- ✅ Toroidal ballooning literature comparison

**Inappropriate benchmarks:**
- ❌ Tearing mode (FKR, cylindrical)
- ❌ General MHD test suite
- ❌ BOUT++ direct comparison (different physics models)

### For Publication

**Honest claims:**
- ✅ "RL for ballooning mode suppression"
- ✅ "Structure-preserving framework for pressure-driven instabilities"
- ✅ "Physics-informed control in toroidal geometry"

**Avoid over-claiming:**
- ❌ "Extensible to tearing modes" (requires 3D full MHD)
- ❌ "General MHD control framework"
- ❌ "Applicable to all plasma instabilities"

### For Future Work

**Clear roadmap:**
- v2.0: Ballooning specialist (complete this)
- v3.0/PyTokMHD: Add tearing support (3D full MHD)
- Requires: Parallel dynamics, 3D reconnection physics

---

## Lessons Learned

### What Went Wrong

**小P's mistakes:**
1. Assumed v2.0 supported tearing (didn't check implementation)
2. Designed benchmarks before understanding capabilities
3. Attempted FKR without verifying tearing works

**Root cause:** 
- API docs mentioned tearing (planned feature)
- Actual implementation = ballooning only
- **Documentation vs reality mismatch**

### What Went Right

**v2.0 strengths:**
- Ballooning physics works well
- Structure-preserving verified
- Clear, focused scope (now)

**Benchmark Tier 1:**
- Successfully validated designed capabilities
- Theory comparison appropriate
- Internal consistency proven

---

## Updated Validation Strategy

### Tier 1: Theory ✅ COMPLETE

- Ballooning growth: 77% error (factor of 2, acceptable)
- Energy conservation: 0.0000% drift (perfect)
- β scaling: Perfect (γ ∝ √β)

### Tier 2: Literature (NEXT)

**Target:** Toroidal ballooning empirical scalings

**Approach:**
1. Find published ballooning γ(β, ε, n) data
2. Compare v2.0 measurements
3. Validate within literature scatter

**NOT pursuing:**
- ❌ Tearing benchmarks (out of scope)
- ❌ BOUT++ comparison (incompatible models)

---

## Communication Guidelines

### Internal (Team)

**Be honest:**
- v2.0 = ballooning specialist
- Tearing requires future development
- Not a limitation — designed focus

### External (Publication/Reviewers)

**Frame positively:**
- "Optimized for ballooning-type instabilities"
- "Demonstrates RL control in realistic tokamak geometry"
- "Extension to tearing modes requires 3D full MHD (future work)"

**Avoid defensive:**
- ❌ "Current version limited to..."
- ✅ "Designed for ballooning mode control, with roadmap toward..."

---

## Conclusion

**v2.0 Status (Honest Assessment):**

**Strengths:**
- ✅ Well-designed for ballooning control
- ✅ Physics validated within scope
- ✅ Structure-preserving framework works
- ✅ RL demonstration successful

**Limitations (Acknowledged):**
- ⚠️ Tearing mode not supported (by design)
- ⚠️ 2D physics model (not general 3D)
- ⚠️ Specialized, not universal

**Path Forward:**
- ✅ Complete ballooning validation (literature)
- ✅ Publish v2.0 as ballooning specialist
- ✅ Plan v3.0/PyTokMHD for tearing (3D extension)

---

**This clarification enables:**
- Honest benchmark design
- Appropriate validation criteria
- Clear publication narrative
- Realistic future work planning

**小P签字** ⚛️  
**Date:** 2026-03-23
