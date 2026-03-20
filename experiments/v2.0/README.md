# PTM-RL v2.0 Experiments

**Started:** 2026-03-20  
**Status:** Early development - Framework validation phase  
**Branch:** `feature/v2.0-elsasser`

---

## Overview

v2.0 development experiments for Vector Elsässer MHD + RL framework.

**Current Focus:**
- Elsässer variables z± = ω ± ψ (2D reduced MHD)
- PyTokEq equilibrium integration
- Physics correctness validation
- RL framework baseline

---

## Files

### Core Environment
- **`mhd_elsasser_env.py`** (474 lines) - Gymnasium environment for Elsässer MHD
  - Observation: z±, derivatives, RMP state
  - Action: 5-coil RMP current control
  - Physics: β~0.17 (realistic tokamak parameters)
  - Stability: 100 steps verified (vs v1.4 77-step crash)

### Training Scripts
- **`train_v2_ppo.py`** (323 lines) - PPO training implementation
  - Hyperparameters: lr=3e-4, γ=0.99, batch_size=64
  - Multi-env support (n_envs=1 default)
  - Checkpointing + evaluation

- **`train_50k_baseline.py`** (152 lines) - 50k baseline run
  - **Status:** Running (PID in train_50k.pid)
  - Checkpoints: every 5000 steps
  - Eval: every 5000 steps
  - Target: Establish v2.0 baseline performance

### Verification & Profiling
- **`quick_verify.py`** (47 lines) - Fast environment sanity check
- **`verify_rmp_5000x.py`** (120 lines) - Long-term RMP stability test
- **`profile_env.py`** (81 lines) - Performance profiling
- **`jit_solver_wrapper.py`** (150 lines) - JIT compilation experiments

---

## Key Results (2026-03-20)

### Physics Validation ✅
**5k test completed:**
- ✅ All episodes: 100 steps (stable)
- ✅ No early termination
- ✅ β = 0.175 (realistic)
- ✅ Energy conservation < 0.2%
- ✅ Growth rate measured: γ~1.06

**vs v1.4 comparison:**
- v1.4 IC (β~10⁹): 77-step crash ❌
- v2.0 IC (β~0.17): 100-step stable ✅

**Root cause fix:**
- YZ's PyTokEq equilibrium setup
- Correct pressure/current profiles
- Realistic physics parameters

### RL Baseline (In Progress)
**50k training:**
- **Started:** 2026-03-20 23:46
- **PID:** Available in `train_50k.pid`
- **Log:** `train_50k.log`
- **Expected:** ~1-1.3 hours
- **Monitoring:** Every 5k steps

**5k preliminary:**
- Reward: -201 (flat, no learning yet)
- Framework: Working ✅
- Physics: Stable ✅

---

## Technical Details

### Environment Specs
```python
Observation space: Box(19,)
  - z+ field (64×64 flattened) → reduced to stats
  - z- field (64×64 flattened) → reduced to stats
  - Derivatives (∂z±/∂r, ∂z±/∂θ)
  - RMP state (5 coils)

Action space: Box(5,)
  - 5-coil RMP currents
  - Range: [-1, 1] normalized

Reward: -island_width
  - Measured via O-point detection
  - Goal: Minimize island growth

Physics:
  - Grid: 64×64 (r,θ)
  - Time step: dt = 0.01
  - Episode length: 100 steps
  - Resistivity: η = 1e-5
```

### Performance
- **Step time:** ~0.11 sec/step
- **FPS:** ~8-9 steps/sec
- **Episode time:** ~11 sec (100 steps)

---

## Next Steps

**Immediate (待50k完成):**
1. Validate 50k baseline results
2. Analyze learning curves
3. Verify RMP control effectiveness

**Short-term (v2.0 Phase 1):**
1. Morrison bracket implementation
2. Energy-conserving discretization
3. Structure-preserving validation

**Mid-term (v2.0 Phase 2-3):**
1. Vector Elsässer z± = u ± B (full 3D)
2. Incompressibility constraint
3. Production-ready RL training

---

## Development Log

### 2026-03-20
- **21:00-23:00** Environment development (小A)
  - mhd_elsasser_env.py created (474 lines)
  - PyTokEq integration
  - Physics validation

- **23:00-23:30** Quick verification
  - 5k test: 100% success ✅
  - Physics correctness confirmed
  - 小P cross-validation ⚛️

- **23:45** 50k baseline started
  - Background training launched
  - Monitoring active
  - Expected completion: ~01:00

---

**Team:**
- 小A 🤖: RL framework lead
- 小P ⚛️: Physics validation
- ∞: Coordination + Git management
- YZ 🐙: Direction + decisions

**Repository:** https://github.com/callme-YZ/ptm-rl  
**Branch:** `feature/v2.0-elsasser`
