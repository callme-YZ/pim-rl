# RL Training Results (v2.0)

This directory contains complete training records for PIM-RL v2.0 RL experiments.

## Phase 2.0: Baseline Training

**Objective:** Validate basic RL learning with structure-preserving MHD

**Results:**
- Training: 200k steps (PPO)
- Improvement: +32.1% (-3.40 → -2.31 reward)
- Episode stability: 100 steps (vs v1.4's 77-step crash)
- Training time: ~2h (8-core, 46 FPS)

**Files:**
- `phase2.0_baseline/REPORT.md` - Training summary
- `phase2.0_baseline/best_model.zip` - Best checkpoint (step 15k)
- `phase2.0_baseline/checkpoint_200000_steps.zip` - Final model
- `phase2.0_baseline/training.log` - Last 1000 lines of training log

## Phase 2.1: Energy Penalty Ablation

**Objective:** Test energy penalty in reward function

**Variants tested:**
1. Baseline (no penalty)
2. Energy penalty λ=0.1
3. Energy penalty λ=1.0
4. Energy penalty λ=10.0
5. Multi-objective (amplitude + energy)

**Key finding:** Baseline performs best (-2.61), energy penalties harmful

**Files:**
- `phase2.1_ablation/analysis.json` - Complete results (5 variants)
- `phase2.1_ablation/train_phase2.1_reward_shaping.py` - Training script

## Reproducibility

To reproduce training:

```bash
cd experiments/v2.0

# Phase 2.0 baseline
python train_v2_ppo.py --steps 200000

# Phase 2.1 ablation
python training_results/phase2.1_ablation/train_phase2.1_reward_shaping.py
```

Expected results should match `phase2.1_ablation/analysis.json`.

---

**Training completed:** 2026-03-21  
**Validated by:** YZ Research Team
