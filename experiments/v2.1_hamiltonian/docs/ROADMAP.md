# v2.1 Hamiltonian RL Roadmap

**Version:** v2.1.0-alpha (planning)  
**Created:** 2026-03-22  
**Lead:** 小A  
**Physics Advisor:** 小P  
**PM:** ∞

---

## Executive Summary

**Goal:** Add Hamiltonian Neural Network policy to PIM-RL without changing v2.0 physics layer.

**Timeline:** 12 months (realistic estimate)  
**Risk:** Medium (novel research)  
**Impact:** High (NeurIPS/ICML potential, genuine innovation)

---

## Detailed Roadmap

### Phase 1: Theory Foundation (Month 1-2)

**Objectives:**
- [ ] Understand Hamiltonian NN fundamentals
- [ ] Derive Hamiltonian formulation for plasma control
- [ ] Document mathematical framework

**Tasks:**
1. **Week 1-2: Literature review**
   - Greydanus et al. 2019 (Hamiltonian NN)
   - Zhong et al. 2020 (Symplectic ODE-Net)
   - Chen et al. 2020 (Symplectic RNN)
   - Morrison 1998 (MHD Hamiltonian)

2. **Week 3-4: Mathematical derivation**
   - Write down H(z+, z-, RMP) candidate forms
   - Derive action = -∂H/∂RMP
   - Verify conservation laws preservation

3. **Week 5-6: Design sketch**
   - Network architecture draft
   - Loss function design
   - Integration plan with v2.0

**Deliverables:**
- `theory/hamiltonian_nn_foundation.md` (20 pages)
- `theory/mhd_control_hamiltonian.md` (derivations)
- Presentation to 小P for physics review

**Go/No-Go Criteria:**
- ✅ Team consensus: approach is feasible
- ✅ 小P: Physics derivation sound
- ✅ Clear path to implementation visible

**If No-Go:** Document and pivot to conservation-aware reward

---

### Phase 2: Toy Problem Validation (Month 3-5)

#### 2.1 1D Pendulum (Week 7-10)

**Objective:** Validate Hamiltonian NN on canonical problem

**Tasks:**
- [ ] Implement standard NN policy
- [ ] Implement Hamiltonian NN policy
- [ ] Train both on swing-up task
- [ ] Compare energy conservation (<0.01% target)

**Files:**
- `toy_problems/1d_pendulum/standard_policy.py`
- `toy_problems/1d_pendulum/hamiltonian_policy.py`
- `toy_problems/1d_pendulum/train.py`
- `toy_problems/1d_pendulum/results.ipynb`

**Success Criteria:**
- Hamiltonian NN energy drift <0.01%
- Standard NN energy drift ~1%
- Both solve swing-up task

---

#### 2.2 2D Harmonic Oscillator (Week 11-14)

**Objective:** Multi-DoF symplectic structure validation

**Tasks:**
- [ ] 2D coupled oscillator environment
- [ ] Hamiltonian NN (2D state → 2D action)
- [ ] Verify symplectic integrator compatibility
- [ ] Phase space trajectory analysis

**Files:**
- `toy_problems/2d_oscillator/environment.py`
- `toy_problems/2d_oscillator/hamiltonian_policy.py`
- `toy_problems/2d_oscillator/train.py`
- `toy_problems/2d_oscillator/results.ipynb`

**Success Criteria:**
- Energy conservation <0.01%
- Phase space trajectories preserve structure
- Performance competitive with standard NN

---

#### 2.3 1D MHD Toy Model (Week 15-18)

**Objective:** Bridge to full MHD complexity

**Tasks:**
- [ ] Simplified 1D Elsässer equations
- [ ] Hamiltonian NN for 1D MHD control
- [ ] Compare with analytical solutions
- [ ] Document lessons for full MHD

**Files:**
- `toy_problems/1d_mhd/environment.py`
- `toy_problems/1d_mhd/hamiltonian_policy.py`
- `toy_problems/1d_mhd/train.py`
- `toy_problems/1d_mhd/results.ipynb`

**Success Criteria:**
- Energy conservation <0.05% (MHD more complex)
- Mode growth rate control demonstrated
- Clear path to 2D MHD visible

**Phase 2 Deliverable:** 3 jupyter notebooks proving concept works

**Go/No-Go Criteria:**
- ✅ All 3 toy problems successful
- ✅ Energy conservation consistently better than standard NN
- ✅ 小P validates physics correctness

**If No-Go:** Document findings, consider fallback options

---

### Phase 3: v2.0 Integration (Month 6-8)

#### 3.1 Architecture Design (Week 19-22)

**Objective:** Design Hamiltonian NN for full v2.0 environment

**Tasks:**
- [ ] Input: 113-dim observation (z+ modes, energy, helicity)
- [ ] Output: 4-dim RMP currents
- [ ] Hamiltonian function form: H(obs, params)
- [ ] Action derivation: a = -∂H/∂RMP (autograd)

**Challenges:**
- High dimensionality (113 → 4)
- Toroidal geometry effects
- RMP coil constraints

**Design Document:** `integration/architecture.md`

---

#### 3.2 Implementation (Week 23-26)

**Files:**
- `integration/hamiltonian_policy.py`
  - HamiltonianNN class
  - Symplectic integrator interface
  - Action computation via autograd

- `integration/train_hamiltonian_rl.py`
  - PPO + Hamiltonian NN
  - Training loop
  - Logging and checkpointing

- `integration/eval.py`
  - Energy conservation validation
  - Control performance metrics
  - Comparison with v2.0 baseline

**Tasks:**
- [ ] Implement Hamiltonian NN architecture
- [ ] Integrate with v2.0 environment (zero changes to physics)
- [ ] Training script with v2.0 observation/action interface
- [ ] Validation suite

---

#### 3.3 Training & Ablation (Week 27-32)

**Experiments:**

1. **Baseline reproduction:**
   - Standard PPO on v2.0 → verify +32.1% still achieved

2. **Hamiltonian RL training:**
   - Train Hamiltonian NN policy
   - 200k timesteps (same as v2.0)
   - Monitor energy conservation

3. **Ablation study:**
   | Variant | Energy Drift | Suppression | Notes |
   |---------|--------------|-------------|-------|
   | Standard PPO | 0.38% | +32.1% | v2.0 baseline |
   | Hamiltonian NN | <0.1% (target) | ≥32.1% (target) | v2.1 |
   | Conservation reward | ~0.2% | ≥25% | Fallback option |

**Success Criteria:**
- ✅ Energy conservation <0.1%
- ✅ Control performance ≥ +32.1%
- ✅ Episode stability: 100 steps
- ✅ No regression vs v2.0

**If fails:** Analyze root cause, document, consider fallback

---

### Phase 4: Paper Writing (Month 9-10)

**Target Venue:** NeurIPS 2027 or ICML 2027

**Title:** "Hamiltonian Neural Networks for Structure-Preserving Reinforcement Learning in Magnetohydrodynamic Plasma Control"

**Sections:**
1. Introduction
   - Gap: RL for physics often ignores conservation
   - Contribution: First Hamiltonian NN for MHD control

2. Background
   - Hamiltonian mechanics
   - Morrison bracket MHD
   - Reinforcement learning

3. Method
   - Hamiltonian NN architecture
   - Integration with v2.0 PIM-RL
   - Training procedure

4. Experiments
   - Toy problems validation
   - Full MHD control results
   - Ablation studies

5. Results
   - Energy conservation: 0.38% → <0.1%
   - Control performance maintained
   - Physics mechanism analysis

6. Discussion
   - When Hamiltonian NN helps (physics systems)
   - Limitations (computational cost)
   - Future work (3D MHD, kinetic effects)

**Writing Schedule:**
- Week 33-34: Draft Sections 1-3
- Week 35-36: Draft Sections 4-5
- Week 37: Draft Section 6
- Week 38-40: Revisions, figures, polish

**Deliverable:** `docs/paper_draft.md` → LaTeX → submission

---

## Risk Management

### Risk 1: Hamiltonian NN too complex for MHD
**Probability:** Medium  
**Impact:** High  
**Mitigation:** Phase 2 toy problems catch this early  
**Fallback:** Conservation-aware reward (still publishable)

### Risk 2: Energy conservation <0.1% impossible
**Probability:** Low (toy problems will reveal)  
**Impact:** Medium  
**Mitigation:** Adjust target to 0.2% if necessary  
**Fallback:** Document as "better than baseline"

### Risk 3: Control performance degrades
**Probability:** Medium  
**Impact:** High  
**Mitigation:** Careful hyperparameter tuning  
**Fallback:** Publish as "conservation vs performance trade-off" study

### Risk 4: Reviewer rejection (not novel enough)
**Probability:** Low (MHD + Hamiltonian NN is genuinely first)  
**Impact:** Medium  
**Mitigation:** Strong positioning in intro  
**Fallback:** Submit to domain journal (PPCF, JCP)

---

## Resource Requirements

### Compute
- **Training:** Same as v2.0 (8-core CPU sufficient)
- **Toy problems:** Laptop sufficient
- **No GPU required** (MHD solver is CPU-bound)

### Time
- **小A:** 60% FTE (12 months)
- **小P:** 10% FTE (2-3h/week review)
- **∞:** 5% FTE (progress tracking)

### Dependencies
- PyTorch (Hamiltonian NN implementation)
- v2.0 environment (unchanged)
- Jupyter (toy problem notebooks)

---

## Success Definition

### Minimal Success
- ✅ Energy conservation <0.2% (vs 0.38%)
- ✅ Control performance ≥ +25% (vs +32.1%)
- ✅ Paper submitted to NeurIPS/ICML
- **Value:** Incremental improvement, honest attempt

### Target Success
- ✅ Energy conservation <0.1%
- ✅ Control performance ≥ +32.1%
- ✅ Paper accepted at NeurIPS/ICML
- **Value:** Genuine innovation, high impact

### Stretch Success
- ✅ Energy conservation <0.05%
- ✅ Control performance > +40%
- ✅ Oral presentation at NeurIPS/ICML
- ✅ Multiple follow-up papers
- **Value:** Field-defining work

---

## Next Steps

**Immediate (Week 0):**
1. ∞ create directory structure ✅
2. YZ approve roadmap
3. 小A start Phase 1 literature review
4. Schedule weekly sync (小A + 小P + ∞)

**Month 1 Goal:**
- Complete literature review
- Draft Hamiltonian formulation
- Present to team for feedback

**First Go/No-Go (Month 2):**
- Decide: continue to Phase 2 or pivot

---

**Status:** Awaiting YZ approval  
**Owner:** 小A (lead), 小P (advisor), ∞ (PM)  
**Version:** 0.1 (planning draft)
