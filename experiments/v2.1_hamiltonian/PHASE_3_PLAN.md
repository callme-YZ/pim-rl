# Phase 3: v2.0集成 (Month 6-8)

**Based on:** 原始12个月规划 (17:44截图)

**Goal:** 实现Hamiltonian policy并验证performance

**Duration:** Month 6-8 (3个月)

---

## Overview

**核心任务:**
- 实现Hamiltonian policy (113-dim obs → 4-dim action)
- 训练并ablation study (vs standard PPO)
- Target: <0.1% energy drift, performance ≥ +32.1%

**验收标准:**
- 小P cross-validation ✅

---

## Phase 3 详细步骤

### Step 3.1: Hamiltonian Policy架构设计

**Goal:** 设计policy network使用HNN

**Tasks:**
1. **Observation encoder:** 113-dim MHD state → latent z
   - Similar to Step 2.2 autoencoder
   - Canonical coordinates (q, p)
2. **HNN component:** Learn H(z)
   - 与Step 2.3 controlled HNN类似
   - H(z, action) formulation
3. **Policy output:** 
   - Compute ∂H/∂action
   - Map to 4-dim RMP control
4. **Architecture diagram**

**Deliverables:**
- Design document (`designs/hamiltonian_policy_v2.0.md`)
- Network architecture sketch
- 小P physics review

**Time:** Week 1-2

---

### Step 3.2: 实现Hamiltonian Policy

**Goal:** Code完整policy network

**Tasks:**
1. **Encoder network:**
   - Input: 113-dim MHD observation
   - Output: latent z (dimensionality TBD, 8-16D)
2. **HNN network:**
   - Input: z + action
   - Output: H(z, a)
3. **Policy integration:**
   - Compute symplectic gradient
   - Output action distribution
4. **Training infrastructure:**
   - Compatible with Stable-Baselines3
   - Custom policy class

**Deliverables:**
- `hamiltonian_policy.py` (完整实现)
- Unit tests
- Integration test with dummy env

**Time:** Week 3-4

---

### Step 3.3: Baseline对比训练

**Goal:** 训练Hamiltonian policy vs standard PPO

**Tasks:**
1. **Setup两组实验:**
   - Experiment A: Hamiltonian PPO
   - Experiment B: Standard PPO (baseline)
2. **相同条件:**
   - Same environment (v2.0 MHD)
   - Same hyperparameters (除policy结构)
   - Same training budget (100k steps)
3. **Metrics记录:**
   - Episode return
   - Energy drift
   - Island width suppression
   - Training stability

**Deliverables:**
- `experiments/phase3_ablation/train_hamiltonian_ppo.py`
- `experiments/phase3_ablation/train_standard_ppo.py`
- Training logs (TensorBoard)

**Time:** Week 5-8

---

### Step 3.4: Performance评估

**Goal:** 验证Hamiltonian policy达到target

**Metrics:**
1. **Energy conservation:** <0.1% drift
2. **Performance gain:** ≥ +32.1% vs baseline
3. **Suppression effectiveness:** Island width reduction
4. **Stability:** Training convergence

**Tasks:**
1. 运行evaluation (100 episodes each)
2. Statistical analysis
3. Ablation studies:
   - HNN vs no-HNN
   - Different latent dimensions
   - Different auxiliary loss weights

**Deliverables:**
- `PHASE_3_EVALUATION_REPORT.md`
- Performance plots
- Statistical comparison

**Time:** Week 9-10

---

### Step 3.5: 小P Cross-validation

**Goal:** 小P验证physics correctness

**Tasks:**
1. 小P review:
   - Hamiltonian formulation
   - Conservation properties
   - Control physics
2. 小P独立测试:
   - Energy conservation验证
   - Action effectiveness验证
3. 迭代修正 (if needed)

**Deliverables:**
- 小P validation report
- Physics corrections (if any)
- Final sign-off

**Time:** Week 11

---

### Step 3.6: 文档和总结

**Goal:** 完整记录Phase 3成果

**Tasks:**
1. **Technical documentation:**
   - Architecture完整说明
   - Training details
   - Hyperparameter choices
2. **Results summary:**
   - Performance metrics
   - Ablation insights
   - Lessons learned
3. **Code cleanup:**
   - Refactor for readability
   - Add comprehensive comments
   - Create examples

**Deliverables:**
- `PHASE_3_SUMMARY.md`
- Clean codebase
- Usage examples

**Time:** Week 12

---

## Success Criteria (验收标准)

**Must have (Required):**
1. ✅ Energy drift <0.1% (vs baseline)
2. ✅ Performance gain ≥ +32.1%
3. ✅ 小P cross-validation passed
4. ✅ Reproducible results

**Nice to have (Bonus):**
- Better than +32.1%
- Faster training convergence
- More stable policy
- Generalizes to different parameters

---

## Dependencies

**From Phase 2:**
- ✅ Step 2.1: Basic HNN implementation
- ✅ Step 2.2: Latent space HNN
- ✅ Step 2.3: Controlled HNN

**From Phase 1 (needed):**
- Week 5-6: MHD Hamiltonian formulation (小P)
- v2.0 MHD environment (小P已完成)

---

## Risk Mitigation

**Risk 1: Hamiltonian formulation难度**
- Mitigation: 依赖小P Week 5-6理论工作
- Backup: 简化Hamiltonian (partial conservation)

**Risk 2: 113-dim → latent mapping难**
- Mitigation: Step 2.2已验证concept
- Backup: 分阶段,先用low-dim state

**Risk 3: Training不稳定**
- Mitigation: Careful hyperparameter tuning
- Backup: 参考Step 2.3 training经验

---

## Phase 3 Timeline (12 weeks)

**Week 1-2:** Design (Step 3.1)  
**Week 3-4:** Implementation (Step 3.2)  
**Week 5-8:** Training (Step 3.3)  
**Week 9-10:** Evaluation (Step 3.4)  
**Week 11:** 小P validation (Step 3.5)  
**Week 12:** Documentation (Step 3.6)

---

## Notes

**Phase 2成功经验应用到Phase 3:**
1. ✅ 小步验证 (6个sub-steps)
2. ✅ 每步明确交付
3. ✅ 小P physics review
4. ✅ 快速迭代,及时调整

**Phase 3特别注意:**
- **写文件,不靠记忆** (YZ强调)
- **所有设计决策文档化**
- **所有实验结果记录**
- **所有代码注释完整**

---

**Created:** 2026-03-22 19:43  
**Owner:** 小A 🤖 (RL/ML研究员)  
**Reviewer:** 小P ⚛️ (Physics验证)  
**PM:** YZ (最终决策)

---

_This is Phase 3 master plan. 所有细节在这个文件,不靠记忆._
