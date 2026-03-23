# Hamiltonian Policy Architecture Design (v2.0 REVISED)

**Date:** 2026-03-22 19:59  
**Version:** v2.0 Draft 2 (Post-小P Review)  
**Status:** Step 3.1 Design Phase - REVISED

**Changes from Draft 1:**
- 🔧 Fixed Issue 1: H不是conserved Hamiltonian → pseudo-Hamiltonian
- 🔧 Fixed Issue 2: Dropped canonical (q,p) structure
- 🔧 Fixed Issue 3: Removed conservation penalty
- 🔧 Updated success criteria

---

## 1. Overview (REVISED)

**Goal:** Design RL policy that uses learned Hamiltonian-like function for structure-aware MHD control

**Key Innovation:**
- Policy learns differentiable function H(z, a)
- Uses ∂H/∂a for control gradient
- **H is NOT physical conserved Hamiltonian** ⚠️
- **H is learned function for control guidance** ✅

**Critical Physics Clarification (小P feedback):**
- ❌ MHD is NOT Hamiltonian system (has dissipation)
- ❌ Energy is NOT conserved (resistivity causes loss)
- ✅ H(z,a) is **pseudo-Hamiltonian** or **control Lyapunov function**
- ✅ ∂H/∂a provides useful gradient, not physics conservation

**Based on Phase 2 validated concepts:**
- ✅ HNN as differentiable function (Step 2.1)
- ✅ Latent space encoding (Step 2.2)
- ✅ Gradient-based control (Step 2.3)
- ⚠️ **But NOT canonical structure** (not applicable to MHD)

---

## 2. Problem Statement (Unchanged)

**Environment:** v2.0 MHD (小P's complete solver)

**Input (Observation):**
- 113-dimensional state vector
- Components: ψ field, mode amplitudes, island width, etc.

**Output (Action):**
- 4-dimensional RMP control

**Objective:**
- Suppress tearing mode (minimize island width)
- **Stable control** (not blow up)
- **Bounded energy dissipation** (not conserve, but reasonable)

---

## 3. Architecture Components (REVISED)

### 3.1 Observation Encoder (REVISED)

**Purpose:** Map high-dim MHD state → low-dim latent space

**Architecture:**
```
Input: obs ∈ ℝ¹¹³

Encoder Network:
  FC(113 → 256) → ReLU
  FC(256 → 128) → ReLU
  FC(128 → 64) → ReLU
  FC(64 → latent_dim)

Output: z ∈ ℝᴰ (D = latent_dim, typically 8-16)
```

**Latent Structure (REVISED):**
- ❌ **NOT** z = [z_q, z_p] (canonical coordinates)
- ✅ z = generic latent encoding
- ✅ No forced symplectic structure
- ✅ Let encoder learn任意useful representation

**Design choice:**
- Latent_dim = 8
- Represents compressed MHD state
- No physics constraint on structure

---

### 3.2 Pseudo-Hamiltonian Network (REVISED)

**Purpose:** Learn differentiable H(z, a) for control gradient

**Architecture:**
```
Input: [z, a] ∈ ℝ⁸⁺⁴ = ℝ¹²

HNN Network:
  FC(12 → 256) → Tanh
  FC(256 → 256) → Tanh
  FC(256 → 128) → Tanh
  FC(128 → 1)

Output: H(z, a) ∈ ℝ (scalar function)
```

**Physics Clarification (小P):**
- ❌ H is NOT physical energy
- ❌ H is NOT conserved
- ✅ H is learned function
- ✅ ∂H/∂a gives useful control gradient
- ✅ Can interpret as Lyapunov-like (decreases with good control)

**Training:**
- Supervised: match empirical reward/value
- No conservation constraint
- No symplectic structure requirement

---

### 3.3 Policy Network (Option B, REVISED)

**Purpose:** Output action distribution using H-gradient

**Hamiltonian-Informed PPO Policy:**

```
Standard PPO policy head:
  Actor: z → μ(a), σ(a)
  Critic: z → V(z)

Additional H-gradient loss:
  L_total = L_PPO + λ_H · L_pseudo_hamiltonian

L_pseudo_hamiltonian (REVISED):
  - Match H(z,a) to empirical value/reward
  - Align ∂H/∂a with advantage
  - NO conservation penalty ❌
```

**Pros:**
- Combines RL flexibility + gradient structure
- Physics-inspired (not physics-constrained)
- Trainable with standard PPO

---

### 3.4 Critic Network (Unchanged)

**Purpose:** Value function estimation

**Architecture:**
```
Input: z ∈ ℝ⁸
Critic Network:
  FC(8 → 128) → ReLU
  FC(128 → 128) → ReLU
  FC(128 → 1)
Output: V(z) ∈ ℝ
```

---

## 4. Training Procedure (REVISED)

### 4.1 Pre-training Phase (Optional)

**Goal:** Train encoder + H-network on offline data

**Data source:**
- Collect trajectories from v2.0 MHD env
- Random or scripted actions
- Record (obs, action, next_obs, reward)

**Training (REVISED):**
1. Train encoder (autoencoder style)
   - Reconstruct obs from z
   - ❌ No canonical structure loss
2. Train H-network
   - Predict value/reward from (z, a)
   - ❌ No conservation requirement
3. Validate H prediction accuracy

**Duration:** ~1-2 days

---

### 4.2 RL Training Phase (REVISED)

**Algorithm:** PPO (Proximal Policy Optimization)

**Modified objective (REVISED):**
```
L_total = L_PPO + λ_H · L_pseudo_hamiltonian

L_PPO: standard PPO loss (policy + value)

L_pseudo_hamiltonian (REVISED):
  - Match H(z, a) to empirical value
  - Align ∂H/∂a with advantage

❌ REMOVED: L_conservation (unphysical for resistive MHD)
```

**Hyperparameters:**
- λ_H: H-gradient loss weight (start 0.1)
- Standard PPO hyperparams

**Training loop:**
1. Collect rollouts (MHD env)
2. Encode obs → z
3. Compute H and ∂H/∂a
4. Update policy + critic + H-network jointly
5. Log metrics (return, energy dissipation, H error)

**Duration:** ~100k steps

---

## 5. Network Specifications (Unchanged)

| Component | Input Dim | Output Dim | Hidden Layers | Activation | Params (est) |
|-----------|-----------|------------|---------------|------------|--------------|
| Encoder | 113 | 8 | [256,128,64] | ReLU | ~45k |
| H-network | 12 | 1 | [256,256,128] | Tanh | ~100k |
| Actor | 8 | 8 (μ+σ) | [128,128] | ReLU/Tanh | ~20k |
| Critic | 8 | 1 | [128,128] | ReLU | ~18k |
| **Total** | - | - | - | - | **~183k** |

---

## 6. Loss Functions (REVISED)

### 6.1 Encoder Loss (REVISED)

```python
L_encoder = MSE(obs_reconstructed, obs_true)

❌ REMOVED: L_canonical (unphysical for MHD)
```

### 6.2 H-Network Loss (REVISED)

```python
L_H = MSE(H(z,a), empirical_value)
     + alignment_loss(∂H/∂a, advantage)

❌ REMOVED: conservation_penalty (MHD不conserve)
❌ REMOVED: symplectic constraints (MHD不是Hamiltonian)
```

### 6.3 PPO + H Loss (REVISED)

```python
L_total = L_PPO_standard + λ_H · L_pseudo_hamiltonian

L_pseudo_hamiltonian = MSE(H_pred, V_empirical)
                      + alignment_loss(∂H/∂a, advantage)

alignment_loss:
  # ∂H/∂a should align with advantage
  dH_da = compute_gradient(H, action)
  return -dot(dH_da, advantage)
```

---

## 7. Evaluation Metrics (REVISED)

### 7.1 Physics Metrics (REVISED)

1. **Energy Dissipation (REVISED):**
   - ❌ NOT "conservation" (unphysical)
   - ✅ **Bounded dissipation:** Energy loss ∈ [1%, 20%] per episode
   - ✅ **Stable energy budget:** No explosion or collapse
   - Target: Similar or better than baseline

2. **H-Network Accuracy:**
   - H prediction error: MSE(H_pred, V_true)
   - Target: <10% relative error

3. **Gradient Alignment:**
   - ✅ ∂H/∂a correlation with advantage
   - ✅ NOT symplectic check (not applicable)

### 7.2 RL Metrics (Unchanged)

1. **Episode Return:**
   - Mean ± std over 100 episodes
   - Compare to baseline PPO

2. **Island Width Suppression:**
   - Final vs initial island width
   - Suppression effectiveness

3. **Training Stability:**
   - Convergence speed
   - Variance in returns

### 7.3 Ablation Metrics (Unchanged)

Compare:
- **A:** H-informed PPO (full)
- **B:** PPO + encoder (no H-network)
- **C:** Standard PPO (baseline)

---

## 8. Implementation Plan (Unchanged)

Week 1-2: Core components + integration  
Week 3: Pre-training (optional)  
Week 4-7: RL training  
Week 8-9: Evaluation  

---

## 9. Potential Challenges (REVISED)

### Challenge 1: Latent Dimension Choice (Unchanged)

**Solution:** Start D=8, ablate if needed

---

### Challenge 2: H-Network Training (REVISED)

**Issue:** H(z,a) might not align well with value

**Solution:**
- Use V(z) from critic as target
- Start λ_H small (0.01)
- Gradually increase
- Monitor alignment metrics

---

### Challenge 3: Dissipation vs Control (NEW)

**Issue:** MHD dissipates energy naturally, control adds forcing

**Solution:**
- Don't penalize dissipation
- Focus on control effectiveness
- H can decrease (Lyapunov-like)

---

## 10. Success Criteria (REVISED)

**Must achieve:**

1. ✅ **Stable energy budget** (not conserve, but bounded dissipation 1-20%)
2. ✅ Performance ≥ baseline + 32.1%
3. ✅ H prediction error <10%
4. ✅ Training converges within 100k steps
5. ✅ 小P cross-validation passed

**Stretch goals:**

- Energy dissipation <5% (minimal loss)
- Performance >50% gain
- H error <5%

---

## 11. Critical Physics Corrections (Summary)

**小P's 3 Core Issues → Fixed:**

### ✅ Fix 1: H Definition

**Old (Draft 1):**
> H = conserved Hamiltonian, energy conservation

**New (Draft 2):**
> H(z,a) = **learned pseudo-Hamiltonian**  
> - NOT physical energy  
> - NOT conserved  
> - Control-oriented differentiable function  
> - Lyapunov-like interpretation OK

---

### ✅ Fix 2: Latent Structure

**Old (Draft 1):**
> z = [z_q, z_p] (canonical coordinates)  
> Symplectic structure, Eq 7 loss

**New (Draft 2):**
> z = **generic latent encoding**  
> - No canonical structure  
> - No symplectic constraint  
> - Let encoder learn freely

---

### ✅ Fix 3: Conservation Penalty

**Old (Draft 1):**
> L_conservation: penalize ΔH when action constant

**New (Draft 2):**
> ❌ **REMOVED**  
> Resistive MHD **should** dissipate  
> No conservation penalty

---

## 12. Open Questions for 小P Re-Review

**Addressed from Draft 1:**

1. ✅ **H definition:** Now clarified as pseudo-Hamiltonian, not conserved
2. ✅ **Latent structure:** Dropped canonical (q,p)
3. ✅ **Conservation:** Changed to bounded dissipation
4. ✅ **D=8:** Still think reasonable, pending小P re-check

**New questions:**

5. **Lyapunov interpretation:** Can we interpret H as control Lyapunov function (decreasing with good control)?
6. **Dissipation bounds:** Is 1-20% energy loss per episode physically reasonable for v2.0 MHD?

---

## 13. Comparison: Draft 1 vs Draft 2

| Aspect | Draft 1 (❌) | Draft 2 (✅) |
|--------|-------------|-------------|
| H物理意义 | Conserved Hamiltonian | Pseudo-Hamiltonian |
| Latent structure | Canonical (q,p) | Generic encoding |
| Conservation | Required (<0.1%) | Bounded dissipation (1-20%) |
| Symplectic | Yes | No |
| Eq 7 loss | Yes | No |
| L_conservation | Yes | No |
| Physics rigor | 6/10 | ? (pending小P) |

---

## 14. Next Steps (After 小P Re-Review)

**If 小P approves Draft 2:**
- ✅ Proceed to Step 3.2 (implementation)
- ✅ Create `src/` modules
- ✅ Write unit tests

**If still issues:**
- 🔧 Further revisions
- 🔧 Re-review cycle

---

**Document Status:** ✅ Draft 2 Complete (Post-小P Feedback)  
**Author:** 小A 🤖  
**Reviewer:** 小P ⚛️ (pending re-review)  
**Next:** 小P re-validation

---

**Key Lesson Learned (小A):**

**Don't over-apply toy problem patterns to complex physics** ⚠️

- Step 2.1-2.3: Ideal Hamiltonian systems (pendulum, oscillators)
- v2.0 MHD: **Dissipative, non-Hamiltonian**
- **Critical difference:** Ideal vs resistive
- **Must adapt concepts, not copy blindly**

**小P是physics firewall** ⚛️ — 防止unphysical设计进入实现

---

_All design corrections documented. Physics-first approach._
