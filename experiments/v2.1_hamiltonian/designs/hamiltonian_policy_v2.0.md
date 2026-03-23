# Hamiltonian Policy Architecture Design (v2.0)

**Date:** 2026-03-22  
**Version:** v2.0 Draft 1  
**Status:** Step 3.1 Design Phase

---

## 1. Overview

**Goal:** Design RL policy that uses Hamiltonian Neural Networks for structure-preserving MHD control

**Key Innovation:**
- Policy learns in Hamiltonian framework
- Conserves physical structure (energy, momentum)
- Uses ∂H/∂action for control gradient

**Based on Phase 2 validated concepts:**
- ✅ Basic HNN (Step 2.1)
- ✅ Latent space HNN (Step 2.2)
- ✅ Controlled HNN (Step 2.3)

---

## 2. Problem Statement

**Environment:** v2.0 MHD (小P's complete solver)

**Input (Observation):**
- 113-dimensional state vector
- Components:
  - ψ field values (grid points)
  - Mode amplitudes (m/n pairs)
  - Derived quantities (island width, etc.)

**Output (Action):**
- 4-dimensional RMP control
- Components:
  - RMP amplitude
  - RMP phase
  - RMP frequency
  - (TBD: 4th component)

**Objective:**
- Suppress tearing mode (minimize island width)
- Maintain energy conservation
- Stable control

---

## 3. Architecture Components

### 3.1 Observation Encoder

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

**Latent Structure:**
- z = [z_q, z_p] (canonical coordinates)
- z_q: position-like variables (D/2 dimensions)
- z_p: momentum-like variables (D/2 dimensions)

**Inspired by:** Step 2.2 autoencoder

**Design choice:**
- Latent_dim = 8 (4 pairs of q,p)
- Represents 4 dominant MHD modes
- Small enough for HNN to learn
- Large enough to capture dynamics

---

### 3.2 Hamiltonian Neural Network

**Purpose:** Learn H(z, a) - energy function conditioned on action

**Architecture:**
```
Input: [z, a] ∈ ℝ⁸⁺⁴ = ℝ¹²

HNN Network:
  FC(12 → 256) → Tanh
  FC(256 → 256) → Tanh
  FC(256 → 128) → Tanh
  FC(128 → 1)

Output: H(z, a) ∈ ℝ (scalar Hamiltonian)
```

**Physics constraints:**
- Symplectic structure preserved
- dz/dt = (∂H/∂z_p, -∂H/∂z_q)
- Energy-like interpretation

**Inspired by:** Step 2.3 controlled oscillators

**Training:**
- Supervised: match true dynamics
- Auxiliary loss: canonical structure (Eq 7 from Greydanus)
- Conservation penalty: ΔH small when action constant

---

### 3.3 Policy Network

**Purpose:** Output action distribution using Hamiltonian gradient

**Two options:**

#### Option A: Direct Hamiltonian Gradient Policy

```
Given current state → z (via encoder)

For each candidate action a:
  Compute H(z, a)
  Compute ∂H/∂a

Select action that minimizes H
  (or uses ∂H/∂a as control signal)

Action distribution:
  μ(z) = -α · ∂H/∂a  (gradient descent on H)
  σ(z) = learned variance
```

**Pros:**
- Direct physics interpretation
- Guaranteed to decrease H
- Conservative (energy-based)

**Cons:**
- Might be too conservative
- Need good H(z,a) learning

---

#### Option B: Hamiltonian-Informed PPO Policy

```
Standard PPO policy head:
  Actor: z → μ(a), σ(a)
  Critic: z → V(z)

Additional Hamiltonian loss:
  L_total = L_PPO + λ · L_hamiltonian

L_hamiltonian:
  - H(z, a) should match empirical energy
  - ∂H/∂a should align with effective actions
  - Conservation penalty
```

**Pros:**
- Combines RL flexibility + physics structure
- Still trainable with standard RL
- Physics as regularizer

**Cons:**
- Less "pure" Hamiltonian
- Need to tune λ

---

**Recommendation: Start with Option B** ✅

**Reason:**
- More practical for RL training
- Physics as strong prior, not hard constraint
- Can ablate λ to see physics contribution

---

### 3.4 Critic Network

**Purpose:** Value function estimation

**Architecture:**
```
Input: z ∈ ℝ⁸

Critic Network:
  FC(8 → 128) → ReLU
  FC(128 → 128) → ReLU
  FC(128 → 1)

Output: V(z) ∈ ℝ (state value)
```

**Optional enhancement:**
- Use H(z, a) as additional input
- Encourages value aligned with Hamiltonian

---

## 4. Training Procedure

### 4.1 Pre-training Phase (Optional)

**Goal:** Train encoder + HNN on offline data

**Data source:**
- Collect trajectories from v2.0 MHD env
- Random or scripted actions
- Record (obs, action, next_obs, energy)

**Training:**
1. Train encoder (autoencoder style)
   - Reconstruct obs from z
   - Minimize reconstruction loss
2. Train HNN on latent dynamics
   - Predict dz/dt
   - Match empirical energy
3. Validate conservation

**Duration:** ~1-2 days

**Benefit:**
- Good initialization
- HNN learns physics before RL
- Faster RL convergence (hypothesis)

---

### 4.2 RL Training Phase

**Algorithm:** PPO (Proximal Policy Optimization)

**Modified objective:**
```
L_total = L_PPO + λ_H · L_hamiltonian + λ_cons · L_conservation

L_PPO: standard PPO loss (policy + value)

L_hamiltonian:
  - Match H(z, a) to empirical energy
  - Align ∂H/∂a with value gradient

L_conservation:
  - Penalize ΔH when action is constant
  - Encourage energy-preserving actions
```

**Hyperparameters:**
- λ_H: Hamiltonian loss weight (start 0.1)
- λ_cons: Conservation loss weight (start 0.01)
- Standard PPO hyperparams (from baseline)

**Training loop:**
1. Collect rollouts (MHD env)
2. Encode obs → z
3. Compute HNN gradients
4. Update policy + critic + HNN jointly
5. Log metrics (return, energy drift, H error)

**Duration:** ~100k steps (same as baseline)

---

## 5. Network Specifications

### Summary Table

| Component | Input Dim | Output Dim | Hidden Layers | Activation | Params (est) |
|-----------|-----------|------------|---------------|------------|--------------|
| Encoder | 113 | 8 | [256,128,64] | ReLU | ~45k |
| HNN | 12 | 1 | [256,256,128] | Tanh | ~100k |
| Actor | 8 | 8 (μ+σ) | [128,128] | ReLU/Tanh | ~20k |
| Critic | 8 | 1 | [128,128] | ReLU | ~18k |
| **Total** | - | - | - | - | **~183k** |

**Comparable to baseline PPO:** ~150k params ✅

---

## 6. Loss Functions (Detailed)

### 6.1 Encoder Loss

```python
L_encoder = MSE(obs_reconstructed, obs_true)
           + λ_aux · L_canonical(z)

L_canonical(z):
  # Encourage z_p ≈ ∂z_q/∂t (Eq 7 from Greydanus)
  z_p_pred = (z_q[t] - z_q[t+1]) / dt
  return MSE(z_p[t], z_p_pred)
```

### 6.2 HNN Loss

```python
L_HNN = MSE(dz_dt_pred, dz_dt_true)
       + λ_energy · MSE(H(z,a), energy_empirical)
       + λ_cons · conservation_penalty(H)

conservation_penalty(H):
  # When action is constant, H should not drift
  H_variance = Var([H(z[t], a) for t in trajectory])
  return H_variance
```

### 6.3 PPO + Hamiltonian Loss

```python
L_total = L_PPO_standard
         + λ_H · L_hamiltonian
         + λ_reg · L_regularization

L_hamiltonian = MSE(H_pred, H_empirical)
               + alignment_loss(∂H/∂a, advantage)

alignment_loss:
  # ∂H/∂a should point in same direction as advantage
  dH_da = compute_gradient(H, action)
  return -dot(dH_da, advantage)
```

---

## 7. Evaluation Metrics

### 7.1 Physics Metrics

1. **Energy Conservation:**
   - Drift% = |H(t_end) - H(t_0)| / |H(t_0)| × 100%
   - Target: <0.1%

2. **HNN Accuracy:**
   - H prediction error: MSE(H_pred, H_true)
   - Target: <5% relative error

3. **Hamiltonian Structure:**
   - Symplectic check: ∇_z H maintains structure
   - Poisson bracket relations

### 7.2 RL Metrics

1. **Episode Return:**
   - Mean ± std over 100 episodes
   - Compare to baseline PPO

2. **Island Width Suppression:**
   - Final island width vs initial
   - Suppression ratio

3. **Training Stability:**
   - Convergence speed
   - Variance in returns

### 7.3 Ablation Metrics

Compare:
- **A:** Hamiltonian PPO (full)
- **B:** PPO + HNN (no conservation loss)
- **C:** PPO + encoder (no HNN)
- **D:** Standard PPO (baseline)

Measure:
- Performance gap (A vs D)
- Physics correctness (energy drift)
- Contribution of each component

---

## 8. Implementation Plan

### Phase 1: Core Components (Week 1)

1. Implement `LatentEncoder` class
2. Implement `HamiltonianNN` class
3. Implement `HamiltonianPolicy` class
4. Unit tests for each

### Phase 2: Integration (Week 2)

1. Integrate with Stable-Baselines3 PPO
2. Custom callback for Hamiltonian logging
3. Environment wrapper (obs → latent)
4. End-to-end smoke test

### Phase 3: Pre-training (Optional, Week 3)

1. Collect offline data (10k steps)
2. Pre-train encoder + HNN
3. Save checkpoints
4. Validate conservation

### Phase 4: RL Training (Week 4-7)

1. Train Hamiltonian PPO
2. Train baseline PPO (parallel)
3. Monitor metrics
4. Hyperparameter tuning

### Phase 5: Evaluation (Week 8-9)

1. Run ablation studies
2. Statistical analysis
3. Plot comparisons
4. Generate report

---

## 9. Potential Challenges

### Challenge 1: Latent Dimension Choice

**Issue:** Too small → can't capture dynamics; Too large → HNN hard to train

**Solution:**
- Start with D=8 (4 mode pairs)
- Ablate D ∈ {4, 8, 16}
- Use reconstruction error + RL performance to decide

---

### Challenge 2: HNN Training Stability

**Issue:** Hamiltonian learning might be noisy

**Solution:**
- Pre-train on offline data
- Use conservative λ_H initially (0.01)
- Gradually increase to 0.1-0.5
- Monitor H prediction error

---

### Challenge 3: Action Space Coupling

**Issue:** 4-dim action might not decompose cleanly in Hamiltonian

**Solution:**
- Allow H(z, a) to be fully coupled
- Don't assume separable structure
- Let network learn coupling

---

### Challenge 4: Reward Shaping vs Hamiltonian

**Issue:** MHD reward (island width) ≠ Hamiltonian (energy)

**Solution:**
- H is auxiliary objective, not primary
- Primary: RL reward (island suppression)
- H as regularizer for physics consistency
- λ_H tunes trade-off

---

## 10. Success Criteria

**Must achieve (Step 3.4 validation):**

1. ✅ Energy drift <0.1% (better than baseline)
2. ✅ Performance ≥ baseline + 32.1%
3. ✅ HNN H prediction error <10%
4. ✅ Training converges within 100k steps
5. ✅ 小P cross-validation passed

**Stretch goals:**

- Energy drift <0.01%
- Performance >50% gain
- HNN error <5%
- Faster convergence than baseline

---

## 11. Dependencies

**From Phase 2:**
- ✅ HNN implementation patterns
- ✅ Latent encoding experience
- ✅ Conservation testing methodology

**From Phase 1 (needed):**
- ⏳ MHD Hamiltonian formulation (Week 5-6, 小P)
- ✅ v2.0 MHD environment (小P complete)

**External:**
- Stable-Baselines3 (PPO)
- PyTorch
- NumPy/SciPy

---

## 12. Code Structure

```
experiments/v2.1_hamiltonian/
├── designs/
│   └── hamiltonian_policy_v2.0.md (this file)
├── src/
│   ├── encoder.py              # LatentEncoder
│   ├── hnn.py                  # HamiltonianNN
│   ├── policy.py               # HamiltonianPolicy
│   ├── losses.py               # Custom loss functions
│   └── callbacks.py            # SB3 callbacks
├── scripts/
│   ├── pretrain_hnn.py         # Pre-training (optional)
│   ├── train_hamiltonian_ppo.py
│   ├── train_baseline_ppo.py
│   └── evaluate.py
├── tests/
│   ├── test_encoder.py
│   ├── test_hnn.py
│   └── test_policy.py
└── results/
    ├── logs/
    ├── models/
    └── plots/
```

---

## 13. Next Steps (After Design Approval)

**Step 3.1 Deliverables (this document):**
- ✅ Architecture diagram (described above)
- ✅ Component specifications
- ✅ Loss functions
- ✅ Training procedure
- ✅ Evaluation metrics

**Waiting for:**
- 小P physics review of design
- YZ approval to proceed to Step 3.2 (implementation)

**Once approved:**
- Begin Step 3.2: Implementation (Week 1-2)
- Create `src/` modules
- Write unit tests
- Smoke test integration

---

## 14. Open Questions for 小P Review

1. **Latent dimension:** Is D=8 (4 mode pairs) physically reasonable for v2.0 MHD?
2. **Hamiltonian formulation:** Can we approximate MHD energy as H(z, a)? (Will Phase 1 Week 5-6 answer this?)
3. **Conservation target:** Is <0.1% energy drift realistic for resistive MHD?
4. **Action coupling:** Should H(z, a) have specific structure, or fully coupled OK?

---

**Document Status:** ✅ Draft 1 Complete  
**Author:** 小A 🤖  
**Next:** 小P Physics Review ⚛️

---

_All design decisions documented. Not靠记忆._
