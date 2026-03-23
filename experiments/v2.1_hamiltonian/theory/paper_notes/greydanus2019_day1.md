# Greydanus 2019 - Day 1 Reading Notes

**Date:** 2026-03-22 18:05  
**Paper:** Hamiltonian Neural Networks (NeurIPS 2019)  
**Source:** ar5iv.org HTML version

---

## Today's Reading: Abstract + Introduction + Section 2 (Theory)

### Abstract Summary (小A's words)

**核心问题:** Neural networks struggle to learn basic physics laws

**解决方案:** Draw inspiration from Hamiltonian mechanics

**创新点:**
1. Train models that learn and respect **exact conservation laws**
2. **Unsupervised** manner (no explicit energy labels)
3. Side effect: perfectly **reversible in time**

**Results:**
- Trains faster than baseline NN
- Generalizes better
- Conserves energy-like quantities

**Test cases:**
- Two-body problem
- Pixel observations of pendulum

---

## Section 1: Introduction

### Motivation

**Why physics priors matter:**
- Gravity important for: image classification, RL walking, robot manipulation
- Same physical laws underlie diverse tasks
- But untrained NNs don't have physics priors

**The problem with baseline NNs:**
- Learn **approximate** physics from data
- Cannot learn **exact** conservation laws
- Example: Mass-spring system (Figure 1)
  - True system: conserves q² + p² (energy)
  - Baseline NN: **drifts over time** (energy not conserved)

**Research question:**
> "Can we define a class of neural networks that will precisely conserve energy-like quantities over time?"

**Solution approach:**
- Instead of crafting Hamiltonian by hand
- **Parameterize it with a neural network**
- **Learn it directly from data**

**Why general:**
- Almost all physical laws can be expressed as conservation laws

---

## Section 2: Theory

### Problem Setup: Predicting Dynamics

**Goal:** Learn dynamics of a system using NN

**Baseline approach (problems):**
1. Predicts next state given current state
2. Problem 1: Discrete "time steps" (time is continuous)
3. Problem 2: Doesn't learn exact conservation laws → drift

**Better approach (Equation 1):**
```
(q₁, p₁) = (q₀, p₀) + ∫[t₀ to t₁] S(q,p) dt
```
- Continuous time integration
- Neural ODEs take step in this direction

---

### Hamiltonian Mechanics (Quick Review)

**Coordinates:**
- q = (q₁,...,qₙ): positions
- p = (p₁,...,pₙ): momenta
- N coordinate pairs: (q₁,p₁)...(qₙ,pₙ)

**The Hamiltonian ℋ(q,p):**
- Scalar function
- Usually represents total energy

**Hamilton's Equations (Equation 2):**
```
dq/dt = ∂ℋ/∂p
dp/dt = -∂ℋ/∂q
```

**Key insight:**
- Moving in direction of **symplectic gradient** S_ℋ = (∂ℋ/∂p, -∂ℋ/∂q)
- Keeps ℋ output **exactly constant** (energy conservation!)
- Unlike regular gradient (which changes ℋ as quickly as possible)

**Applications:**
- Mass-spring, pendulum (easy)
- Celestial mechanics (chaotic for >2 bodies)
- Many-body quantum systems
- Fluid simulations
- Condensed matter physics

---

### Hamiltonian Neural Networks (HNN)

**Core idea:**
- Learn parametric function for **ℋ** (not S_ℋ)
- NN outputs single scalar "energy-like" value
- Take **in-graph gradient** ∂ℋ/∂(q,p)
- This gives S_ℋ for free!

**Loss function (Equation 3):**
```
L_HNN = ‖∂ℋ_θ/∂p - ∂q/∂t‖² + ‖∂ℋ_θ/∂q + ∂p/∂t‖²
```

**Training target:**
- Match (∂ℋ/∂p, -∂ℋ/∂q) to observed (dq/dt, dp/dt)

**Properties (bonus):**
1. **Perfectly reversible:** (q,p) at t₀ ↔ (q,p) at t₁ bijective
2. **Counterfactual tool:** Can manipulate energy by integrating along ∇ℋ
   - Example: "What if we added 1 Joule of energy?"

---

## Section 3: Learning from Data (Partial)

### Three Tasks

**Task 1: Ideal Mass-Spring**
- Hamiltonian: ℋ = ½kq² + p²/2m (k=m=1)
- Dataset: 25 train + 25 test trajectories
- Noise: Gaussian σ²=0.1
- Each trajectory: 30 observations of (q,p)

**Task 2: Ideal Pendulum**
- Hamiltonian: ℋ = 2mgl(1-cos q) + l²p²/2m
- Parameters: m=l=1, g=3
- Energy range: [1.3, 2.3] (transition from linear to nonlinear)
- Same dataset structure

**Task 3: Real Pendulum**
- Data from Schmidt & Lipson (Science paper)
- Noisy real-world data
- Has friction (doesn't strictly conserve energy)
- Test HNN on biased data

---

## Key Takeaways (Day 1)

**1. The core innovation:**
- Don't learn dynamics S directly
- Learn Hamiltonian ℋ (energy function)
- Dynamics come from ∂ℋ/∂(q,p) automatically

**2. Why it works:**
- Symplectic gradient keeps ℋ constant
- This **guarantees** energy conservation
- No drift over time

**3. Implementation trick:**
- Take **in-graph gradient** of NN output
- Loss on the gradient (not the output itself)
- This is rare in ML!

**4. Applications:**
- Any system with conserved quantities
- Not limited to energy (can be angular momentum, etc.)

---

## Questions for Tomorrow

1. How do they handle **friction** (non-conservative forces)?
2. What's the **network architecture** in detail?
3. How does **test error** compare to baseline quantitatively?
4. Can this extend to **high-dimensional systems** (e.g., many particles)?
5. What happens with **partial observations** (e.g., only see q, not p)?

---

## Next Steps

**Tomorrow (Day 2):**
- Read Section 3.1 (Methods detail)
- Read Section 4 (Results)
- Understand training procedure
- Look at Figure 1 closely (mass-spring comparison)

**Day 3:**
- Read Section 5-6 (Discussion + Related Work)
- Understand limitations
- Start thinking about MHD control application

---

**小A's understanding so far: 7/10**

**What I get:**
- Core idea (learn ℋ not S)
- Why symplectic gradient conserves energy
- Loss function derivation

**What I need to clarify:**
- Exact training procedure details
- How to handle real-world friction
- Extension to high-dim systems

---

**小A签字:** Day 1 reading完成,notes written,no memory reliance ✅

---

## Continuation: Section 3.1-4.1 (Methods & Results)

**Time:** 18:10  
**Status:** Continuing Day 1 reading

---

### Section 3.1: Methods Detail

**Training setup:**
- Learning rate: 10⁻³
- Optimizer: Adam
- Batch size: Full dataset (small datasets)

**Two models trained:**
1. **Baseline NN:**
   - Input: (q, p)
   - Output: (∂q/∂t, ∂p/∂t) directly
   
2. **HNN:**
   - Input: (q, p)
   - Output: scalar ℋ
   - Then compute: (∂ℋ/∂p, -∂ℋ/∂q) via in-graph gradient
   - This gives (∂q/∂t, ∂p/∂t)

**Network architecture (both models):**
- 3 layers
- 200 hidden units per layer
- tanh activations
- Training: 2000 gradient steps

**Targets:**
- Analytic time derivatives when possible
- Finite difference approximations otherwise

**Evaluation metrics (3 types):**
1. **L2 train loss** - fit to training data
2. **L2 test loss** - generalization
3. **MSE energy** - how much dynamics diverge over long integration
   - Integrate from random test point using Equation 1
   - Compare to ground truth
   - Uses 4th-order Runge-Kutta (scipy)
   - Error tolerance: 10⁻⁹

**Key insight on metrics:**
- Loss measures: fit to individual points
- Energy metric measures: long-term stability + conservation

---

### Section 3.2: Results

**Figure 2 analysis (4 columns):**

**Column 1: Trajectories**
- Baseline: gradually drifts from ground truth
- HNN: high accuracy (obscures black baseline in plots)

**Column 2: Coordinate MSE over time**
- Baseline: rapidly diverges
- HNN: does not diverge

**Column 3: HNN-conserved quantity**
- Looks like total energy
- Same scale as energy
- Differs by constant factor (acceptable!)

**Column 4: True total energy**
- Ground truth
- For comparison

**Training performance (Table 1):**
- HNNs train as quickly as baseline ✅
- Converge to similar final losses ✅
- BUT: dramatically outperform on MSE energy metric ⭐

**Why HNN wins:**
- Conserves quantity close to total energy
- Errors don't accumulate over time
- Baseline: errors accumulate → divergence

**Important note on energy:**
- HNN-conserved quantity ≠ exact total energy
- But very close (same scale, constant offset)
- This is OK! Energy is relative quantity
  - Example: cat at 0m vs 1m elevation has different potential energy

**Real pendulum (Task 3) finding:**
- Ground truth doesn't quite conserve energy (friction!)
- HNN still tries to conserve quantity
- **Fundamental limitation:** HNN assumes conserved quantity exists
- Can't account for friction/dissipation
- Would need to model friction separately

---

### Section 4: Modeling Larger Systems

**Motivation:**
- Previous tasks: 1 (p,q) pair
- Now: multiple (p,q) pairs
- Test case: Two-body problem (4 pairs = 8 DoF)

**Task 4: Two-body problem**

**Hamiltonian (Equation 6):**
```
ℋ = |p_CM|²/(m₁+m₂) + (|p₁|²+|p₂|²)/(2μ) + g·m₁m₂/|q₁-q₂|²
```
- μ = reduced mass
- p_CM = center of mass momentum
- g = gravitational constant

**Simplifications:**
- m₁ = m₂ = g = 1
- Center of mass momentum = 0

**Degrees of freedom:**
- 8 total: (x,y) position and momentum for 2 bodies
- "Interesting challenge"

**Section 4.1: Methods (partial)**

**Dataset:**
- 1000 near-circular two-body trajectories
- Initialization:
  - Center of mass = 0
  - Total momentum = 0
  - Radius r = ‖q₂-q₁‖ ∈ [0.5, 1.5]
- Initial velocities: perfectly circular orbits
- Then add Gaussian noise: σ² = 0.05 (for stability control)

---

## Updated Questions

**1. Friction handling (partially answered):**
- HNN assumes conserved quantity exists
- Can't model friction/dissipation
- Need separate friction model

**2. Network architecture (answered):**
- 3 layers, 200 hidden units, tanh
- Simple MLP

**3. Quantitative results (partially seen):**
- Table 1 shows comparison
- HNN dramatically better on energy MSE
- Need to see actual numbers

**4. High-dimensional systems (partially answered):**
- Two-body = 8 DoF works
- Need to see if scales further

**5. Partial observations:**
- Not yet addressed
- Still pending

**New questions:**
6. How does energy MSE scale with trajectory length?
7. What's the constant factor between HNN-conserved vs true energy?
8. Can we combine HNN + friction model?

---

## Progress Update

**Read so far:**
- Abstract ✅
- Introduction ✅
- Section 2 (Theory) ✅
- Section 3 (Tasks + Methods + Results) ✅
- Section 4.1 (Two-body setup, partial) ✅

**Still to read:**
- Section 4 (Two-body results)
- Section 5 (Pixel observations)
- Section 6 (Discussion)
- Section 7 (Related work)
- Section 8 (Conclusion)

**Time spent:** ~18 minutes total
**Understanding: 8/10** (improved from 7/10)

---

**小A notes:** Architecture clear now, energy conservation mechanism understood, limitations identified (friction) ✅

**Next session:** Finish Section 4 + start Section 5 (pixel observations interesting for v2.1!)

---

## Step 1.3: Section 4.2 + Section 5 (Two-body Results + Pixel Pendulum)

**Time:** 18:20 (YZ指令: 删除时间预估)  
**Status:** Continuing

---

### Section 4.2: Two-body Results

**Figure 3 analysis (2 rows):**

**Row 1: Energy conservation**
- HNN: conserves quantity nearly equal to total energy ✅
- Baseline: does not conserve energy ❌

**Row 2: Trajectory comparison**
- After **1 orbit:**
  - Baseline: completely diverged from ground truth ❌
  - HNN: small amount of error only ✅

- **Long-term (t=50 and beyond):**
  - Both models diverge
  - HNN diverges **much slower**
  - **Key:** Even when HNN diverges from true orbit, total energy remains stable
  - Baseline: energy decays to zero or spirals to infinity

**Quantitative results (Table 1, referenced):**
- Train/test losses: HNN ~10× better than baseline
- Energy MSE: HNN **several orders of magnitude** better ⭐⭐⭐
- This is the key metric!

**Scaling to 8 DoF:**
- HNN scales well ✅
- Architecture unchanged (same as Task 1-3)

**Three-body problem (Appendix B):**
- Preliminary results shown
- HNN outperforms baseline by considerable margin
- Still needs improvement (authors focus on two-body here)

---

### Section 5: Learning Hamiltonian from Pixels ⭐

**CRITICAL for v2.1 MHD application!**

**Task 5: Pixel Pendulum**

**Innovation:**
- Train HNN on **latent vectors** from autoencoder
- Not on (q,p) directly
- **First instance** of Hamiltonian learned from pixel data

**Why important for MHD:**
- MHD observations = field maps (like "pixels")
- Need to learn latent representation
- Then apply HNN in latent space

---

### Section 5.1: Methods

**Dataset:**
- OpenAI Gym Pendulum-v0 environment
- 200 trajectories × 100 frames each
- Action: "no torque" (free evolution)
- Constraint: max displacement π/6 radians

**Image preprocessing:**
1. Start: 400×400×3 RGB
2. Crop + desaturate + downsample → 28×28×1
3. **Concatenate with next frame** → 28×28×2 input
4. Why 2 frames? So velocity observable (no recurrence needed)

**Autoencoder architecture:**

**Encoder + Decoder:**
- 4 fully-connected layers each
- ReLU activations
- **Residual connections** (important!)
- 200 hidden units per layer
- **Latent vector z:** 2 dimensions (matches (q,p)!)

**Why fully-connected (not CNN)?**
- Simpler
- CNNs sometimes struggle with position info (cite [23])

**HNN component:**
- Same as Section 3 (3 layers, 200 hidden, tanh)
- Operates on latent z

**Training:**
- Same procedure as Section 4.1
- **Weight decay:** 10⁻⁵ (beneficial for this task)

**Loss function:**
- Different from previous tasks (Section mentions but truncated)
- Needs to combine:
  1. Autoencoder reconstruction loss
  2. HNN dynamics loss

---

## Key Insights for v2.1 MHD Control

**Pixel pendulum → MHD field observations:**

**Analogy:**
- Pixels → MHD field maps (ψ, B, etc.)
- 28×28×2 → Could be 64×64×2 (2 time steps)
- Latent z (2D) → Latent representation of MHD state

**Architecture for MHD (hypothetical):**
```
MHD field maps → Encoder → z (latent) → HNN(z) → ℋ
                               ↓
                         dz/dt = (∂ℋ/∂p_z, -∂ℋ/∂q_z)
                               ↓
                         Decoder → predicted field maps
```

**Key design choice:**
- Latent dimension = 2N (N pairs of (q,p))
- For MHD: maybe N=4 (for 2/1 + 3/2 modes)?
- Or N=8 (for more modes)?

**Challenges (needs Section 5 results):**
1. How accurate is autoencoder reconstruction?
2. Does HNN still conserve energy in latent space?
3. What's the tradeoff: latent dim vs conservation quality?

---

## Updated Questions

**Answered:**
1. ✅ Friction: HNN can't model, need separate component
2. ✅ Architecture: 3 layers, 200 hidden, tanh (simple)
3. ✅ Scaling: Works for 8 DoF (two-body)
4. ✅ Pixel observations: Yes! Use autoencoder + HNN

**New questions (from Section 5):**
9. What's the **exact loss function** for pixel pendulum?
10. How does **latent dimension** affect conservation quality?
11. Can we **disentangle** (q,p) in latent space z?
12. What's **reconstruction error** vs **dynamics error**?

---

## Progress Update

**Read so far:**
- Abstract ✅
- Introduction ✅
- Section 2 (Theory) ✅
- Section 3 (Tasks + Methods + Results) ✅
- Section 4 (Two-body problem complete) ✅
- Section 5.1 (Pixel pendulum methods) ✅

**Still to read:**
- Section 5.2 (Pixel pendulum results) ⏳
- Section 6 (Discussion) ⏳
- Section 7 (Related work) ⏳
- Section 8 (Conclusion) ⏳

**Coverage:** ~70% of paper

**Understanding: 8.5/10** (improved, pixel task清晰)

---

## Critical Takeaway for v2.1

**Pixel pendulum proves:**
- HNN can work with **learned representations** (not just hand-crafted (q,p))
- Autoencoder extracts latent dynamics
- HNN enforces conservation in latent space

**For MHD control:**
- Observation: ψ field maps (64×64 or similar)
- Encoder → latent z (e.g., 8D for 4 mode pairs)
- HNN(z) → conserved Hamiltonian
- Control: action = -∂ℋ/∂RMP in latent space
- Decoder → predicted ψ evolution

**This is the bridge from theory to real MHD observation!** ⭐⭐⭐

---

**小A notes:** Pixel pendulum section极其重要, 这是v2.1 MHD应用的关键! 必须仔细读Section 5.2结果.

**Next:** 读完Section 5.2 + 6-8, 然后总结.

---

## Step 1.4: Section 5.2 + Section 6 (Pixel Results + Properties)

**Status:** Finishing paper

---

### Section 5.2: Pixel Pendulum Results

**Loss function (3 terms):**

1. **HNN loss:** Standard Equation 3 loss
2. **Autoencoder loss:** L2 pixel reconstruction
3. **Auxiliary loss (Equation 7):**
   ```
   L_CC = ‖z_p^t - (z_q^t - z_q^{t+1})‖²
   ```

**Purpose of auxiliary loss:**
- Make z_p ≈ derivative of z_q
- Encourages latent (z_q, z_p) to have **canonical coordinate properties**
- Measured by **Poisson bracket relations**
- **Necessary for writing Hamiltonian** ⭐
- Domain-agnostic (works for any even-sized latent space)

**Results (Figure 4):**

**Trajectory comparison:**
- Baseline: rapidly decays to lower energy ❌
- HNN: remains close to ground truth for hundreds of frames ✅
- Integration in **latent space**, then project to pixels via decoder

**Quantitative (Table 1):**
- Train/test loss: HNN competitive with baseline
- Energy metric: HNN **dramatically outperforms** ⭐

**Key finding:**
- HNN conserves scalar quantity analogous to total energy
- Enables accurate long-term prediction
- Even in latent space!

---

### Section 6: Useful Properties of HNNs

**Property 1: Adding/Removing Energy**

**Riemann gradient (vs Symplectic):**
```
Symplectic: S_ℋ = (∂ℋ/∂p, -∂ℋ/∂q)  → conserves ℋ
Riemann:    R_ℋ = (∂ℋ/∂q,  ∂ℋ/∂p)  → changes ℋ
```

**Application (Figure 5):**
- Integrate S_ℋ at low energy (blue circle)
- Integrate R_ℋ (purple line) → "bump" to higher energy
- Integrate S_ℋ at high energy (red circle)

**Counterfactual reasoning:**
- "What if we applied a torque?"
- "What if we added 1 Joule?"
- **Control application:** Manipulate energy to desired level!

**For MHD control:**
- Could use R_ℋ to find RMP action that changes energy
- Then verify with S_ℋ integration

---

**Property 2: Perfect Reversibility**

**Memory bottleneck problem:**
- Large NNs: transient activations consume memory
- Need for backpropagation

**Existing solutions:**
- Semi-reversible models [[13,25,19]]
- Neural ODEs [[7]]
- **Problem:** Only approximately reversible (not bijective)

**HNN advantage:**
- **Guaranteed perfectly reversible** through time ✅
- Based on **Liouville's Theorem:**
  - "Density of particles in phase space is constant"
  - Implies: mapping (q,p) at t₀ ↔ (q,p) at t₁ is bijective

**Practical benefit:**
- Can reconstruct past activations exactly
- Reduces memory footprint
- Useful for very deep networks

---

## Table 1: Quantitative Results Summary

**Tasks 1-5 comparison (all values ×10³):**

**Metrics:**
- Train loss
- Test loss  
- **Energy MSE** (key metric!)

**Pattern across all tasks:**
- HNN competitive on train/test loss
- HNN **dramatically outperforms** on energy MSE
- Multiple orders of magnitude better

**Specific highlights:**
- Two-body: Energy MSE several orders of magnitude better
- Pixel pendulum: Even in latent space, energy conserved

---

## Final Understanding Summary

**Core innovation (recap):**
1. Learn Hamiltonian ℋ (not dynamics S)
2. Symplectic gradient keeps ℋ constant
3. Guarantees energy conservation

**Key results:**
- Works for 1D, 2D, 8D systems ✅
- Works with learned representations (pixels) ✅
- Energy MSE orders of magnitude better ✅

**Limitations:**
- Cannot model friction/dissipation
- Assumes conserved quantity exists
- Need separate model for non-conservative forces

**Extensions:**
- Three-body problem (preliminary)
- Counterfactual reasoning (R_ℋ integration)
- Memory-efficient (perfect reversibility)

---

## Application to v2.1 MHD Control

### Direct Analogy

**Pixel pendulum architecture:**
```
Pixels (28×28×2) → Encoder → z (2D) → HNN(z) → ℋ
                                ↓
                     S_ℋ = (∂ℋ/∂z_p, -∂ℋ/∂z_q)
                                ↓
                     Integrate → predicted z
                                ↓
                     Decoder → predicted pixels
```

**MHD control architecture (proposed):**
```
ψ field (64×64×2) → Encoder → z (8D) → HNN(z, RMP) → ℋ
                                        ↓
                         ∂ℋ/∂z conserved, ∂ℋ/∂RMP = action
                                        ↓
                         RL policy: learn π(RMP | z) to minimize tearing
                                        ↓
                         Decoder → predicted ψ evolution
```

### Key Design Choices

**1. Latent dimension:**
- Pixel pendulum: 2D (1 mode pair)
- MHD: 8D? (4 mode pairs: 2/1, 3/2, 4/3, 5/4)
- Or 4D? (2 dominant modes only)
- **Trade-off:** More dimensions = more physics, but harder to train

**2. Auxiliary loss (Equation 7):**
- **Critical** for canonical coordinates
- Ensures z_p ≈ ∂z_q/∂t
- Poisson bracket relations satisfied
- **Must include** in MHD version!

**3. Control integration:**
- HNN(z, RMP): Hamiltonian depends on observation + action
- Action gradient: ∂ℋ/∂RMP
- RL learns: π(RMP | z) to optimize ∂ℋ/∂RMP

**4. Conservation in latent space:**
- Pixel pendulum proves this works ✅
- MHD fields → latent z → conserved ℋ(z)
- **Even if real MHD has friction**, latent ℋ can be conserved

---

## Critical Questions for Phase 2

**From Greydanus 2019:**
1. How many latent dimensions needed for MHD?
2. Does auxiliary loss (Eq 7) work for MHD fields?
3. Can we identify z_q vs z_p in latent space?
4. How to incorporate RMP into Hamiltonian?

**New questions:**
5. Can we use Morrison bracket (v2.0) in latent space?
6. Does HNN + Morrison = double conservation?
7. How to handle multi-mode coupling in latent ℋ?

---

## Paper Complete! 🎉

**Read:**
- Abstract ✅
- Introduction ✅
- Section 2 (Theory) ✅
- Section 3 (Simple tasks) ✅
- Section 4 (Two-body) ✅
- Section 5 (Pixel pendulum) ✅
- Section 6 (Properties) ✅

**Skipped:**
- Section 7 (Related work) — Can read later if needed
- Section 8 (Conclusion) — Likely summary

**Coverage:** 95% of paper (all technical content)

---

## Final Understanding: 9/10 ⭐

**What 小A now fully understands:**
- ✅ Core HNN theory (symplectic gradient)
- ✅ Training procedure (in-graph gradient + loss)
- ✅ Architecture (simple MLP, 3 layers)
- ✅ Results (orders of magnitude better on energy)
- ✅ Pixel extension (autoencoder + HNN)
- ✅ Auxiliary loss necessity (canonical coordinates)
- ✅ Latent space conservation
- ✅ Counterfactual reasoning (R_ℋ integration)
- ✅ Perfect reversibility (Liouville's theorem)

**What 小A needs to verify by implementation:**
- ⏳ Actual energy conservation <0.01% on toy example
- ⏳ Latent dimension choice for MHD
- ⏳ Combining HNN + Morrison bracket

---

## Next Steps (Phase 1 Week 1)

**Tomorrow (no time estimates per YZ):**
1. Implement 1D pendulum with HNN
2. Verify energy conservation <0.01%
3. Write greydanus2019_summary.md

**Then:**
- Read Zhong 2020 (Symplectic ODE-Net)
- Read Chen 2020 (Symplectic RNN)
- Prepare for Morrison 1982/1998

---

**小A总结:**

Greydanus 2019 = **Foundation for v2.1**

**Key insight:** Pixel pendulum architecture直接适用于MHD field observations!

**Confidence:** Ready to implement toy example and move to Phase 1 Week 2 ✅

---

**小A签字:** Day 1 reading完成, 397+150=547行notes, 理解9/10, 准备实现 🤖📚✅
