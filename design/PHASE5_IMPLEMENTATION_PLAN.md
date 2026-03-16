# Phase 5: RL Interface - Implementation Plan

**Author:** 小A 🤖 (RL Lead)  
**Date:** 2026-03-16  
**Physics Review:** 小P ⚛️ APPROVED  
**Version:** 1.0

---

## Implementation Overview

**Timeline:** 4 weeks  
**Start:** Week 1 Day 1 (YZ批准后立即开始)  
**End:** Week 4 Day 5 (Phase 5 completion report)

**Key Milestones:**
- M5.1: Environment implementation (Week 1) ✅
- M5.2: PPO training完成 (Week 2) ✅
- M5.3: Baseline对比完成 (Week 3) ✅
- M5.4: Phase 5完成报告 (Week 4) ✅

---

## Week 1: Environment Implementation

### Day 1-2: Gym Wrapper Implementation

**Task:** 实现MHDTearingControlEnv

**Location:** `src/pytokmhd/rl/env.py`

**Implementation:**
```python
import gym
import numpy as np
from gym import spaces

from pytokmhd.solver import (
    rk4_step_with_rmp,
    setup_tearing_mode,
    create_equilibrium_cache
)
from pytokmhd.diagnostics import TearingModeMonitor

class MHDTearingControlEnv(gym.Env):
    """
    Gym environment for RL-based tearing mode control.
    
    Observation: 18D (w, gamma, x_o, z_o, psi×8, omega×8, energy, helicity, drift, prev_action, t, dt)
    Action: Continuous RMP amplitude ∈ [-0.1, 0.1]
    Reward: -w - 0.1*|gamma| - 0.01*|action| + convergence_bonus
    """
    
    def __init__(
        self,
        Nr=64,
        Nz=128,
        dt=0.01,
        eta=1e-3,
        nu=1e-3,
        m=2,
        n=1,
        A_max=0.1,
        max_steps=200,
        w_0=0.01,  # Initial island width
        convergence_threshold=0.005
    ):
        super().__init__()
        
        # Environment config
        self.Nr = Nr
        self.Nz = Nz
        self.dt = dt
        self.eta = eta
        self.nu = nu
        self.m = m
        self.n = n
        self.A_max = A_max
        self.max_steps = max_steps
        self.w_0 = w_0
        self.convergence_threshold = convergence_threshold
        
        # Grid setup
        self.r = np.linspace(0.1, 1.0, Nr)
        self.z = np.linspace(0, 2*np.pi, Nz)
        self.dr = self.r[1] - self.r[0]
        self.dz = self.z[1] - self.z[0]
        R, Z = np.meshgrid(self.r, self.z, indexing='ij')
        self.r_grid = R
        
        # Equilibrium cache (Phase 2)
        self.eq_cache = create_equilibrium_cache(self.r, self.z)
        
        # Diagnostics monitor (Phase 3)
        self.monitor = TearingModeMonitor(m=m, n=n)
        
        # Gym spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(18,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        
        # State
        self.psi = None
        self.omega = None
        self.t = 0.0
        self.step_count = 0
        self.prev_action = 0.0
        
    def reset(self):
        """
        Reset environment to initial tearing mode state.
        """
        # Get equilibrium from cache
        eq = self.eq_cache.get_equilibrium(beta_p=0.5, I_p=1e6)
        psi_eq = eq['psi']
        q_profile = eq['q_profile']
        
        # Setup tearing mode (Phase 1)
        from pytokmhd.diagnostics import find_rational_surface
        r_s, _ = find_rational_surface(q_profile, self.r, 2.0)
        
        self.psi, self.omega = setup_tearing_mode(
            psi_eq, self.r, self.z,
            m=self.m, n=self.n,
            eta=self.eta,
            w_0=self.w_0,
            rational_surface_r=r_s
        )
        
        # Reset time
        self.t = 0.0
        self.step_count = 0
        self.prev_action = 0.0
        
        # Compute initial observation
        obs = self._get_observation()
        
        return obs
    
    def step(self, action):
        """
        Execute one step of MHD evolution with RMP control.
        
        Args:
            action: np.array([rmp_amplitude]) ∈ [-1, 1]
        
        Returns:
            obs, reward, done, info
        """
        # Scale action to physical range
        rmp_amplitude = float(action[0]) * self.A_max
        
        # Execute MHD step with RMP (Phase 4)
        self.psi, self.omega = rk4_step_with_rmp(
            self.psi, self.omega,
            self.dt, self.dr, self.dz,
            self.r_grid,
            self.eta, self.nu,
            rmp_amplitude=rmp_amplitude,
            m=self.m, n=self.n
        )
        
        # Update time
        self.t += self.dt
        self.step_count += 1
        self.prev_action = rmp_amplitude
        
        # Get observation
        obs = self._get_observation()
        
        # Compute reward
        reward = self._compute_reward(obs, rmp_amplitude)
        
        # Check termination
        done = self._check_done(obs)
        
        # Info
        info = {
            'w': obs[0],
            'gamma': obs[1],
            't': self.t,
            'step': self.step_count
        }
        
        return obs, reward, done, info
    
    def _get_observation(self):
        """
        Construct 18D observation vector.
        """
        # Phase 3 diagnostics
        diag = self.monitor.update(
            self.psi, self.omega, self.t,
            self.r, self.z,
            q_profile=None  # Can recompute if needed
        )
        
        if diag is None:
            # No island detected (equilibrium state)
            w, gamma, x_o, z_o = 0.0, 0.0, 0.5, 0.0
        else:
            w = diag['w']
            gamma = diag['gamma']
            x_o = diag['x_o']
            z_o = diag['z_o']
        
        # Sample MHD state (8 psi + 8 omega)
        # Sample points: evenly spaced in r
        r_samples = np.linspace(0, self.Nr-1, 8, dtype=int)
        z_sample = self.Nz // 2  # Mid-plane
        
        psi_samples = self.psi[r_samples, z_sample]
        omega_samples = self.omega[r_samples, z_sample]
        
        # Conservation quantities
        energy = self._compute_energy()
        mag_helicity = self._compute_helicity()
        energy_drift = 0.0  # Placeholder (would need history)
        
        # Context
        prev_action_norm = self.prev_action / self.A_max
        t_norm = self.t / (self.max_steps * self.dt)
        dt_since_reset = self.step_count * self.dt
        
        # Construct observation (18D)
        obs = np.array([
            w, gamma, x_o, z_o,  # 4D
            *psi_samples,  # 8D
            *omega_samples,  # 8D
            energy, mag_helicity, energy_drift,  # 3D
            prev_action_norm, t_norm, dt_since_reset  # 3D
        ], dtype=np.float32)
        
        return obs
    
    def _compute_reward(self, obs, action):
        """
        Compute reward: -w - 0.1*|gamma| - 0.01*|action| + bonus
        """
        w = obs[0]
        gamma = obs[1]
        
        # Main penalties
        width_penalty = -w
        growth_penalty = -0.1 * abs(gamma)
        effort_penalty = -0.01 * abs(action)
        
        # Convergence bonus
        if w < self.convergence_threshold:
            convergence_bonus = 1.0
        else:
            convergence_bonus = 0.0
        
        reward = width_penalty + growth_penalty + effort_penalty + convergence_bonus
        
        return reward
    
    def _check_done(self, obs):
        """
        Check episode termination.
        """
        # Max steps reached
        if self.step_count >= self.max_steps:
            return True
        
        # NaN/Inf check (numerical instability)
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            return True
        
        return False
    
    def _compute_energy(self):
        """
        Compute total MHD energy.
        """
        # Simplified: |∇ψ|^2 + |ω|^2
        energy = np.sum(self.psi**2 + self.omega**2)
        return energy
    
    def _compute_helicity(self):
        """
        Compute magnetic helicity.
        """
        # Simplified: ∫ A·B dV ≈ ∫ ψ·ω dV
        helicity = np.sum(self.psi * self.omega)
        return helicity
    
    def render(self, mode='human'):
        """
        Render environment (optional).
        """
        pass
```

**Deliverables:**
- ✅ `src/pytokmhd/rl/env.py` (完整实现)
- ✅ `src/pytokmhd/rl/__init__.py` (导出)

---

### Day 3-4: Unit Tests

**Location:** `src/pytokmhd/tests/test_rl_env.py`

**Test Cases:**
```python
import pytest
import numpy as np
from pytokmhd.rl import MHDTearingControlEnv

def test_env_creation():
    """Test environment can be created."""
    env = MHDTearingControlEnv()
    assert env is not None
    
def test_reset():
    """Test environment reset."""
    env = MHDTearingControlEnv()
    obs = env.reset()
    
    assert obs.shape == (18,)
    assert not np.any(np.isnan(obs))
    assert not np.any(np.isinf(obs))
    
def test_step():
    """Test environment step."""
    env = MHDTearingControlEnv()
    obs = env.reset()
    
    action = np.array([0.5])  # RMP amplitude
    obs, reward, done, info = env.step(action)
    
    assert obs.shape == (18,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert 'w' in info
    
def test_random_policy():
    """Test random policy rollout."""
    env = MHDTearingControlEnv(max_steps=50)
    obs = env.reset()
    
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            break
    
    # Should complete without NaN/Inf
    assert not np.any(np.isnan(obs))
    
def test_conservation():
    """Test energy conservation (no control)."""
    env = MHDTearingControlEnv(max_steps=20)
    obs = env.reset()
    
    energy_initial = obs[8]  # energy in obs
    
    for _ in range(20):
        action = np.array([0.0])  # No control
        obs, _, _, _ = env.step(action)
    
    energy_final = obs[8]
    
    # Energy should be conserved (within numerical error)
    drift = abs(energy_final - energy_initial) / energy_initial
    assert drift < 0.1  # <10% drift acceptable for 20 steps
```

**Run tests:**
```bash
cd /Users/yz/.openclaw/workspace-xiaoa/ptm-rl
pytest src/pytokmhd/tests/test_rl_env.py -v
```

**Acceptance:**
- ✅ All tests pass
- ✅ Random policy runs without crash
- ✅ Conservation reasonable (<10% drift)

---

### Day 5: Physics Review Checkpoint

**小P Review内容:**

**1. Observation物理充分性**
- ✅ w, gamma, x_o, z_o: 核心物理量
- ✅ psi/omega samples: 捕捉MHD state
- ✅ Conservation监控: energy, helicity

**2. Action范围物理合理性**
- ✅ [-0.1, 0.1]: Phase 4已验证

**3. Reward无物理违反**
- ✅ Minimize w: 物理目标
- ✅ Penalize gamma: 合理
- ✅ No unphysical constraints

**验收标准:**
- ✅ 所有unit tests通过
- ✅ Random policy稳定运行
- ✅ Conservation drift <10%

**小P签字: Environment design physics approved** ✅

---

## Week 2: Training Pipeline

### Day 1-2: PPO Baseline Training

**Setup Stable-Baselines3:**
```bash
pip install stable-baselines3[extra]
pip install tensorboard
```

**Training script:** `scripts/train_ppo_baseline.py`

```python
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from pytokmhd.rl import MHDTearingControlEnv

# Create environment
env = MHDTearingControlEnv(
    Nr=64, Nz=128,
    dt=0.01,
    max_steps=200,
    w_0=0.01  # Start easy
)

# PPO config (gamma=0.95 baseline)
model = PPO(
    'MlpPolicy',
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    gamma=0.95,  # Baseline
    n_epochs=10,
    clip_range=0.2,
    tensorboard_log="./logs/ppo_baseline/",
    verbose=1
)

# Checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path='./checkpoints/ppo_baseline/',
    name_prefix='ppo_model'
)

# Train 10k steps pilot
model.learn(
    total_timesteps=10000,
    callback=checkpoint_callback,
    tb_log_name="gamma_0.95"
)

# Save final model
model.save("models/ppo_baseline_10k")
```

**Run:**
```bash
python scripts/train_ppo_baseline.py
```

**Monitor:**
```bash
tensorboard --logdir logs/ppo_baseline
```

**Metrics to track:**
- Episode return
- Island width trajectory
- Control effort
- Policy entropy

---

### Day 3-4: Gamma Tuning

**小P建议: gamma=0.98 (physics-motivated)**  
**小A策略: empirical试[0.95, 0.98, 0.99]**

**Training script:** `scripts/train_ppo_gamma_sweep.py`

```python
gamma_values = [0.95, 0.98, 0.99]

for gamma in gamma_values:
    env = MHDTearingControlEnv(...)
    
    model = PPO(
        'MlpPolicy', env,
        gamma=gamma,  # Vary gamma
        # ... other params same
        tensorboard_log=f"./logs/gamma_{gamma}/",
    )
    
    model.learn(
        total_timesteps=50000,  # Longer run
        tb_log_name=f"gamma_{gamma}"
    )
    
    model.save(f"models/ppo_gamma_{gamma}")
```

**Evaluation:**
```python
# Compare final performance
for gamma in gamma_values:
    model = PPO.load(f"models/ppo_gamma_{gamma}")
    
    # Evaluate on 10 episodes
    rewards = []
    for _ in range(10):
        obs = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
    
    print(f"Gamma={gamma}: mean_reward={np.mean(rewards):.2f}±{np.std(rewards):.2f}")
```

**Decision:**
- 选择mean reward最高的gamma
- 记录所有实验到文档

---

### Day 5: 100k Training Run

**最佳gamma (假设0.98):**

```python
model = PPO(
    'MlpPolicy',
    env,
    gamma=0.98,  # Best from sweep
    # ... other params
    tensorboard_log="./logs/ppo_final/",
)

model.learn(
    total_timesteps=100000,
    tb_log_name="ppo_final_100k"
)

model.save("models/ppo_final_100k")
```

**Deliverables:**
- ✅ Trained PPO policy (100k steps)
- ✅ TensorBoard logs
- ✅ Checkpoints
- ✅ Gamma tuning report

---

## Week 3: Baseline Comparison

### Day 1-2: P/PID Baseline运行

**Proportional control:**
```python
from pytokmhd.control import RMPController

controller = RMPController(
    control_type='proportional',
    K_p=1.0,
    A_max=0.1
)

# Run 10 episodes
for episode in range(10):
    obs = env.reset()
    episode_data = {'w': [], 'action': []}
    
    for step in range(200):
        w = obs[0]
        action = controller.compute_action({'w': w})
        obs, _, done, _ = env.step(action)
        
        episode_data['w'].append(w)
        episode_data['action'].append(action)
    
    # Save episode_data
```

**PID control:** (同样流程,用PID controller)

**Collect metrics:**
- Convergence time
- Final error
- Overshoot
- Control effort

---

### Day 3-4: RL vs Baseline对比

**4个指标计算:**

```python
def compute_metrics(episode_data):
    w_trajectory = np.array(episode_data['w'])
    action_trajectory = np.array(episode_data['action'])
    
    # Metric 1: Convergence time
    threshold = 0.005
    converged = w_trajectory < threshold
    if np.any(converged):
        convergence_time = np.argmax(converged)
    else:
        convergence_time = len(w_trajectory)
    
    # Metric 2: Final error
    final_error = np.mean(w_trajectory[-50:])
    
    # Metric 3: Overshoot
    w_initial = w_trajectory[0]
    overshoot = (np.max(w_trajectory) - w_initial) / w_initial * 100
    
    # Metric 4: Control effort
    effort = np.sum(np.abs(action_trajectory))
    
    return {
        'convergence_time': convergence_time,
        'final_error': final_error,
        'overshoot': overshoot,
        'effort': effort
    }
```

**Statistical analysis:**
- 每个controller运行10次
- 计算mean ± std
- t-test显著性

**Visualization:**
```python
import matplotlib.pyplot as plt

# Trajectory comparison
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

axes[0].plot(w_p, label='P control')
axes[0].plot(w_pid, label='PID control')
axes[0].plot(w_rl, label='RL control')
axes[0].set_ylabel('Island width')
axes[0].legend()

axes[1].plot(action_p, label='P')
axes[1].plot(action_pid, label='PID')
axes[1].plot(action_rl, label='RL')
axes[1].set_ylabel('RMP amplitude')
axes[1].legend()

plt.savefig('figures/comparison.png')
```

---

### Day 5: Physics Validation (小P)

**小P review学到的策略:**

**1. Action时序合理性**
- 不应该高频抖动 (vs PID)
- 应该smooth调整

**2. 物理直觉利用**
- 岛大时用大RMP?
- 收敛后减小control?

**3. Conservation检查**
- Energy drift acceptable?

**可解释性分析:**
```python
# Policy sensitivity
obs_base = env.reset()
w_values = np.linspace(0, 0.1, 20)

actions = []
for w in w_values:
    obs_test = obs_base.copy()
    obs_test[0] = w  # Vary w
    action, _ = model.predict(obs_test, deterministic=True)
    actions.append(action)

plt.plot(w_values, actions)
plt.xlabel('Island width')
plt.ylabel('RL action')
plt.title('Policy response to w')
```

**小P签字: 策略物理合理性approved** ✅

---

## Week 4: Optimization & Analysis

### Day 1-2: SAC Alternative (If Needed)

**只有当PPO不超越PID时才执行**

```python
from stable_baselines3 import SAC

model_sac = SAC(
    'MlpPolicy',
    env,
    learning_rate=3e-4,
    buffer_size=100000,
    gamma=0.98,  # Use physics-motivated
    tau=0.005,
    ent_coef='auto',
    tensorboard_log="./logs/sac/",
)

model_sac.learn(total_timesteps=100000)
model_sac.save("models/sac_100k")
```

**Compare with PPO:**
- 同样4个指标评估
- 选择最好的

---

### Day 3-4: Performance Analysis

**Ablation studies:**
1. Reward components ablation
2. Observation ablation (哪些维度最重要)
3. Network size ablation

**失败cases分析:**
- 哪些初始条件失败?
- 为什么失败?
- 如何改进?

**可解释性:**
- Feature importance
- Policy visualization
- 与物理直觉对照

---

### Day 5: Phase 5 Completion Report

**Report内容:**

**1. Executive Summary**
- 目标达成情况
- Success criteria满足情况

**2. Results**
- 4个指标对比表格
- Statistical significance
- Trajectory plots

**3. Physics Validation**
- 小P签字结果
- 策略可解释性分析

**4. Lessons Learned**
- What worked
- What didn't
- 为什么

**5. Next Steps**
- Phase 6建议 (如果有)
- 可迁移性分析
- Real tokamak考虑

**Deliverables:**
- ✅ `PHASE5_COMPLETION_REPORT.md`
- ✅ Trained models
- ✅ Comparison figures
- ✅ All logs

---

## Physics-RL协作界面

### 小P Checkpoints

**Week 1 End: Environment Design**
- Review: Observation/Action/Reward物理正确性
- 验收: Unit tests通过 + conservation check
- 签字: Physics approved ✅

**Week 3 End: 策略物理合理性**
- Review: 学到的策略是否物理
- 验收: Action时序合理 + conservation满足
- 签字: Physics结果approved ✅

**Week 4 End: 最终结果分析**
- Review: 整体结果物理意义
- 验收: 可解释性分析完成
- 签字: Phase 5 physics complete ✅

### 职责边界

**小P职责 (Physics):**
- ✅ Observation包含必要物理量?
- ✅ Action范围物理合理?
- ✅ Reward不违反物理?
- ✅ 策略物理可解释?

**小A职责 (RL):**
- ✅ RL算法选择 (PPO/SAC)
- ✅ Hyperparameter tuning
- ✅ Reward shaping细节
- ✅ Network architecture

**协作原则:**
- 小P不介入RL调参 ✅
- 小A trust小P physics判断 ✅
- 清晰边界,互相尊重 ✅
- **符合COLLABORATION_PROTOCOL.md** ✅

---

## 风险分析与缓解

### Risk 1: RL不收敛

**Likelihood:** Medium  
**Impact:** High

**Mitigation:**
- Curriculum learning (从易到难)
- 简化reward (只用-w)
- 增大network capacity

**Backup:**
- SAC算法
- 延长训练时间

---

### Risk 2: RL不超越PID

**Likelihood:** Medium  
**Impact:** Medium

**Mitigation:**
- Reward tuning (尝试小P建议)
- Hyperparameter sweep
- Network architecture调整

**Acceptance:**
- 平手也是进展 (RL可解释性更好)
- 记录为future work

---

### Risk 3: 策略不物理

**Likelihood:** Low  
**Impact:** High

**Mitigation:**
- Week 3小P review
- Physics-informed reward
- Conservation constraints

**Fix:**
- 增加position penalty
- 增加equilibrium bonus
- 重新训练

---

## Success Criteria

### Technical

**RL必须在至少2/4指标上超越PID:**

| Metric | P | PID | RL Target | Success? |
|--------|---|-----|-----------|----------|
| Convergence | ~200 | ~150 | <150 | ⏳ |
| Final error | <0.005 | <0.003 | <0.003 | ⏳ |
| Overshoot | ~10% | <20% | <10% | ⏳ |
| Effort | Mod | High | Min | ⏳ |

**Minimum: 2/4 ✅** (Week 3验证)

### Physics

- ✅ 策略物理可解释 (小P签字)
- ✅ Conservation无严重违反
- ✅ Action时序合理

### Project

- ✅ 代码clean可维护
- ✅ 文档完整
- ✅ 可迁移到真实tokamak (理论上)

---

## Deliverables

**Code:**
- ✅ `src/pytokmhd/rl/env.py` (Environment)
- ✅ `src/pytokmhd/rl/__init__.py` (API)
- ✅ `scripts/train_ppo_*.py` (Training scripts)
- ✅ `scripts/evaluate_baseline.py` (Baseline)

**Models:**
- ✅ `models/ppo_final_100k.zip` (Trained policy)
- ✅ `models/ppo_gamma_*.zip` (Tuning experiments)
- ✅ `models/sac_100k.zip` (If needed)

**Data:**
- ✅ `logs/` (TensorBoard logs)
- ✅ `checkpoints/` (Model checkpoints)
- ✅ `results/` (Comparison data)

**Docs:**
- ✅ `PHASE5_COMPLETION_REPORT.md` (Main deliverable)
- ✅ `design/PHASE5_*.md` (All design docs)

**Tests:**
- ✅ `tests/test_rl_env.py` (Unit tests)

---

**Phase 5 Implementation Plan Complete** ✅  
**Physics Checkpoints Defined** ✅  
**Ready for Week 1 Start** ✅

**Contact:**  
- 小A 🤖 - Week 1-4 implementation  
- 小P ⚛️ - Physics checkpoints (Week 1/3/4)  
- ∞ - Coordination

**Created:** 2026-03-16  
**Approved:** YZ pending  
**Status:** READY TO START
