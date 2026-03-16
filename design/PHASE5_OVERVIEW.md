# Phase 5: RL Interface - Overview

**Author:** 小A 🤖 (RL Lead)  
**Date:** 2026-03-16  
**Status:** Design Complete, Physics Approved  
**Version:** 1.0

---

## 1. Phase 5在PTM-RL中的定位

### 项目整体架构
```
PTM-RL (Plasma Tearing Mode RL Framework)
│
├── Layer 1: PyTokEq ✅
│   └── Real tokamak equilibrium
│
├── Layer 2: PyTokMHD (Phase 1-4) ✅
│   ├── Phase 1: Core MHD Solver ✅
│   ├── Phase 2: PyTokEq Integration ✅
│   ├── Phase 3: Diagnostics ✅
│   └── Phase 4: RMP Control ✅
│
├── Layer 3: Phase 5 (RL Interface) ← 当前
│   └── RL-based tearing mode control
│
└── 最终目标: 可迁移到真实托卡马克 (EAST/ITER)
```

### Phase 5目标

**Primary Goal:**
训练RL policy控制tearing mode,性能超越传统PID控制。

**Success Criteria:**
- RL在至少2/4指标上超越PID ✅
- 策略物理可解释 ✅
- 代码可迁移到真实设备 ✅

---

## 2. 核心组件概览

### 2.1 Gym Environment (18D Observation)

**Observation Space:**
```python
obs = {
    # Phase 3 diagnostics (4D) - 核心控制目标
    'w': island_width,          # 主要控制目标
    'gamma': growth_rate,       # 增长趋势
    'x_o', 'z_o': island_center,# 岛位置
    
    # MHD state (16D) - 状态监控
    'psi_samples': [8点],       # 磁通分布
    'omega_samples': [8点],     # 涡度分布
    
    # Conservation (3D) - 物理正确性
    'energy', 'mag_helicity', 'energy_drift',
    
    # Context (3D) - RL需要
    'prev_action', 't', 'dt_since_reset'
}
```

**小P Physics Review: ✅ APPROVED**

### 2.2 Action Space

**Primary: Continuous**
- Range: [-0.1, 0.1] (RMP amplitude)
- Physically verified in Phase 4 ✅

**Alternative: Discrete**
- 5 levels: {-0.1, -0.05, 0, 0.05, 0.1}
- Easier baseline comparison

### 2.3 Reward Function

**Baseline (Week 1):**
```python
reward = -w - 0.1*|gamma| - 0.01*|action| + convergence_bonus
```

**Optional (Week 2+, 小P建议):**
```python
reward += position_penalty + equilibrium_bonus
```

### 2.4 RL Algorithm

**Primary: PPO**
- Sample efficient (MHD慢,样本贵)
- Stable training
- Proven in continuous control

**Alternative: SAC**
- Better asymptotic performance
- Use if PPO不够好

**Hyperparameters:**
```python
{
    'learning_rate': 3e-4,
    'batch_size': 256,
    'gamma': 0.95,  # Week 2 tune [0.95, 0.98, 0.99]
    'n_epochs': 10
}
```

---

## 3. Success Criteria

### Technical Metrics

| Metric | P | PID | RL Target | Priority |
|--------|---|-----|-----------|----------|
| Convergence | ~200步 | ~150步 | <150步 | High |
| Final error | <0.005 | <0.003 | <0.003 | High |
| Overshoot | ~10% | <20% | <10% | Med |
| Control effort | Mod | High | Min | Med |

**Success: RL在至少2/4指标上超越PID** ✅

### Physics Validation

**小P Checkpoints:**
- Week 1: Environment design ✅
- Week 3: 策略物理合理性 ✅
- Week 4: 最终结果分析 ✅

---

## 4. Implementation Timeline

**Week 1: Environment (2-3天)**
- Gym wrapper实现
- Unit tests + random policy
- 小P physics checkpoint

**Week 2: Training (5天)**
- PPO baseline (gamma=0.95)
- Gamma tuning [0.95, 0.98, 0.99]
- 100k training run

**Week 3: Baseline (5天)**
- P/PID对比运行
- 4指标评估
- 小P physics validation

**Week 4: Optimization (5天)**
- SAC alternative (if needed)
- Performance analysis
- Phase 5 completion report

**Total: ~4周**

---

## 5. 文档结构

```
design/
├── PHASE5_OVERVIEW.md (本文档)
├── PHASE5_ENVIRONMENT_DESIGN.md (详细设计)
├── PHASE5_RL_ALGORITHM.md (算法与超参)
├── PHASE5_BASELINE_COMPARISON.md (对比策略)
└── PHASE5_IMPLEMENTATION_PLAN.md (实施计划)
```

---

## 6. Quick Start

### Week 1 Day 1: 第一步

**1. 创建environment文件:**
```bash
cd /Users/yz/.openclaw/workspace-xiaoa/ptm-rl
mkdir -p src/pytokmhd/rl
touch src/pytokmhd/rl/env.py
```

**2. 实现MHDTearingControlEnv:**
```python
import gym
from pytokmhd.solver import rk4_step_with_rmp
from pytokmhd.diagnostics import TearingModeMonitor

class MHDTearingControlEnv(gym.Env):
    def __init__(self, Nr=64, Nz=128, dt=0.01, ...):
        # 复用Phase 1-4 API
        
    def reset(self):
        # 初始化MHD state
        
    def step(self, action):
        # 1步MHD演化 + diagnostics
        
    def render(self):
        # 可视化 (optional)
```

**3. Unit test:**
```python
env = MHDTearingControlEnv()
obs = env.reset()
for _ in range(10):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
```

**关键依赖:**
- ✅ Phase 1-4 API (已完成)
- ✅ gym, numpy
- ⏳ stable-baselines3 (Week 2安装)

---

## 7. 与Phase 1-4的关系

**Phase 1-4提供的API (小A复用):**
```python
# Phase 1: MHD Solver
from pytokmhd.solver import rk4_step_with_rmp

# Phase 2: PyTokEq Integration
from pytokmhd.solver import create_equilibrium_cache

# Phase 3: Diagnostics
from pytokmhd.diagnostics import TearingModeMonitor

# Phase 4: RMP Control
from pytokmhd.control import RMPController  # Baseline
```

**Phase 5新增:**
```python
# RL Environment
from pytokmhd.rl import MHDTearingControlEnv

# RL Policy (Week 2训练)
from pytokmhd.rl import load_trained_policy
```

---

## 8. Why RL? (vs P/PID)

### 传统控制的局限

**P Control:**
- 简单但性能有限 (convergence ~200步)
- 固定增益,不适应变化

**PID Control:**
- 性能更好 (convergence ~150步)
- 需要手工调参 (K_p, K_i, K_d)
- 难以泛化到不同场景

### RL的优势

**Adaptive:**
- 自动学习最优策略
- 适应不同初始条件

**Data-driven:**
- 不需要手工调参
- 从经验中学习

**Generalizable:**
- 训练在simulation
- 可迁移到真实tokamak (理论上)

**Trade-off:**
- 需要大量训练样本
- 策略可解释性 (需要分析)

---

## 9. Physics-RL协作界面

### 小P职责 (Physics Correctness)

**Review内容:**
- ✅ Observation包含必要物理量?
- ✅ Action范围物理合理?
- ✅ Reward不违反物理?
- ✅ 学到的策略物理可解释?

**Checkpoints:**
- Week 1 End: Environment design
- Week 3 End: 策略物理合理性
- Week 4 End: 最终结果分析

### 小A职责 (RL Decisions)

**独立决策:**
- ✅ RL算法选择 (PPO/SAC/...)
- ✅ Hyperparameter tuning
- ✅ Reward shaping细节
- ✅ Network architecture

**不需要小P帮忙:**
- ❌ RL调参
- ❌ 算法选择
- ❌ Training策略

### 协作原则

- 小P不越界到RL领域 ✅
- 小A trust小P physics判断 ✅
- 清晰边界,互相尊重 ✅
- **符合COLLABORATION_PROTOCOL.md** ✅

---

## 10. 项目目标对齐

### PTM-RL最终目标

**Scientific:**
- 证明RL可控制tearing mode ✅
- 性能超越传统控制 ✅
- 策略物理可解释 ✅

**Engineering:**
- 代码可迁移到EAST/ITER ✅
- Clean codebase ✅
- 完整文档 ✅

### Phase 5贡献

**直接贡献:**
- RL policy for tearing mode control

**间接贡献:**
- 验证PTM-RL框架可行性
- 为Layer 3 (PyTokTearRL) 打基础

---

## 11. Next Steps

**立即可开始 (YZ批准后):**

**Week 1 Day 1:**
1. 创建`src/pytokmhd/rl/env.py`
2. 实现MHDTearingControlEnv (复用Phase 1-4 API)
3. Unit test: random policy rollout

**Week 1 Day 2:**
4. 完善observation normalization
5. 完善reward function
6. Environment sanity checks

**Week 1 Day 5:**
7. 小P physics checkpoint
8. 签字environment design approved

**Then Week 2: Training!** 🚀

---

**Phase 5 Overview Complete** ✅  
**Physics Approved by 小P** ✅  
**Ready for Implementation** ✅

**Contact:**  
- 小A 🤖 (RL Lead) - Implementation  
- 小P ⚛️ (Physics Lead) - Review  
- ∞ (PM) - Coordination

**Created:** 2026-03-16  
**Status:** APPROVED FOR WEEK 1 START
