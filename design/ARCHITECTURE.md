# PTM-RL Architecture Design

**Version:** 1.0  
**Created:** 2026-03-16  
**Status:** Draft

---

## 1. 架构总览

### 三层设计

```
┌─────────────────────────────────────┐
│  Layer 3: RL Control Framework      │
│  - Observation: [w, γ, E, ...]      │
│  - Action: RMP current              │
│  - Reward: -w - λa²                 │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  Layer 2: MHD Dynamics Solver       │
│  - Reduced MHD equations            │
│  - Tearing mode evolution           │
│  - RMP forcing integration          │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  Layer 1: PyTokEq Equilibrium       │
│  - Grad-Shafranov solver            │
│  - Real tokamak profiles            │
│  - Physics validation               │
└─────────────────────────────────────┘
```

### 双实现方案

**CPU Version (NumPy + Ray):**
- 10-core parallelization
- ~10× speedup vs single-core
- Accessible on standard workstations

**GPU Version (JAX):**
- JIT-compiled, GPU-accelerated
- ~100× speedup vs CPU
- High-performance training

---

## 2. Layer 1: PyTokEq集成

### 接口定义

**输入配置:**
```python
@dataclass
class EquilibriumConfig:
    R0: float      # Major radius (m)
    a: float       # Minor radius (m)  
    B0: float      # Toroidal field (T)
    Ip: float      # Plasma current (MA)
    kappa: float   # Elongation
    delta: float   # Triangularity
    beta_p: float  # Poloidal beta
```

**输出:**
```python
@dataclass
class Equilibrium:
    psi: np.ndarray        # Poloidal flux (nr, nz)
    pressure: np.ndarray   # Pressure profile
    current: np.ndarray    # Current profile
    q_profile: np.ndarray  # Safety factor
    grid: Grid             # Computational grid
```

**API:**
```python
class EquilibriumProvider:
    def solve(self, config: EquilibriumConfig) -> Equilibrium:
        """Solve Grad-Shafranov equation"""
        
    def validate(self, eq: Equilibrium) -> bool:
        """Validate equilibrium quality"""
```

### 性能优化

**Caching strategy:**
- Pre-compute equilibria for common configs
- Cache by (R0, a, Ip, beta_p) hash
- ~1s solve → ~0.001s lookup

**GPU acceleration (optional):**
- JAX port of PyTokEq
- ~100× speedup for solve

---

## 3. Layer 2: MHD Solver

### 物理方程

**Reduced MHD (2D cylindrical):**
```
∂ψ/∂t = -∇·(v×B) + η∇²ψ + ψ_RMP
∂ω/∂t = -v·∇ω + B·∇j + ν∇²ω
```

### 初始化接口

```python
def initialize_from_equilibrium(
    eq: Equilibrium, 
    mode: tuple = (2, 1),
    amplitude: float = 1e-5
) -> MHDState:
    """
    Initialize MHD state from PyTokEq equilibrium
    
    1. Interpolate equilibrium psi to MHD grid
    2. Compute equilibrium current j_eq
    3. Add (m,n) tearing mode perturbation
    4. Return initial MHDState(psi, omega)
    """
```

### CPU实现 (NumPy + Ray)

```python
class MHDSolverCPU:
    def step(self, state, dt, rmp_current):
        """RK2 time integration"""
        # Compute RHS
        dpsi_dt, domega_dt = self.rhs(state, rmp_current)
        
        # RK2
        state_new = self.rk2_step(state, dpsi_dt, domega_dt, dt)
        return state_new
```

**Ray parallelization:**
```python
@ray.remote
def run_episode(env_config, seed):
    env = PTMRLEnv(env_config)
    # ... run episode
    return rewards

# 10-core parallel
futures = [run_episode.remote(config, i) for i in range(10)]
results = ray.get(futures)
```

### GPU实现 (JAX)

```python
class MHDSolverGPU:
    def __init__(self, ...):
        self.step_jit = jax.jit(self._step)
        
    def step(self, state, dt, rmp):
        return self.step_jit(state, dt, rmp)
```

**Batch parallelization:**
```python
# 128 episodes in parallel on GPU
vmapped_env = jax.vmap(env.step)
batch_next_states = vmapped_env(batch_states, batch_actions)
```

---

## 4. Layer 3: RL Framework

### Environment

```python
class PTMRLEnv(gym.Env):
    observation_space = Box(
        low=-inf, high=inf, 
        shape=(7,),  # [w, γ, E_kin, E_mag, ψ_amp, ω_amp, q_min]
    )
    action_space = Box(
        low=-1, high=1, 
        shape=(1,),  # RMP current
    )
    
    def reset(self, seed):
        # Solve equilibrium
        eq = self.eq_solver.solve(self.eq_config)
        # Initialize MHD
        self.state = initialize_from_equilibrium(eq)
        return self._get_obs()
        
    def step(self, action):
        # Scale action to RMP current
        rmp = self._scale_action(action)
        # Evolve MHD
        self.state = self.mhd_solver.step(self.state, self.dt, rmp)
        # Compute reward
        obs = self._get_obs()
        reward = -self.island_width(self.state) - 0.01 * action**2
        return obs, reward, done, {}
```

### Observation

**7-dimensional:**
1. Island width w
2. Growth rate γ
3. Kinetic energy E_kin
4. Magnetic energy E_mag
5. Flux amplitude ψ_amp
6. Vorticity amplitude ω_amp
7. Minimum q value

### Reward Function

```python
reward = -w - λ * a²
```
- Primary: Minimize island width
- Secondary: Minimize control effort

### Training

**CPU (Stable-Baselines3):**
```python
from stable_baselines3 import PPO

model = PPO("MlpPolicy", env, n_steps=2048)
model.learn(total_timesteps=100_000)
```

**GPU (Custom JAX PPO):**
```python
# Batched training on GPU
for epoch in range(epochs):
    batch_traj = collect_batch_gpu(policy, env, 128)
    loss, grads = ppo_loss(policy.params, batch_traj)
    policy.params = update(policy.params, grads)
```

---

## 5. 数据流图

```
EquilibriumConfig
    ↓
PyTokEq.solve()
    ↓ Equilibrium
initialize_from_equilibrium()
    ↓ MHDState(t=0)
    ┌─────────────────┐
    │  Training Loop  │
    │                 │
    │  obs ← extract  │
    │  action ← π(obs)│
    │  state' ← step  │
    │  reward ← -w    │
    │  π ← update     │
    └─────────────────┘
```

---

## 6. 技术栈对比

| Component | CPU | GPU |
|-----------|-----|-----|
| Layer 1 | NumPy (~1s) | JAX (~0.01s) |
| Layer 2 | NumPy + RK2 | JAX JIT |
| Parallel | Ray (10×) | GPU batch (100×) |
| Layer 3 | SB3 PPO | Custom JAX PPO |
| Speed | 10× baseline | 1000× baseline |

---

## 7. 验收标准

**Architecture完成:**
- [x] 接口定义明确
- [x] 数据流清晰
- [x] 技术栈选择有依据
- [x] CPU/GPU方案完整

**Team review:**
- [ ] 小P: Physics correctness
- [ ] 小A: ML/RL feasibility  
- [ ] YZ: Strategic approval

---

**下一步:** 小P/小A review → 开始Layer 1实现
