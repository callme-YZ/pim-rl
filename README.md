# PTM-RL - Plasma Tearing Mode RL Framework

**Status:** 🚧 Development (Day 1)  
**Version:** 0.1.0-alpha  
**Started:** 2026-03-16

---

## Overview

PTM-RL integrates PyTokEq (tokamak equilibrium) with PyTearRL (MHD dynamics) to create a physics-based RL framework for tearing mode control.

### Key Features

- ✅ **Layer 1:** Real tokamak equilibrium (PyTokEq)
- ✅ **Layer 2:** Physics-correct MHD evolution
- ✅ **Layer 3:** RL control training
- 🔄 **Dual Architecture:** CPU (NumPy+Ray) & GPU (JAX)

---

## Architecture

```
Layer 1: PyTokEq Equilibrium Solver
    ↓ (真实平衡态)
Layer 2: MHD Dynamics (Tearing Mode Evolution)
    ↓ (物理正确演化)
Layer 3: RL Control (RMP Suppression)
    ↓ (可迁移控制策略)
Real Tokamak Application
```

---

## Versions

### CPU Version (NumPy + Ray)
- Parallel execution across 10+ cores
- ~10× speedup vs single-core
- Stable and production-ready

### GPU Version (JAX)
- GPU-accelerated computation
- ~100× speedup vs CPU
- High-performance training

---

## Development Status

**Current Phase:** Project Initialization

- [x] Project structure created
- [x] Documentation initialized
- [ ] Technical design complete
- [ ] Layer 1 (PyTokEq) integration
- [ ] Layer 2 (MHD) implementation
- [ ] Layer 3 (RL) training

---

## Team

- **Physics Lead:** 小P ⚛️ (PyTokEq, MHD validation)
- **ML/RL Lead:** 小A 🤖 (RL framework, GPU optimization)
- **PM:** ∞ (Coordination, Git workflow)
- **Decision:** YZ 🐙

---

## Quick Start

*(Coming soon after Phase 1 completion)*

---

## Documentation

- [Project Plan](PROJECT_PTM_RL.md) — Full project specification
- [Status](STATUS.md) — Current development status
- [Design](design/) — Technical design documents

---

## License

*(TBD)*

---

**Created:** 2026-03-16  
**Last Updated:** 2026-03-16
