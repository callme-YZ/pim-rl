# Standard Tokamak Benchmark Equilibria

**Issue #13 deliverables**

Provides three standard tokamak reference cases for PyTokEq validation.

## Available Benchmarks

### 1. ITER Baseline Scenario

**File:** `iter_baseline.py`

**Reference:** ITER Physics Basis, Nucl. Fusion 47 (2007)

**Parameters:**
- R₀ = 6.2 m, a = 2.0 m (A = 3.1)
- B₀ = 5.3 T, Ip = 15 MA
- κ = 1.7, δ = 0.33
- β_N = 1.8, q₉₅ = 3.0, q₀ = 1.0

**Purpose:** Standard fusion reference, high-performance scenario

---

### 2. DIII-D H-mode Reference

**File:** `diiid_hmode.py`

**Reference:** Typical H-mode (Lao et al., Nucl. Fusion 30, 1990)

**Parameters:**
- R₀ = 1.67 m, a = 0.67 m (A = 2.5)
- B₀ = 2.0 T, Ip = 1.2 MA
- κ = 1.8, δ = 0.4
- β_N = 2.5, q₉₅ = 4.0, q₀ = 1.1

**Purpose:** Well-documented experimental benchmark, H-mode physics

---

### 3. EAST Long-Pulse Reference

**File:** `east_reference.py`

**Reference:** Typical scenario (Wan et al., Nucl. Fusion 55, 2015)

**Parameters:**
- R₀ = 1.85 m, a = 0.45 m (A = 4.1)
- B₀ = 2.0 T, Ip = 0.5 MA
- κ = 1.6, δ = 0.3
- β_N = 1.5, q₉₅ = 5.0, q₀ = 1.2

**Purpose:** Long-pulse operation, high aspect ratio

---

## Usage

### Basic Usage

```python
from pytokeq.benchmarks import iter_baseline_equilibrium

# Get ITER benchmark
iter_eq = iter_baseline_equilibrium()

# Access parameters
params = iter_eq['params']
print(f"ITER R0 = {params['R0']} m")

# Use profiles
import jax.numpy as jnp
psi_norm = jnp.linspace(0, 1, 100)
q_profile = iter_eq['profiles']['q'](psi_norm)
```

### All Benchmarks

```python
from pytokeq.benchmarks import (
    iter_baseline_equilibrium,
    diiid_hmode_equilibrium,
    east_reference_equilibrium
)

benchmarks = {
    'ITER': iter_baseline_equilibrium(),
    'DIII-D': diiid_hmode_equilibrium(),
    'EAST': east_reference_equilibrium(),
}

for name, eq in benchmarks.items():
    print(f"{name}: {eq['name']}")
```

### Custom Grid

```python
iter_eq = iter_baseline_equilibrium(
    nr=129, nz=129,  # Higher resolution
    rmin=3.0, rmax=9.0,  # Custom domain
)
```

---

## Validation

Each benchmark includes validation functions:

```python
from pytokeq.benchmarks.iter_baseline import validate_iter_equilibrium

# After solving equilibrium
eq_solution = your_solver.solve(iter_eq)

# Validate against reference
metrics = validate_iter_equilibrium(eq_solution)

if metrics['q_95']['pass']:
    print("✅ q₉₅ matches ITER reference")
```

---

## Profiles

Each benchmark provides callable profile functions:

**Available profiles:**
- `q(psi_norm)` - Safety factor profile
- `pressure(psi_norm)` - Pressure profile
- `current(psi_norm)` - Current density profile (ITER only)

**Example:**

```python
import jax.numpy as jnp
import matplotlib.pyplot as plt

iter_eq = iter_baseline_equilibrium()
psi = jnp.linspace(0, 1, 100)

# Get profiles
q = iter_eq['profiles']['q'](psi)
p = iter_eq['profiles']['pressure'](psi)

# Plot
plt.plot(psi, q, label='q-profile')
plt.xlabel('ψ_norm')
plt.ylabel('q')
plt.legend()
plt.show()
```

---

## Benchmark Comparison

| Parameter | ITER | DIII-D | EAST |
|-----------|------|--------|------|
| R₀ [m] | 6.2 | 1.67 | 1.85 |
| a [m] | 2.0 | 0.67 | 0.45 |
| A | 3.1 | 2.5 | 4.1 |
| Ip [MA] | 15 | 1.2 | 0.5 |
| q₉₅ | 3.0 | 4.0 | 5.0 |
| β_N | 1.8 | 2.5 | 1.5 |
| Focus | Fusion | H-mode | Long-pulse |

---

## Tests

Run benchmark tests:

```bash
cd src/pytokeq/tests
python3 -m pytest test_benchmarks.py -v

# Output:
# ============================== 11 passed in 0.34s ===============================
```

---

## Notes

**Data Sources:**
- ITER: Physics Basis 2007 (official baseline)
- DIII-D: Representative H-mode (Lao et al., literature)
- EAST: Typical long-pulse scenario (Wan et al., literature)

**Limitations:**
- Profiles are representative, not from specific shots
- For exact shot reconstruction, use experimental data files
- Pedestal structure simplified in H-mode cases

**Future Work:**
- Add specific shot reconstructions
- Include experimental error bars
- Add more machines (JET, JT-60SA, etc.)

---

## Author

小P ⚛️

**Date:** 2026-03-24

**Issue:** #13
