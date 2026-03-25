#!/usr/bin/env python3
"""
Diagnose why tearing mode decays instead of grows.

Author: 小P ⚛️
Date: 2026-03-25
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import jax.numpy as jnp
from pim_rl.physics.v2.complete_solver_v2 import CompleteMHDSolver, _rhs_jit
from pim_rl.physics.v2.elsasser_bracket import ElsasserState
from pim_rl.physics.v2.tearing_ic import create_tearing_ic

print("="*60)
print("Diagnosing Tearing Mode Physics")
print("="*60)

# Setup
nr, ntheta, nz = 32, 64, 8
dr, dtheta, dz = 0.03125, 2*np.pi/ntheta, 0.1

solver = CompleteMHDSolver(
    grid_shape=(nr, ntheta, nz),
    dr=dr, dtheta=dtheta, dz=dz,
    epsilon=0.3, eta=0.05, pressure_scale=0.2
)

# Tearing mode IC
psi_2d, phi_2d = create_tearing_ic(nr, ntheta, lam=0.1, m=2, eps=0.05, eta=0.05)
psi = np.repeat(psi_2d[:, :, np.newaxis], nz, axis=2)
phi = np.repeat(phi_2d[:, :, np.newaxis], nz, axis=2)

state = ElsasserState(
    z_plus=jnp.array(psi + phi, dtype=jnp.float32),
    z_minus=jnp.array(psi - phi, dtype=jnp.float32),
    P=jnp.zeros((nr, ntheta, nz), dtype=jnp.float32)
)

print("\nInitial State:")
print(f"  z+ range: [{state.z_plus.min():.3f}, {state.z_plus.max():.3f}]")
print(f"  z- range: [{state.z_minus.min():.3f}, {state.z_minus.max():.3f}]")

# Extract m=2 mode amplitude
def get_m2_amplitude(s):
    fft_theta = jnp.fft.fft(s.z_plus, axis=1)
    return float(jnp.abs(fft_theta[:, 2, :]).max())

# Track energy and amplitude
dt = 1e-4
n_steps = 200

energies = []
amplitudes = []
times = []

for step in range(n_steps):
    # Compute RHS
    dstate = _rhs_jit(state, solver.grid, solver.epsilon, solver.eta, solver.pressure_scale)
    
    # Check RHS magnitude
    if step == 0:
        print(f"\nFirst RHS:")
        print(f"  dz+/dt range: [{dstate.z_plus.min():.6f}, {dstate.z_plus.max():.6f}]")
        print(f"  dz-/dt range: [{dstate.z_minus.min():.6f}, {dstate.z_minus.max():.6f}]")
        print(f"  dP/dt range: [{dstate.P.min():.6f}, {dstate.P.max():.6f}]")
    
    # Euler step
    state = ElsasserState(
        z_plus=state.z_plus + dt * dstate.z_plus,
        z_minus=state.z_minus + dt * dstate.z_minus,
        P=state.P + dt * dstate.P
    )
    
    # Track
    if step % 20 == 0:
        E = float(solver.hamiltonian(state))
        amp = get_m2_amplitude(state)
        energies.append(E)
        amplitudes.append(amp)
        times.append(step * dt)
        
        if step % 100 == 0:
            print(f"\nStep {step}:")
            print(f"  Energy: {E:.6f}")
            print(f"  m=2 amplitude: {amp:.6f}")

print("\n" + "="*60)
print("Energy Evolution:")
print(f"  Initial: {energies[0]:.6f}")
print(f"  Final: {energies[-1]:.6f}")
print(f"  ΔE: {energies[-1] - energies[0]:.6e}")
print(f"  ΔE/E₀: {(energies[-1] - energies[0])/energies[0]:.6e}")

print("\nAmplitude Evolution:")
print(f"  Initial: {amplitudes[0]:.6f}")
print(f"  Final: {amplitudes[-1]:.6f}")
print(f"  Growth factor: {amplitudes[-1]/amplitudes[0]:.3f}")

# Fit growth rate
if amplitudes[-1] > amplitudes[0]:
    gamma = np.log(amplitudes[-1]/amplitudes[0]) / times[-1]
    print(f"  Estimated γ: {gamma:.2f} (growth)")
else:
    gamma = -np.log(amplitudes[0]/amplitudes[-1]) / times[-1]
    print(f"  Estimated γ: {gamma:.2f} (decay)")

print("\n" + "="*60)
print("Diagnosis:")

# Check if energy is conserved or dissipated correctly
dE_relative = abs((energies[-1] - energies[0])/energies[0])
if dE_relative < 0.001:
    print("✅ Energy evolution: Good (small change)")
elif energies[-1] < energies[0]:
    print("✅ Energy evolution: Dissipating (expected for resistive MHD)")
else:
    print("⚠️  Energy evolution: Growing (unexpected)")

# Check amplitude
if amplitudes[-1] > amplitudes[0]:
    print("✅ Tearing mode: Growing (expected)")
else:
    print("❌ Tearing mode: Decaying (UNEXPECTED for tearing with eta=0.05)")
    print("\nPossible causes:")
    print("  1. Resistivity too high (damping > drive)")
    print("  2. IC perturbation not in unstable eigenmode")
    print("  3. Pressure term stabilizing (shouldn't be for ballooning)")
    print("  4. Numerical diffusion too strong")
    
    print("\nRecommendation:")
    print("  - Try lower eta (0.01 instead of 0.05)")
    print("  - Check if IC matches theoretical eigenmode")
    print("  - Run with eta=0 (ideal MHD) to see if grows")
