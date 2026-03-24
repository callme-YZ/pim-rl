#!/usr/bin/env python3
"""Quick real-time test - skip slow configs"""
import sys
sys.path.insert(0, 'src')
import time, numpy as np
from pytokmhd.rl.hamiltonian_env import make_hamiltonian_mhd_env
from pytokmhd.rl.classical_controllers import make_baseline_agent
from pim_rl.physics.v2.tearing_ic import create_tearing_ic
import jax.numpy as jnp

print("Quick Real-time Test")
print()

env = make_hamiltonian_mhd_env(nr=32, ntheta=64, nz=8, dt=1e-4, max_steps=1000, eta=0.05, nu=1e-4, normalize_obs=False)
psi, phi = create_tearing_ic(nr=32, ntheta=64)
env.mhd_solver.initialize(jnp.array(psi, dtype=jnp.float32), jnp.array(phi, dtype=jnp.float32))
env.current_step = 0
env.obs_computer.reset()
obs = env.obs_computer.compute_observation(jnp.array(psi, dtype=jnp.float32), jnp.array(phi, dtype=jnp.float32))
env._last_obs = obs
env._last_psi = jnp.array(psi, dtype=jnp.float32)
env._last_phi = jnp.array(phi, dtype=jnp.float32)
env.psi = jnp.array(psi, dtype=jnp.float32)
env.phi = jnp.array(phi, dtype=jnp.float32)

agent = make_baseline_agent('pid', env.action_space, Kp=5.0, Ki=0.5, Kd=0.01, target=0.0, dt=1e-4)

# Warm-up
print("Warming up...")
for _ in range(10):
    _ = env.step(agent.act(obs), compute_obs=False)
print("Done\n")

# Test key intervals only
intervals = [50, 100, 200, 500]
n_steps = 1000

for interval in intervals:
    print(f"Testing interval={interval}...", end=" ", flush=True)
    env.mhd_solver.initialize(jnp.array(psi, dtype=jnp.float32), jnp.array(phi, dtype=jnp.float32))
    env.current_step = 0
    env._last_obs = obs
    
    start = time.time()
    for step in range(n_steps):
        obs_new, r, term, trunc, info = env.step(agent.act(env._last_obs), compute_obs=(step % interval == 0))
    elapsed = time.time() - start
    
    freq = n_steps / elapsed
    print(f"{freq:.1f} Hz")

print()
print("Recommendation:")
for interval in intervals:
    env.mhd_solver.initialize(jnp.array(psi, dtype=jnp.float32), jnp.array(phi, dtype=jnp.float32))
    env.current_step = 0
    env._last_obs = obs
    start = time.time()
    for step in range(n_steps):
        obs_new, r, term, trunc, info = env.step(agent.act(env._last_obs), compute_obs=(step % interval == 0))
    freq = n_steps / (time.time() - start)
    if freq >= 100:
        print(f"✅ Use interval={interval} → {freq:.1f} Hz (>100 Hz target)")
        break
else:
    print("⚠️ Cannot reach 100 Hz with current setup")
