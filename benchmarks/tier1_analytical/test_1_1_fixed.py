"""
Test 1.1: Alfvén Wave - Fixed IC

Correct Elsasser formulation for Alfvén wave
"""
import numpy as np
import jax.numpy as jnp
import sys
sys.path.insert(0, '../../src')

from pim_rl.physics.v2.complete_solver import CompleteMHDSolver
from pim_rl.physics.v2.elsasser_bracket import ElsasserState


def alfven_ic_correct(grid_shape=(32,64,32), kz=1.0, amp=0.01):
    """Alfvén wave IC (correct Elsasser)
    
    Theory:
    - Background: B₀ẑ, v=0
    - Wave: δv_x = A sin(kz), δB_x = A sin(kz) (equal for Alfvén)
    - Elsasser: z⁺ = (v+B), z⁻ = (v-B)
    """
    Nr, Ntheta, Nz = grid_shape
    
    # Grid
    r_idx = np.arange(Nr)[:, None, None]
    theta_idx = np.arange(Ntheta)[None, :, None]
    z_idx = np.arange(Nz)[None, None, :]
    
    z_coord = z_idx * (2*np.pi/Nz)
    
    # Background (all in z direction)
    B0_z = 1.0
    
    # Perturbation (in x direction, for simplicity use first component)
    delta = amp * np.sin(kz * z_coord)
    
    # Initialize 3-component vectors (r,θ,z)
    # Background: v=(0,0,0), B=(0,0,B₀)
    # Pert: δv=(δ,0,0), δB=(δ,0,0)
    
    # For simplicity in Elsasser vars (assuming aligned):
    # z⁺ = v + B = (0,0,0) + (0,0,B₀) + (δ,0,0) + (δ,0,0) = (2δ, 0, B₀)
    # z⁻ = v - B = (0,0,0) - (0,0,B₀) + (δ,0,0) - (δ,0,0) = (0, 0, -B₀)
    
    # Actually need full 3D treatment
    # Simplify: perturbation in radial direction
    z_plus = jnp.zeros((Nr, Ntheta, Nz))
    z_minus = jnp.zeros((Nr, Ntheta, Nz))
    
    # Radial component: 2*delta (v+B perturbation)
    z_plus = z_plus.at[:,:,:].set(B0_z + 2*delta)
    # For Alfvén: z⁻ background only
    z_minus = z_minus.at[:,:,:].set(-B0_z)
    
    P = jnp.zeros_like(z_plus)
    
    return ElsasserState(z_plus=z_plus, z_minus=z_minus, P=P)


def run_test():
    grid_shape = (16, 32, 64)  # Increase z resolution
    solver = CompleteMHDSolver(
        grid_shape=grid_shape,
        dr=0.1, dtheta=2*np.pi/32, dz=2*np.pi/64,
        epsilon=0.3, eta=0.0, pressure_scale=0.0
    )
    
    state = alfven_ic_correct(grid_shape, kz=2.0, amp=0.05)
    
    states = [state]
    for _ in range(50):
        state = solver.step_rk2(state, dt=0.02)
        states.append(state)
    
    # Check if wave moves
    z_grid = np.linspace(0, 2*np.pi, grid_shape[2])
    
    profile_0 = np.mean(np.array(states[0].z_plus), axis=(0,1))
    profile_end = np.mean(np.array(states[-1].z_plus), axis=(0,1))
    
    # Cross-correlation to find shift
    corr = np.correlate(profile_end - np.mean(profile_end), 
                       profile_0 - np.mean(profile_0), mode='same')
    shift_idx = np.argmax(corr) - len(corr)//2
    shift = shift_idx * (2*np.pi/grid_shape[2])
    
    v_measured = shift / (50 * 0.02)
    
    print(f"Shift: {shift:.4f} rad")
    print(f"v_phase: {v_measured:.4f} (theory: 1.0)")
    print(f"Error: {abs(v_measured-1.0)/1.0*100:.1f}%")
    
    if abs(v_measured - 1.0) < 0.2:
        print("✅ PASS (rough)")
    else:
        print("❌ Still wrong")

run_test()
