"""
Test 1.2: Tearing Mode Growth Rate

Compare v2.0 measured growth rate with FKR theory
Author: 小P ⚛️
"""
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../../src')

from pim_rl.physics.v2.complete_solver import CompleteMHDSolver
from pim_rl.physics.v2.elsasser_bracket import ElsasserState


def harris_sheet_ic(grid_shape=(64, 64, 32), width=0.2, amp=0.01):
    """Harris current sheet initial condition
    
    B_z(r) = B0 * tanh(r/a)
    Small perturbation δB_x ~ sin(mθ)
    """
    Nr, Ntheta, Nz = grid_shape
    
    r_grid = np.linspace(0.1, 0.9, Nr)[:, None, None]
    theta_grid = np.linspace(0, 2*np.pi, Ntheta)[None, :, None]
    
    # Background field (Harris sheet)
    B_z = np.tanh(r_grid / width)
    
    # Perturbation (m=1 tearing mode)
    m = 1
    delta_Bx = amp * np.sin(m * theta_grid) * np.exp(-((r_grid - 0.5)/0.2)**2)
    
    # Elsasser (simplified)
    z_plus = jnp.array((B_z + delta_Bx) * np.ones((Nr, Ntheta, Nz)))
    z_minus = jnp.array((B_z - delta_Bx) * np.ones((Nr, Ntheta, Nz)))
    P = jnp.zeros((Nr, Ntheta, Nz))
    
    return ElsasserState(z_plus=z_plus, z_minus=z_minus, P=P)


def measure_growth_rate(times, amplitudes):
    """Fit exponential growth: A(t) = A0 * exp(γt)"""
    # Linear fit in log space
    log_A = np.log(amplitudes + 1e-10)
    gamma = np.polyfit(times, log_A, 1)[0]
    return gamma


def fkr_growth_rate(eta, k, a):
    """FKR theory growth rate (simplified)
    
    γ ~ (η * k^2)^{1/3}
    """
    gamma_theory = (eta * k**2)**(1/3)
    return gamma_theory


def run_test():
    print("\n" + "="*60)
    print("Test 1.2: Tearing Mode Growth Rate")
    print("="*60)
    
    # Setup
    grid_shape = (64, 64, 32)
    eta = 0.01
    solver = CompleteMHDSolver(
        grid_shape=grid_shape,
        dr=0.02, dtheta=2*np.pi/64, dz=2*np.pi/32,
        epsilon=0.3,
        eta=eta,
        pressure_scale=0.0
    )
    
    # Initial condition
    state = harris_sheet_ic(grid_shape, width=0.2, amp=0.01)
    
    # Simulate
    print("\nRunning simulation...")
    times = []
    amplitudes = []
    
    dt = 0.01
    n_steps = 200
    
    for i in range(n_steps):
        # Extract amplitude (m=1 mode)
        B_pert = (state.z_plus - state.z_minus) / 2
        profile_theta = np.mean(np.array(B_pert), axis=(0,2))
        amp = np.max(np.abs(profile_theta - np.mean(profile_theta)))
        
        times.append(i * dt)
        amplitudes.append(amp)
        
        # Step
        state = solver.step_rk2(state, dt)
        
        if (i+1) % 40 == 0:
            print(f"  Step {i+1}/{n_steps}, amplitude: {amp:.6f}")
    
    times = np.array(times)
    amplitudes = np.array(amplitudes)
    
    # Measure growth rate (fit linear phase)
    idx_start = 20
    idx_end = 150
    gamma_measured = measure_growth_rate(
        times[idx_start:idx_end], 
        amplitudes[idx_start:idx_end]
    )
    
    # Theory (simplified FKR)
    k = 1.0  # Mode wavenumber
    a = 0.2  # Sheet width
    gamma_theory = fkr_growth_rate(eta, k, a)
    
    error = abs(gamma_measured - gamma_theory) / gamma_theory * 100
    
    # Results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"η: {eta}")
    print(f"γ (theory):   {gamma_theory:.4f}")
    print(f"γ (measured): {gamma_measured:.4f}")
    print(f"Error:        {error:.1f}%")
    print("="*60)
    
    # Pass/fail
    if error < 15:
        print("✅ PASS (error < 15%)")
        passed = True
    else:
        print(f"❌ FAIL (error {error:.1f}% > 15%)")
        passed = False
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Growth curve
    ax1.semilogy(times, amplitudes, 'b-', label='Measured')
    fit = amplitudes[idx_start] * np.exp(gamma_measured * (times - times[idx_start]))
    ax1.semilogy(times[idx_start:idx_end], fit[idx_start:idx_end], 'r--', 
                 label=f'Fit γ={gamma_measured:.4f}')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Tearing Mode Growth')
    ax1.legend()
    ax1.grid(True)
    
    # Growth rate comparison
    ax2.bar(['Theory', 'Measured'], [gamma_theory, gamma_measured], color=['orange', 'blue'])
    ax2.set_ylabel('Growth rate γ')
    ax2.set_title(f'Comparison (error: {error:.1f}%)')
    ax2.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('tearing_mode_results.png', dpi=150)
    print("\nPlot saved: tearing_mode_results.png")
    plt.close()
    
    return {
        'passed': passed,
        'gamma_theory': gamma_theory,
        'gamma_measured': gamma_measured,
        'error_pct': error
    }


if __name__ == '__main__':
    results = run_test()
    print("\nTest 1.2 complete ⚛️")
