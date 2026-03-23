"""
Test 1.1: Alfvén Wave Propagation Benchmark

Verify v2.0 MHD solver against analytical Alfvén wave solution.

Theory:
- Alfvén speed: v_A = B₀/√(μ₀ρ)
- Phase velocity: v_phase = v_A
- No damping in ideal MHD

Test:
- Initialize sinusoidal perturbation
- Propagate for multiple wavelengths
- Measure phase velocity
- Check energy conservation

Author: 小P ⚛️
Date: 2026-03-23
"""

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

# Import v2.0 solver
import sys
sys.path.insert(0, '../../src')
from pim_rl.physics.v2.complete_solver import CompleteMHDSolver
from pim_rl.physics.v2.elsasser_bracket import ElsasserState


class AlfvenWaveBenchmark:
    """Alfvén wave analytical benchmark"""
    
    def __init__(self, grid_shape=(32, 64, 32)):
        """Initialize benchmark
        
        Args:
            grid_shape: (Nr, Nθ, Nz) resolution
        """
        self.Nr, self.Ntheta, self.Nz = grid_shape
        
        # Grid spacing
        self.dr = 0.05
        self.dtheta = 2*np.pi / self.Ntheta
        self.dz = 2*np.pi / self.Nz
        
        # Physical parameters (ideal MHD for Alfvén wave)
        self.epsilon = 0.3
        self.eta = 0.0  # No resistivity for pure Alfvén wave
        self.pressure_scale = 0.0  # No pressure gradient
        
        # Background field
        self.B0 = 1.0  # Tesla
        self.rho0 = 1.0  # kg/m³
        self.mu0 = 4*np.pi*1e-7
        
        # Alfvén speed (analytical)
        self.v_A_theory = self.B0 / np.sqrt(self.mu0 * self.rho0)
        
        print(f"Alfvén Wave Benchmark initialized:")
        print(f"  Grid: {grid_shape}")
        print(f"  B₀: {self.B0} T")
        print(f"  ρ₀: {self.rho0} kg/m³")
        print(f"  v_A (theory): {self.v_A_theory:.2f} m/s")
        
        # Initialize solver
        self.solver = CompleteMHDSolver(
            grid_shape=grid_shape,
            dr=self.dr,
            dtheta=self.dtheta,
            dz=self.dz,
            epsilon=self.epsilon,
            eta=self.eta,
            pressure_scale=self.pressure_scale
        )
    
    def create_initial_condition(self, kz=1.0, amplitude=0.01):
        """Create sinusoidal Alfvén wave initial condition
        
        Args:
            kz: Wavenumber in z direction
            amplitude: Perturbation amplitude (small)
            
        Returns:
            ElsasserState with sinusoidal perturbation
        """
        # Grid
        r = np.linspace(0, self.Nr*self.dr, self.Nr)[:, None, None]
        theta = np.linspace(0, 2*np.pi, self.Ntheta)[None, :, None]
        z = np.linspace(0, 2*np.pi, self.Nz)[None, None, :]
        
        # Background equilibrium (uniform B field in z direction)
        B_background = self.B0 * np.ones_like(z)
        v_background = np.zeros_like(z)
        
        # Perturbation: Alfvén wave in z direction
        # δB_x = A sin(kz·z)
        # δv_x = ±A sin(kz·z)  (same sign for Alfvén wave)
        delta_B_x = amplitude * np.sin(kz * z)
        delta_v_x = amplitude * np.sin(kz * z) * self.v_A_theory / self.B0
        
        # Elsasser variables: z± = v ± B
        z_plus = jnp.array(v_background + delta_v_x + B_background + delta_B_x)
        z_minus = jnp.array(v_background + delta_v_x - (B_background + delta_B_x))
        
        # Pressure (constant)
        P = jnp.zeros_like(z_plus)
        
        state = ElsasserState(z_plus=z_plus, z_minus=z_minus, P=P)
        
        print(f"Initial condition created:")
        print(f"  Wavenumber k_z: {kz}")
        print(f"  Wavelength λ: {2*np.pi/kz:.2f}")
        print(f"  Amplitude: {amplitude}")
        
        return state
    
    def run_simulation(self, state, dt=0.01, n_steps=100):
        """Run v2.0 simulation
        
        Args:
            state: Initial state
            dt: Time step
            n_steps: Number of steps
            
        Returns:
            states: List of states
            times: Time array
        """
        states = [state]
        times = [0.0]
        
        for i in range(n_steps):
            state = self.solver.step_rk2(state, dt)
            states.append(state)
            times.append((i+1)*dt)
            
            if (i+1) % 20 == 0:
                print(f"  Step {i+1}/{n_steps}, t={times[-1]:.2f}")
        
        return states, np.array(times)
    
    def extract_wave_position(self, state):
        """Extract wave peak position from state
        
        Args:
            state: ElsasserState
            
        Returns:
            z_peak: Position of wave maximum
        """
        # Extract perturbation (difference from mean)
        z_plus_pert = state.z_plus - jnp.mean(state.z_plus)
        
        # Find peak in z direction (averaged over r, θ)
        z_profile = jnp.mean(z_plus_pert, axis=(0, 1))
        z_peak_idx = jnp.argmax(jnp.abs(z_profile))
        
        z_grid = np.linspace(0, 2*np.pi, self.Nz)
        z_peak = z_grid[z_peak_idx]
        
        return z_peak
    
    def measure_phase_velocity(self, states, times, kz=1.0):
        """Measure phase velocity from simulation
        
        Args:
            states: List of states
            times: Time array
            kz: Wavenumber
            
        Returns:
            v_phase_measured: Measured phase velocity
        """
        # Track wave peak position
        z_peaks = [self.extract_wave_position(s) for s in states]
        
        # Linear fit: z_peak = v_phase * t + z0
        # (Handle periodic boundary)
        z_peaks = np.array(z_peaks)
        z_unwrapped = np.unwrap(z_peaks)
        
        # Fit
        coeffs = np.polyfit(times, z_unwrapped, 1)
        v_phase_measured = coeffs[0]
        
        return v_phase_measured, z_peaks
    
    def measure_energy_conservation(self, states):
        """Measure energy conservation
        
        Args:
            states: List of states
            
        Returns:
            energies: Energy at each time
            drift: Relative energy drift
        """
        energies = [self.solver.hamiltonian(s) for s in states]
        energies = np.array(energies)
        
        E0 = energies[0]
        drift = (energies[-1] - E0) / E0 * 100
        
        return energies, drift
    
    def run_benchmark(self, kz=1.0, dt=0.01, n_steps=100):
        """Run complete benchmark
        
        Returns:
            results: Dictionary with all results
        """
        print("\n" + "="*60)
        print("Test 1.1: Alfvén Wave Benchmark")
        print("="*60)
        
        # Initial condition
        state0 = self.create_initial_condition(kz=kz)
        
        # Run simulation
        print("\nRunning v2.0 simulation...")
        states, times = self.run_simulation(state0, dt=dt, n_steps=n_steps)
        
        # Measure phase velocity
        print("\nMeasuring phase velocity...")
        v_phase_measured, z_peaks = self.measure_phase_velocity(states, times, kz)
        
        # Measure energy conservation
        print("Measuring energy conservation...")
        energies, drift = self.measure_energy_conservation(states)
        
        # Results
        error = abs(v_phase_measured - self.v_A_theory) / self.v_A_theory * 100
        
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"Alfvén speed (theory):    {self.v_A_theory:.4f} m/s")
        print(f"Phase velocity (measured): {v_phase_measured:.4f} m/s")
        print(f"Relative error:            {error:.2f}%")
        print(f"Energy drift:              {drift:.4f}%")
        print("="*60)
        
        # Pass/fail
        pass_velocity = error < 1.0  # <1% error
        pass_energy = abs(drift) < 0.1  # <0.1% drift
        
        if pass_velocity and pass_energy:
            print("✅ PASS: Test 1.1 Alfvén Wave")
        else:
            print("❌ FAIL: Test 1.1 Alfvén Wave")
            if not pass_velocity:
                print(f"   - Velocity error {error:.2f}% > 1%")
            if not pass_energy:
                print(f"   - Energy drift {drift:.4f}% > 0.1%")
        
        results = {
            'v_A_theory': self.v_A_theory,
            'v_phase_measured': v_phase_measured,
            'error_percent': error,
            'energy_drift_percent': drift,
            'pass': pass_velocity and pass_energy,
            'times': times,
            'energies': energies,
            'z_peaks': z_peaks,
            'states': states
        }
        
        return results
    
    def plot_results(self, results, output_path='alfven_wave_results.png'):
        """Generate benchmark plots
        
        Args:
            results: Results dictionary
            output_path: Output file path
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Wave propagation
        ax = axes[0, 0]
        ax.plot(results['times'], results['z_peaks'], 'b-', label='Peak position')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('z position (m)')
        ax.set_title('Wave Propagation')
        ax.legend()
        ax.grid(True)
        
        # Plot 2: Phase velocity fit
        ax = axes[0, 1]
        z_fit = results['v_phase_measured'] * results['times']
        ax.plot(results['times'], np.unwrap(results['z_peaks']), 'b.', label='Measured')
        ax.plot(results['times'], z_fit, 'r--', label=f'Fit (v={results["v_phase_measured"]:.2f})')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('z (unwrapped)')
        ax.set_title(f'Phase Velocity (error: {results["error_percent"]:.2f}%)')
        ax.legend()
        ax.grid(True)
        
        # Plot 3: Energy conservation
        ax = axes[1, 0]
        E0 = results['energies'][0]
        ax.plot(results['times'], (results['energies'] - E0) / E0 * 100, 'g-')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Energy drift (%)')
        ax.set_title(f'Energy Conservation (drift: {results["energy_drift_percent"]:.4f}%)')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.grid(True)
        
        # Plot 4: Final state snapshot
        ax = axes[1, 1]
        state_final = results['states'][-1]
        z_profile = np.mean(state_final.z_plus, axis=(0, 1))
        z_grid = np.linspace(0, 2*np.pi, len(z_profile))
        ax.plot(z_grid, z_profile, 'b-', label='Final state')
        ax.set_xlabel('z (m)')
        ax.set_ylabel('z⁺ (averaged)')
        ax.set_title('Final Wave Profile')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved: {output_path}")
        plt.close()


if __name__ == '__main__':
    # Run benchmark
    benchmark = AlfvenWaveBenchmark(grid_shape=(32, 64, 32))
    results = benchmark.run_benchmark(kz=1.0, dt=0.01, n_steps=100)
    
    # Generate plots
    benchmark.plot_results(results, 'alfven_wave_results.png')
    
    print("\nTest 1.1 complete ⚛️")
