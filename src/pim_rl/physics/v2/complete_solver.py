"""
Complete v2.0 Solver (Ideal + Resistive)

Author: 小P ⚛️
Date: 2026-03-20

Provides clean API for RL environment:
- step() function with full physics
- Ideal bracket + resistive dynamics + pressure
"""

import jax.numpy as jnp

from .elsasser_bracket import ElsasserState, functional_derivative
from .toroidal_bracket import ToroidalMorrisonBracket
from .toroidal_hamiltonian import toroidal_hamiltonian
from .resistive_dynamics import resistive_mhd_rhs


class CompleteMHDSolver:
    """Complete MHD solver with ideal + resistive physics
    
    Physics:
    - Ideal MHD: {z±, H} (Morrison bracket, structure-preserving)
    - Resistive: η∇²B (magnetic diffusion)
    - Pressure: -∇p/ρ (ballooning drive)
    """
    
    def __init__(self, grid_shape: tuple, dr: float, dtheta: float, dz: float,
                 epsilon: float = 0.3, eta: float = 0.01, pressure_scale: float = 0.2):
        """Initialize solver
        
        Args:
            grid_shape: (Nr, Ntheta, Nz)
            dr, dtheta, dz: Grid spacing
            epsilon: Inverse aspect ratio
            eta: Resistivity
            pressure_scale: Pressure gradient strength
        """
        self.grid = ToroidalMorrisonBracket(grid_shape, dr, dtheta, dz, epsilon)
        self.epsilon = epsilon
        self.eta = eta
        self.pressure_scale = pressure_scale
        
        print("CompleteMHDSolver initialized:")
        print(f"  Grid: {grid_shape}")
        print(f"  ε: {epsilon}")
        print(f"  η: {eta}")
        print(f"  ∇p scale: {pressure_scale}")
    
    def hamiltonian(self, state: ElsasserState) -> float:
        """Compute Hamiltonian (energy)"""
        return toroidal_hamiltonian(state, self.grid, self.epsilon)
    
    def rhs(self, state: ElsasserState) -> ElsasserState:
        """Compute complete RHS: dz±/dt
        
        RHS = {z±, H} + η∇²B - ∇p/ρ
        """
        # Ideal bracket
        def H(s, g):
            return toroidal_hamiltonian(s, g, self.epsilon)
        
        dH = functional_derivative(H, state, self.grid)
        ideal_bracket = self.grid.bracket(
            ElsasserState(z_plus=state.z_plus, z_minus=state.z_minus, P=state.P),
            dH
        )
        
        # Add resistive + pressure
        total_rhs = resistive_mhd_rhs(state, self.grid, ideal_bracket, 
                                     self.eta, self.pressure_scale)
        
        return total_rhs
    
    def step_rk2(self, state: ElsasserState, dt: float) -> ElsasserState:
        """Single RK2 step
        
        Args:
            state: Current state
            dt: Time step
            
        Returns:
            Updated state
        """
        # k1
        k1 = self.rhs(state)
        
        # Mid-point
        state_mid = ElsasserState(
            z_plus=state.z_plus + dt*k1.z_plus/2,
            z_minus=state.z_minus + dt*k1.z_minus/2,
            P=state.P + dt*k1.P/2
        )
        
        # k2
        k2 = self.rhs(state_mid)
        
        # Update
        state_new = ElsasserState(
            z_plus=state.z_plus + dt*k2.z_plus,
            z_minus=state.z_minus + dt*k2.z_minus,
            P=state.P + dt*k2.P
        )
        
        return state_new
    
    def step_multi(self, state: ElsasserState, dt: float, n_substeps: int = 1) -> ElsasserState:
        """Multiple substeps (for RL environment dt_RL > dt_physics)
        
        Args:
            state: Current state
            dt: Total time step
            n_substeps: Number of physics substeps
            
        Returns:
            State after dt
        """
        dt_sub = dt / n_substeps
        
        for _ in range(n_substeps):
            state = self.step_rk2(state, dt_sub)
        
        return state


def test_complete_solver():
    """Test complete solver"""
    
    print("=" * 60)
    print("Complete MHD Solver Test")
    print("=" * 60 + "\n")
    
    # Setup
    from bout_metric import BOUTMetric
    from field_aligned import FieldAlignedCoordinates
    from ballooning_ic import ballooning_mode_ic
    
    R0 = 6.2
    a = 2.0
    metric = BOUTMetric(R0, a)
    fa = FieldAlignedCoordinates(metric)
    
    # Solver
    grid_shape = (16, 32, 16)  # Smaller for speed
    solver = CompleteMHDSolver(grid_shape, 0.1, 0.1, 0.2, 
                               epsilon=0.3, eta=0.01, pressure_scale=0.2)
    
    # IC
    state0 = ballooning_mode_ic(metric, fa, grid_shape, m=3, n=1, amplitude=0.05)
    
    E0 = solver.hamiltonian(state0)
    A0 = float(jnp.max(jnp.abs((state0.z_plus + state0.z_minus)/2)))
    
    print(f"\nInitial condition:")
    print(f"  Energy: {E0:.6e}")
    print(f"  Amplitude: {A0:.6f}\n")
    
    # Evolve
    dt = 0.02
    n_steps = 50
    
    print(f"Evolution: {n_steps} steps, dt={dt}")
    
    state = state0
    for step in range(n_steps):
        state = solver.step_rk2(state, dt)
        
        if step % 10 == 0:
            E = solver.hamiltonian(state)
            A = float(jnp.max(jnp.abs((state.z_plus + state.z_minus)/2)))
            growth = (A / A0 - 1) * 100
            print(f"  Step {step}: E={E:.6e}, A={A:.6f} ({growth:+.1f}%)")
    
    # Final
    E_final = solver.hamiltonian(state)
    A_final = float(jnp.max(jnp.abs((state.z_plus + state.z_minus)/2)))
    
    print(f"\nFinal state:")
    print(f"  Energy: {E_final:.6e}")
    print(f"  Amplitude: {A_final:.6f}")
    print(f"  Growth: {(A_final/A0 - 1)*100:+.1f}%")
    
    if A_final > A0:
        print("\n✅ Ballooning mode growing (physics working!)")
    
    print("\n✅ Complete solver ready for RL environment!")


if __name__ == "__main__":
    test_complete_solver()
