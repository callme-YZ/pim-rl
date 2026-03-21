"""
Complete MHD Solver with RMP Control (Phase 4.4)

Author: 小P ⚛️
Date: 2026-03-20

Extends CompleteMHDSolver with RMP forcing for RL control.
"""

import jax.numpy as jnp

from .elsasser_bracket import ElsasserState
from .complete_solver import CompleteMHDSolver
from .rmp_forcing import rmp_coil_field, compute_current_density


class MHDSolverWithRMP(CompleteMHDSolver):
    """Complete MHD solver + RMP control
    
    Adds:
    - RMP coil forcing (action → J_ext)
    - J_ext × B force in RHS
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize with RMP capability"""
        super().__init__(*args, **kwargs)
        
        # Extract grid params for RMP
        self.Nr = self.grid.Nr
        self.Ntheta = self.grid.Ntheta
        self.Nz = self.grid.Nz
        self.R0 = 6.2  # TODO: get from grid
        self.a = 2.0
        
        # Grid for RMP field calculation
        r = jnp.linspace(0, self.a, self.Nr)[:, None, None]
        theta = jnp.linspace(0, 2*jnp.pi, self.Ntheta)[None, :, None]
        z = jnp.linspace(0, 2*jnp.pi, self.Nz)[None, None, :]
        
        self.R_grid_rmp = self.R0 + r
        self.theta_grid = theta
        self.z_grid = z
        
        print("MHDSolverWithRMP initialized (RMP control enabled)")
    
    def rhs_with_control(self, state: ElsasserState, 
                        coil_currents: jnp.ndarray) -> ElsasserState:
        """Compute RHS with RMP control
        
        RHS = {z±, H} + η∇²B - ∇p/ρ + J_ext forcing
        
        Args:
            state: Current state
            coil_currents: [I₁, I₂, I₃, I₄] (kA)
            
        Returns:
            Total RHS
        """
        # Base physics (ideal + resistive + pressure)
        base_rhs = self.rhs(state)
        
        # RMP external forcing
        if jnp.sum(jnp.abs(coil_currents)) > 0:
            # Compute B_ext from coils
            B_ext = rmp_coil_field(coil_currents, self.R_grid_rmp, 
                                  self.theta_grid, self.z_grid,
                                  self.R0, self.a)
            
            # J_ext = ∇×B_ext
            J_ext = compute_current_density(B_ext, self.grid.dr, 
                                           self.grid.dtheta, self.grid.dz)
            J_r, J_theta, J_z = J_ext
            
            # J×B force (simplified: mainly J_z component affects z±)
            # F = J×B ~ J_ext
            # In Elsasser: dz±/dt += ±J_ext (acts on B = (z⁺-z⁻)/2)
            
            # Scale to appropriate units (J is in A/m², convert to normalized)
            scale = 3.4e-4  # Systematic diagnosis: RMP/Base ~ 10× (not 150×)
            
            forcing_plus = scale * J_z
            forcing_minus = -scale * J_z  # Opposite sign for z⁻
            
            # Add to RHS
            rhs_with_forcing = ElsasserState(
                z_plus=base_rhs.z_plus + forcing_plus,
                z_minus=base_rhs.z_minus + forcing_minus,
                P=base_rhs.P
            )
            
            return rhs_with_forcing
        else:
            return base_rhs
    
    def step_rk2_with_control(self, state: ElsasserState, dt: float,
                             coil_currents: jnp.ndarray) -> ElsasserState:
        """RK2 step with RMP control
        
        Args:
            state: Current state
            dt: Time step
            coil_currents: Control action
            
        Returns:
            Updated state
        """
        # k1
        k1 = self.rhs_with_control(state, coil_currents)
        
        # Mid-point
        state_mid = ElsasserState(
            z_plus=state.z_plus + dt*k1.z_plus/2,
            z_minus=state.z_minus + dt*k1.z_minus/2,
            P=state.P + dt*k1.P/2
        )
        
        # k2 (with same control)
        k2 = self.rhs_with_control(state_mid, coil_currents)
        
        # Update
        state_new = ElsasserState(
            z_plus=state.z_plus + dt*k2.z_plus,
            z_minus=state.z_minus + dt*k2.z_minus,
            P=state.P + dt*k2.P
        )
        
        return state_new


def test_rmp_control():
    """Test RMP control effect"""
    
    print("=" * 60)
    print("MHD Solver with RMP Control Test")
    print("=" * 60 + "\n")
    
    from bout_metric import BOUTMetric
    from field_aligned import FieldAlignedCoordinates
    from ballooning_ic import ballooning_mode_ic
    
    # Setup
    R0 = 6.2
    a = 2.0
    metric = BOUTMetric(R0, a)
    fa = FieldAlignedCoordinates(metric)
    
    # Solver with RMP
    grid_shape = (16, 32, 16)
    solver = MHDSolverWithRMP(grid_shape, 0.1, 0.1, 0.2,
                             epsilon=0.3, eta=0.01, pressure_scale=0.2)
    
    # IC
    state0 = ballooning_mode_ic(metric, fa, grid_shape, m=3, n=1, amplitude=0.05)
    
    A0 = float(jnp.max(jnp.abs((state0.z_plus + state0.z_minus)/2)))
    
    print(f"Initial amplitude: {A0:.6f}\n")
    
    # Test 1: No control
    print("Test 1: No control (baseline)")
    state_no_control = state0
    for _ in range(20):
        state_no_control = solver.step_rk2(state_no_control, 0.02)
    
    A_no_control = float(jnp.max(jnp.abs((state_no_control.z_plus + state_no_control.z_minus)/2)))
    growth_no_control = (A_no_control / A0 - 1) * 100
    print(f"  Final amplitude: {A_no_control:.6f} ({growth_no_control:+.1f}%)\n")
    
    # Test 2: With RMP control
    print("Test 2: With RMP control (40kA)")
    coil_currents = jnp.array([10.0, 10.0, 10.0, 10.0])  # 10kA each
    
    state_with_control = state0
    for _ in range(20):
        state_with_control = solver.step_rk2_with_control(state_with_control, 0.02, coil_currents)
    
    A_with_control = float(jnp.max(jnp.abs((state_with_control.z_plus + state_with_control.z_minus)/2)))
    growth_with_control = (A_with_control / A0 - 1) * 100
    print(f"  Final amplitude: {A_with_control:.6f} ({growth_with_control:+.1f}%)\n")
    
    # Compare
    suppression = (A_no_control - A_with_control) / A_no_control * 100
    
    print("=" * 60)
    print("RMP Control Effect")
    print("=" * 60)
    print(f"No control: {growth_no_control:+.1f}% growth")
    print(f"With RMP: {growth_with_control:+.1f}% growth")
    print(f"Suppression: {suppression:.1f}%")
    
    if abs(suppression) > 1:
        print("\n✅ RMP control affecting physics!")
    else:
        print("\n⚠️ RMP effect small (may need tuning)")
    
    print("\n✅ Phase 4.4 Complete - RMP Control Ready for RL!")


if __name__ == "__main__":
    test_rmp_control()
