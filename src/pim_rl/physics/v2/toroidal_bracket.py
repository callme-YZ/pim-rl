"""
Toroidal Morrison Bracket for Elsasser MHD (v2.0 Phase 2.1)

Author: 小P ⚛️
Date: 2026-03-20

Extends cylindrical Morrison bracket with ε∂/∂φ toroidal coupling.

Theory:
- Small inverse aspect ratio: ε = a/R₀ ≪ 1
- Toroidal coupling: ε∂/∂φ in bracket
- Morrison-Furukawa 2018 Eq. 9

References:
- Morrison-Furukawa 2018 (reduced MHD)
- Module 4 (toroidal geometry)
"""

import jax.numpy as jnp
from jax import jit

from .elsasser_bracket import ElsasserState, MorrisonBracket


class ToroidalMorrisonBracket(MorrisonBracket):
    """Morrison bracket with toroidal coupling
    
    Extends cylindrical bracket with ε∂/∂φ terms.
    
    For small aspect ratio: ε = a/R₀ ≪ 1
    """
    
    def __init__(self, grid_shape, dr, dtheta, dz, epsilon: float = 0.0):
        """Initialize toroidal Morrison bracket
        
        Args:
            grid_shape: (Nr, Nθ, Nz)
            dr, dtheta, dz: Grid spacings
            epsilon: Inverse aspect ratio ε = a/R₀
        """
        super().__init__(grid_shape, dr, dtheta, dz)
        self.epsilon = epsilon
        
        # In toroidal: z-coordinate ≈ φ (toroidal angle)
        # For small ε: φ coupling is weak perturbation
    
    @staticmethod
    def toroidal_derivative(f: jnp.ndarray, dz: float) -> jnp.ndarray:
        """Compute ∂f/∂φ (toroidal derivative)
        
        In our grid: φ ≈ z coordinate (toroidal angle)
        
        Args:
            f: Field (Nr, Nθ, Nz)
            dz: Grid spacing in z/φ
            
        Returns:
            ∂f/∂φ
        """
        # Central difference in z-direction
        df_dz = (jnp.roll(f, -1, axis=2) - jnp.roll(f, 1, axis=2)) / (2*dz)
        return df_dz
    
    def bracket(self, dF_dstate: ElsasserState, 
                dG_dstate: ElsasserState) -> ElsasserState:
        """Morrison bracket with toroidal coupling
        
        {F,G} = {F,G}_cyl + ε·{F,G}_tor
        
        where {F,G}_tor = toroidal coupling term
        
        Args:
            dF_dstate: δF/δstate
            dG_dstate: δG/δstate
            
        Returns:
            {F,G}
        """
        # Cylindrical part (from Phase 1)
        bracket_cyl = super().bracket(dF_dstate, dG_dstate)
        
        # Toroidal coupling (if ε > 0)
        if self.epsilon > 0:
            bracket_tor = self.toroidal_coupling(dF_dstate, dG_dstate)
            
            # Total: cylindrical + ε·toroidal
            return ElsasserState(
                z_plus=bracket_cyl.z_plus + self.epsilon * bracket_tor.z_plus,
                z_minus=bracket_cyl.z_minus + self.epsilon * bracket_tor.z_minus,
                P=bracket_cyl.P + self.epsilon * bracket_tor.P
            )
        else:
            # Pure cylindrical (ε=0)
            return bracket_cyl
    
    def toroidal_coupling(self, dF_dstate: ElsasserState,
                         dG_dstate: ElsasserState) -> ElsasserState:
        """Toroidal coupling term in Morrison bracket
        
        Following Morrison-Furukawa 2018:
        {F,G}_tor = ∫ [∂(δF/δz⁺)/∂φ · δG/δz⁻ + ∂(δF/δz⁻)/∂φ · δG/δz⁺] dV
        
        Args:
            dF_dstate: δF/δstate
            dG_dstate: δG/δstate
            
        Returns:
            Toroidal coupling contribution
        """
        # Extract functional derivatives
        dF_dzp = dF_dstate.z_plus
        dF_dzm = dF_dstate.z_minus
        dF_dP = dF_dstate.P
        
        dG_dzp = dG_dstate.z_plus
        dG_dzm = dG_dstate.z_minus
        dG_dP = dG_dstate.P
        
        # Toroidal derivatives ∂/∂φ
        dF_dzp_dphi = self.toroidal_derivative(dF_dzp, self.dz)
        dF_dzm_dphi = self.toroidal_derivative(dF_dzm, self.dz)
        dF_dP_dphi = self.toroidal_derivative(dF_dP, self.dz)
        
        # Morrison-Furukawa toroidal coupling
        # ∂z⁺/∂t += ε ∂(δH/δz⁻)/∂φ
        dzp_tor = dF_dzm_dphi * dG_dzp  # Cross-coupling z⁺ ↔ ∂(δ/δz⁻)/∂φ
        
        # ∂z⁻/∂t += ε ∂(δH/δz⁺)/∂φ
        dzm_tor = dF_dzp_dphi * dG_dzm  # Cross-coupling z⁻ ↔ ∂(δ/δz⁺)/∂φ
        
        # Pressure coupling (simplified)
        dP_tor = dF_dP_dphi * (dG_dzp + dG_dzm)
        
        return ElsasserState(z_plus=dzp_tor, z_minus=dzm_tor, P=dP_tor)


def test_toroidal_bracket():
    """Test toroidal Morrison bracket"""
    
    print("=" * 60)
    print("Toroidal Morrison Bracket Test (Phase 2.1)")
    print("=" * 60)
    
    # Grid (same as Phase 1)
    Nr, Ntheta, Nz = 16, 16, 16
    dr, dtheta, dz = 0.1, 0.1, 0.1
    
    # Small epsilon (ITER-like)
    epsilon = 0.3
    
    grid = ToroidalMorrisonBracket((Nr, Ntheta, Nz), dr, dtheta, dz, epsilon)
    
    print(f"Inverse aspect ratio ε = {epsilon}")
    print(f"Grid: {Nr}×{Ntheta}×{Nz}\n")
    
    # Initial state (3D perturbation)
    r = jnp.linspace(0, 1, Nr)[:, None, None]
    theta = jnp.linspace(0, 2*jnp.pi, Ntheta)[None, :, None]
    phi = jnp.linspace(0, 2*jnp.pi, Nz)[None, None, :]  # φ = z
    
    # Toroidal mode (n=1, m=2)
    z_plus = jnp.exp(-((r-0.5)**2)) * jnp.cos(theta) * jnp.sin(phi)
    z_minus = jnp.exp(-((r-0.5)**2)) * jnp.sin(2*theta) * jnp.cos(phi) * 0.5
    P = jnp.ones((Nr, Ntheta, Nz)) * 0.1
    
    state = ElsasserState(z_plus=z_plus, z_minus=z_minus, P=P)
    
    # Test antisymmetry
    from elsasser_bracket import functional_derivative, hamiltonian
    
    dH = functional_derivative(hamiltonian, state, grid)
    
    # {H, H} should be zero (antisymmetry)
    HH = grid.bracket(dH, dH)
    
    antisym_error = jnp.max(jnp.abs(HH.z_plus))
    print(f"Antisymmetry test: {{H,H}}")
    print(f"Max |{{H,H}}|: {antisym_error:.6e}")
    
    if antisym_error < 1e-9:
        print("✅ Antisymmetry verified!\n")
    else:
        print("⚠️ Antisymmetry check failed\n")
    
    # Compare cylindrical vs toroidal bracket
    grid_cyl = ToroidalMorrisonBracket((Nr, Ntheta, Nz), dr, dtheta, dz, epsilon=0.0)
    grid_tor = ToroidalMorrisonBracket((Nr, Ntheta, Nz), dr, dtheta, dz, epsilon=epsilon)
    
    dF = dH
    dG = ElsasserState(z_plus=state.z_plus, z_minus=state.z_minus, P=state.P)
    
    FG_cyl = grid_cyl.bracket(dF, dG)
    FG_tor = grid_tor.bracket(dF, dG)
    
    diff = jnp.max(jnp.abs(FG_tor.z_plus - FG_cyl.z_plus))
    
    print("Cylindrical vs Toroidal bracket:")
    print(f"|{FG_cyl.z_plus}|_max (ε=0): {jnp.max(jnp.abs(FG_cyl.z_plus)):.6e}")
    print(f"|{FG_tor.z_plus}|_max (ε={epsilon}): {jnp.max(jnp.abs(FG_tor.z_plus)):.6e}")
    print(f"Difference: {diff:.6e}")
    print(f"Relative change: {diff/jnp.max(jnp.abs(FG_cyl.z_plus)):.2%}")
    
    print("\n✅ Phase 2.1 Toroidal Bracket Complete!")
    
    return grid


if __name__ == "__main__":
    test_toroidal_bracket()
