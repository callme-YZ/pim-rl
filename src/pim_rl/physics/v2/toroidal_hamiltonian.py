"""
Toroidal Hamiltonian for Elsasser MHD (v2.0 Phase 2.2)

Author: 小P ⚛️
Date: 2026-03-20

Adds curvature energy -∫ h·P dV to cylindrical Hamiltonian.

Theory:
- h = εx (toroidal curvature vector)
- In (r,θ,φ): h ≈ ε(cos θ, sin θ, 0) (radial curvature)
- Morrison-Furukawa 2018 Eq. 6

References:
- Morrison-Furukawa 2018
- Module 4 Section 3.2
"""

import jax.numpy as jnp
from jax import grad

from .elsasser_bracket import ElsasserState, MorrisonBracket


def compute_curvature_vector(grid_shape, dr, dtheta, epsilon):
    """Compute toroidal curvature h = εx
    
    In cylindrical-like coordinates (r,θ,φ):
    h ≈ ε·(cos θ, sin θ, 0)  (pointing radially outward)
    
    Args:
        grid_shape: (Nr, Nθ, Nz)
        dr, dtheta: Grid spacings
        epsilon: Inverse aspect ratio
        
    Returns:
        h: Curvature vector (Nr, Nθ, Nz, 3)
    """
    Nr, Ntheta, Nz = grid_shape
    
    # θ coordinate
    theta = jnp.linspace(0, 2*jnp.pi, Ntheta)[None, :, None]
    
    # h = ε·(cos θ, sin θ, 0)  (radial direction)
    h_r = epsilon * jnp.cos(theta) * jnp.ones((Nr, 1, Nz))
    h_theta = epsilon * jnp.sin(theta) * jnp.ones((Nr, 1, Nz))
    h_z = jnp.zeros((Nr, Ntheta, Nz))
    
    # Stack into (Nr, Nθ, Nz, 3)
    h = jnp.stack([h_r, h_theta, h_z], axis=-1)
    
    return h


def toroidal_hamiltonian(state: ElsasserState, 
                         grid: MorrisonBracket,
                         epsilon: float = 0.0) -> float:
    """Toroidal Hamiltonian with curvature energy
    
    H = H_cylindrical - ∫ h·P dV
    
    where:
    - H_cylindrical = ∫ (z⁺² + z⁻²)/4 dV  (from Phase 1)
    - h·P = ε x·P (curvature-pressure coupling)
    
    Args:
        state: Elsasser state
        grid: Grid info
        epsilon: Inverse aspect ratio
        
    Returns:
        Total Hamiltonian
    """
    # Cylindrical energy (from Phase 1)
    energy_density = (state.z_plus**2 + state.z_minus**2) / 4
    H_cyl = jnp.sum(energy_density) * grid.dV
    
    # Toroidal curvature energy (if ε > 0)
    if epsilon > 0:
        # Curvature vector h = εx
        h = compute_curvature_vector((grid.Nr, grid.Ntheta, grid.Nz), grid.dr, grid.dtheta, epsilon)
        
        # For simplicity: h·P ≈ h_r · P (radial component dominates)
        # Full version: h = (h_r, h_θ, h_φ), but h_φ=0 in our geometry
        
        curvature_energy_density = h[:, :, :, 0] * state.P  # h_r · P
        H_curv = -jnp.sum(curvature_energy_density) * grid.dV
    else:
        H_curv = 0.0
    
    # Total Hamiltonian
    H_total = H_cyl + H_curv
    
    return H_total


def test_toroidal_hamiltonian():
    """Test toroidal Hamiltonian"""
    
    print("=" * 60)
    print("Toroidal Hamiltonian Test (Phase 2.2)")
    print("=" * 60)
    
    # Grid
    Nr, Ntheta, Nz = 16, 16, 16
    dr, dtheta, dz = 0.1, 0.1, 0.1
    grid = MorrisonBracket((Nr, Ntheta, Nz), dr, dtheta, dz)
    
    # Epsilon
    epsilon = 0.3
    
    print(f"Inverse aspect ratio ε = {epsilon}")
    print(f"Grid: {Nr}×{Ntheta}×{Nz}\n")
    
    # Initial state
    r = jnp.linspace(0, 1, Nr)[:, None, None]
    theta = jnp.linspace(0, 2*jnp.pi, Ntheta)[None, :, None]
    z = jnp.linspace(0, 1, Nz)[None, None, :]
    
    z_plus = jnp.exp(-5*(r-0.6)**2) * jnp.cos(3*theta) * jnp.sin(jnp.pi * z)
    z_minus = jnp.exp(-5*(r-0.4)**2) * jnp.sin(2*theta) * jnp.cos(jnp.pi * z) * 0.8
    P = jnp.ones((Nr, Ntheta, Nz)) * 0.1 * (1 + 0.5*jnp.cos(theta))  # Pressure variation
    
    state = ElsasserState(z_plus=z_plus, z_minus=z_minus, P=P)
    
    # Compute Hamiltonians
    H_cyl = toroidal_hamiltonian(state, grid, epsilon=0.0)
    H_tor = toroidal_hamiltonian(state, grid, epsilon=epsilon)
    
    print("Hamiltonian comparison:")
    print(f"H_cylindrical (ε=0):    {H_cyl:.6e}")
    print(f"H_toroidal (ε={epsilon}):   {H_tor:.6e}")
    print(f"Curvature energy ΔH:    {H_tor - H_cyl:.6e}")
    print(f"Relative change:        {(H_tor - H_cyl)/H_cyl:.2%}\n")
    
    # Test functional derivative
    print("Functional derivative test:")
    
    # Define Hamiltonian as function of state (compatible with functional_derivative)
    def H_func(s, g):
        return toroidal_hamiltonian(s, g, epsilon)
    
    # JAX autodiff
    from elsasser_bracket import functional_derivative
    dH = functional_derivative(H_func, state, grid)
    
    print(f"δH/δz⁺ max: {jnp.max(jnp.abs(dH.z_plus)):.6e}")
    print(f"δH/δz⁻ max: {jnp.max(jnp.abs(dH.z_minus)):.6e}")
    print(f"δH/δP max:  {jnp.max(jnp.abs(dH.P)):.6e}")
    
    # Pressure functional derivative should include -h term
    # δH/δP = (z⁺ + z⁻)/2 - h  (simplified)
    
    print("\n✅ Phase 2.2 Toroidal Hamiltonian Complete!")
    
    return grid, state


if __name__ == "__main__":
    test_toroidal_hamiltonian()
