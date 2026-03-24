"""
Elsasser MHD Solver Wrapper (Issue #26)

Bridges Elsasser evolution and MHD observation.

Design:
- Evolution: (z⁺, z⁻) via CompleteMHDSolver (no Poisson)
- Observation: convert to (ψ, φ) via Poisson solver

Author: 小P ⚛️
Date: 2026-03-24
"""

import jax.numpy as jnp
import numpy as np
import sys
sys.path.insert(0, '/Users/yz/.openclaw/workspace-xiaop/pim-rl-v3.0/src')

from pim_rl.physics.v2.elsasser_bracket import ElsasserState
from pim_rl.physics.v2.complete_solver_v2 import CompleteMHDSolver

from pytokmhd.operators import laplacian_toroidal
from pytokmhd.solvers import solve_poisson_toroidal
from pytokmhd.geometry import ToroidalGrid


class ElsasserMHDSolver:
    """
    MHD solver with Elsasser evolution and MHD observation interface.
    
    Internal state: (z⁺, z⁻) Elsasser variables
    External interface: (ψ, φ) MHD primitives
    
    Workflow:
    1. initialize(ψ, φ) → convert to (z⁺, z⁻) once
    2. step(dt) → evolve (z⁺, z⁻) (no Poisson)
    3. get_mhd_state() → convert to (ψ, φ) for observation (uses Poisson)
    
    Parameters
    ----------
    solver : CompleteMHDSolver
        Physics solver using Elsasser formulation
    grid : ToroidalGrid
        Toroidal geometry for Poisson solver
    """
    
    def __init__(self, solver: CompleteMHDSolver, grid: ToroidalGrid):
        self.solver = solver
        self.grid = grid
        
        # State
        self._state_els = None  # ElsasserState
        
        # BC storage (for Poisson inversion)
        self._psi_prev = None
        self._phi_prev = None
        
        print("ElsasserMHDSolver initialized")
        print(f"  Solver: {type(solver).__name__}")
        print(f"  Integrator: {solver.integrator.name}")
        print(f"  Grid: {grid.nr} × {grid.ntheta}")
        print(f"  Strategy: Evolution in (z⁺,z⁻), observation via Poisson conversion")
    
    def initialize(self, psi: jnp.ndarray, phi: jnp.ndarray):
        """
        Initialize from (ψ, φ).
        
        Forward conversion: (ψ, φ) → (v, B) → (z⁺, z⁻)
        Uses laplacian (not Poisson).
        
        Parameters
        ----------
        psi, phi : jnp.ndarray (nr, ntheta)
            Initial MHD state
        """
        # Convert to NumPy for operators
        psi_np = np.array(psi)
        phi_np = np.array(phi)
        
        # Compute vorticity and current
        v_np = laplacian_toroidal(phi_np, self.grid)  # ∇²φ
        B_np = laplacian_toroidal(psi_np, self.grid)  # ∇²ψ
        
        # Elsasser variables (need 3D for solver)
        # Extend to (nr, ntheta, nz)
        nz = self.solver.grid.Nz
        
        z_plus_2d = jnp.array(v_np + B_np)
        z_minus_2d = jnp.array(v_np - B_np)
        
        # Replicate in z direction
        z_plus = jnp.tile(z_plus_2d[:, :, None], (1, 1, nz))
        z_minus = jnp.tile(z_minus_2d[:, :, None], (1, 1, nz))
        P = jnp.zeros((self.grid.nr, self.grid.ntheta, nz))
        
        self._state_els = ElsasserState(z_plus=z_plus, z_minus=z_minus, P=P)
        
        # Store for BC
        self._psi_prev = psi
        self._phi_prev = phi
        
        print(f"  Initialized from (ψ, φ):")
        print(f"    ψ range: [{float(psi.min()):.3e}, {float(psi.max()):.3e}]")
        print(f"    φ range: [{float(phi.min()):.3e}, {float(phi.max()):.3e}]")
        print(f"    z⁺ range: [{float(z_plus.min()):.3e}, {float(z_plus.max()):.3e}]")
        print(f"    z⁻ range: [{float(z_minus.min()):.3e}, {float(z_minus.max()):.3e}]")
    
    def step(self, dt: float):
        """
        Evolve physics by dt.
        
        Evolution: (z⁺, z⁻) → CompleteMHDSolver → (z⁺, z⁻)_new
        No Poisson solver needed.
        
        Parameters
        ----------
        dt : float
            Timestep
        """
        if self._state_els is None:
            raise RuntimeError("Call initialize() before step()")
        
        self._state_els = self.solver.step(self._state_els, dt)
    
    def get_mhd_state(self) -> tuple:
        """
        Convert current (z⁺, z⁻) to (ψ, φ) for observation.
        
        Inverse conversion: (z⁺, z⁻) → (v, B) → (ψ, φ)
        Uses Poisson solver with BC from previous state.
        
        Returns
        -------
        psi, phi : jnp.ndarray (nr, ntheta)
            MHD primitives for observation
        """
        if self._state_els is None:
            raise RuntimeError("Call initialize() before get_mhd_state()")
        
        # Extract vorticity and current
        v_3d = (self._state_els.z_plus + self._state_els.z_minus) / 2
        B_3d = (self._state_els.z_plus - self._state_els.z_minus) / 2
        
        # Average over z (toroidal direction) to get 2D
        v = jnp.mean(v_3d, axis=2)
        B = jnp.mean(B_3d, axis=2)
        
        # Convert to NumPy for Poisson solver
        v_np = np.array(v)
        B_np = np.array(B)
        
        # Boundary conditions from previous (ψ, φ)
        if self._psi_prev is not None:
            psi_bnd = np.array(self._psi_prev[-1, :])
            phi_bnd = np.array(self._phi_prev[-1, :])
        else:
            # Fallback: zero BC (should not happen after initialize)
            psi_bnd = np.zeros(self.grid.ntheta)
            phi_bnd = np.zeros(self.grid.ntheta)
        
        # Solve ∇²ψ = B, ∇²φ = v
        psi_np, info_psi = solve_poisson_toroidal(B_np, self.grid, psi_bnd, tol=1e-6)
        phi_np, info_phi = solve_poisson_toroidal(v_np, self.grid, phi_bnd, tol=1e-6)
        
        # Check convergence
        if info_psi != 0:
            print(f"Warning: ψ Poisson solve did not converge (info={info_psi})")
        if info_phi != 0:
            print(f"Warning: φ Poisson solve did not converge (info={info_phi})")
        
        # Convert back to JAX
        psi = jnp.array(psi_np)
        phi = jnp.array(phi_np)
        
        # Update stored BC for next call
        self._psi_prev = psi
        self._phi_prev = phi
        
        return psi, phi
    
    def get_elsasser_state(self) -> ElsasserState:
        """
        Get current Elsasser state (for debugging/diagnostics).
        
        Returns
        -------
        state : ElsasserState
            Current (z⁺, z⁻, P)
        """
        return self._state_els
    
    def hamiltonian(self) -> float:
        """
        Compute Hamiltonian (energy) from current state.
        
        Returns
        -------
        H : float
            Total energy
        """
        if self._state_els is None:
            raise RuntimeError("Call initialize() first")
        
        return float(self.solver.hamiltonian(self._state_els))
