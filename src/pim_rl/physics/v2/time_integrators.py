"""
Time Integrator Interface for Structure-Preserving MHD

Issue #26: Pluggable integrator framework

Provides:
- Abstract TimeIntegrator base class
- RK2Integrator (current baseline)
- SymplecticIntegrator (Störmer-Verlet)

Author: 小P ⚛️
Date: 2026-03-24
"""

from abc import ABC, abstractmethod
from typing import Callable
import jax.numpy as jnp

from .elsasser_bracket import ElsasserState


class TimeIntegrator(ABC):
    """
    Abstract base class for time integrators.
    
    Integrators evolve MHD state: state(t) → state(t+dt)
    using RHS function: dz/dt = f(z, t)
    
    Subclasses implement different integration schemes:
    - RK2: 2nd order Runge-Kutta (not symplectic)
    - Symplectic: Structure-preserving (Störmer-Verlet, etc.)
    """
    
    @abstractmethod
    def step(
        self,
        state: ElsasserState,
        rhs_fn: Callable[[ElsasserState], ElsasserState],
        dt: float
    ) -> ElsasserState:
        """
        Advance state by one timestep.
        
        Parameters
        ----------
        state : ElsasserState
            Current state (z⁺, z⁻, P)
        rhs_fn : callable
            RHS function: state → dstate/dt
        dt : float
            Timestep
            
        Returns
        -------
        state_new : ElsasserState
            State at t + dt
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Integrator name (for logging)."""
        pass
    
    @property
    @abstractmethod
    def order(self) -> int:
        """Convergence order (1, 2, 4, etc.)."""
        pass
    
    @property
    @abstractmethod
    def is_symplectic(self) -> bool:
        """Whether integrator preserves symplectic structure."""
        pass


class RK2Integrator(TimeIntegrator):
    """
    2nd-order Runge-Kutta (midpoint method).
    
    Algorithm:
        k1 = f(z_n)
        k2 = f(z_n + dt/2 * k1)
        z_{n+1} = z_n + dt * k2
    
    Properties:
    - Order: 2
    - Symplectic: No
    - Simple and stable
    
    Notes:
    - Current baseline for PyTokMHD v2.0
    - Good accuracy for moderate timesteps
    - NOT structure-preserving (energy drift)
    """
    
    @property
    def name(self) -> str:
        return "RK2"
    
    @property
    def order(self) -> int:
        return 2
    
    @property
    def is_symplectic(self) -> bool:
        return False
    
    def step(
        self,
        state: ElsasserState,
        rhs_fn: Callable[[ElsasserState], ElsasserState],
        dt: float
    ) -> ElsasserState:
        """RK2 step (midpoint method)."""
        
        # k1 = f(z_n)
        k1 = rhs_fn(state)
        
        # Mid-point state
        state_mid = ElsasserState(
            z_plus=state.z_plus + dt * k1.z_plus / 2,
            z_minus=state.z_minus + dt * k1.z_minus / 2,
            P=state.P + dt * k1.P / 2
        )
        
        # k2 = f(z_n + dt/2 * k1)
        k2 = rhs_fn(state_mid)
        
        # z_{n+1} = z_n + dt * k2
        state_new = ElsasserState(
            z_plus=state.z_plus + dt * k2.z_plus,
            z_minus=state.z_minus + dt * k2.z_minus,
            P=state.P + dt * k2.P
        )
        
        return state_new


class SymplecticIntegrator(TimeIntegrator):
    """
    Symplectic integrator (Störmer-Verlet).
    
    For Hamiltonian system: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q
    
    Algorithm:
        p_{n+1/2} = p_n - dt/2 * ∇_q H(q_n)
        q_{n+1} = q_n + dt * ∇_p H(p_{n+1/2})
        p_{n+1} = p_{n+1/2} - dt/2 * ∇_q H(q_{n+1})
    
    Properties:
    - Order: 2
    - Symplectic: Yes (preserves phase space volume)
    - Energy: Bounded drift (oscillates, not monotonic)
    
    For MHD (z± formulation):
    - Need to split into "position" and "momentum"
    - Use: z⁺ as q, z⁻ as p (or vice versa)
    
    Notes:
    - Requires separable Hamiltonian H(q,p) = T(p) + V(q)
    - For full MHD (non-separable), use implicit midpoint instead
    """
    
    def __init__(self, use_implicit_midpoint: bool = True):
        """
        Initialize symplectic integrator.
        
        Parameters
        ----------
        use_implicit_midpoint : bool
            If True, use implicit midpoint (works for non-separable H)
            If False, use Störmer-Verlet (requires separable H)
            Default: True (safer for MHD)
        """
        self.use_implicit = use_implicit_midpoint
    
    @property
    def name(self) -> str:
        if self.use_implicit:
            return "Implicit Midpoint (Symplectic)"
        else:
            return "Störmer-Verlet (Symplectic)"
    
    @property
    def order(self) -> int:
        return 2
    
    @property
    def is_symplectic(self) -> bool:
        return True
    
    def step(
        self,
        state: ElsasserState,
        rhs_fn: Callable[[ElsasserState], ElsasserState],
        dt: float
    ) -> ElsasserState:
        """
        Symplectic step.
        
        For now: Use implicit midpoint (safest for MHD).
        Future: Implement explicit Störmer-Verlet for separable H.
        """
        if self.use_implicit:
            return self._implicit_midpoint_step(state, rhs_fn, dt)
        else:
            raise NotImplementedError("Explicit Störmer-Verlet not yet implemented")
    
    def _implicit_midpoint_step(
        self,
        state: ElsasserState,
        rhs_fn: Callable[[ElsasserState], ElsasserState],
        dt: float
    ) -> ElsasserState:
        """
        Implicit midpoint rule (symplectic).
        
        Algorithm:
            z_{n+1} = z_n + dt * f((z_n + z_{n+1})/2)
        
        Solved iteratively (fixed-point):
            z^{k+1}_{n+1} = z_n + dt * f((z_n + z^k_{n+1})/2)
        
        Properties:
        - Symplectic for all Hamiltonians
        - 2nd order accurate
        - Requires iteration (expensive)
        """
        
        # Initial guess: explicit Euler
        z_new = ElsasserState(
            z_plus=state.z_plus + dt * rhs_fn(state).z_plus,
            z_minus=state.z_minus + dt * rhs_fn(state).z_minus,
            P=state.P + dt * rhs_fn(state).P
        )
        
        # Fixed-point iteration
        max_iter = 5  # Usually converges in 2-3 iterations
        for _ in range(max_iter):
            # Midpoint
            z_mid = ElsasserState(
                z_plus=(state.z_plus + z_new.z_plus) / 2,
                z_minus=(state.z_minus + z_new.z_minus) / 2,
                P=(state.P + z_new.P) / 2
            )
            
            # Update
            rhs_mid = rhs_fn(z_mid)
            z_new = ElsasserState(
                z_plus=state.z_plus + dt * rhs_mid.z_plus,
                z_minus=state.z_minus + dt * rhs_mid.z_minus,
                P=state.P + dt * rhs_mid.P
            )
        
        return z_new


# Factory function
def make_integrator(name: str, **kwargs) -> TimeIntegrator:
    """
    Create integrator by name.
    
    Parameters
    ----------
    name : str
        'rk2', 'symplectic', or 'implicit_midpoint'
    **kwargs : dict
        Integrator-specific options
        
    Returns
    -------
    integrator : TimeIntegrator
        Integrator instance
        
    Examples
    --------
    >>> integrator = make_integrator('rk2')
    >>> integrator = make_integrator('symplectic', use_implicit_midpoint=True)
    """
    name_lower = name.lower()
    
    if name_lower == 'rk2':
        return RK2Integrator()
    elif name_lower in ('symplectic', 'implicit_midpoint'):
        return SymplecticIntegrator(**kwargs)
    else:
        raise ValueError(f"Unknown integrator: {name}. Options: 'rk2', 'symplectic'")
