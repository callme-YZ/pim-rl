"""
MHD Environment with Hamiltonian-Aware Observations (v3.0)

Issue #25 Phase 2: Integration of HamiltonianObservation with Gym environment.

Uses:
- Issue #24 API: HamiltonianGradientComputer
- Issue #25 Phase 1: HamiltonianObservationScalar

Features:
- Exposes Hamiltonian structure to RL (H, ∇H, K, Ω, dH/dt)
- 22D observation vector (vs 11D Fourier-only)
- Compatible with PPO/SAC

Author: 小A 🤖
Physics: 小P ⚛️
Date: 2026-03-24
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jax.numpy as jnp
from typing import Dict, Tuple, Optional

from ..geometry.toroidal import ToroidalGrid
from ..solvers.hamiltonian_mhd_grad import HamiltonianGradientComputer
from .hamiltonian_observation import HamiltonianObservationScalar, ObservationNormalizer


class HamiltonianMHDEnv(gym.Env):
    """
    MHD control environment with Hamiltonian-aware observations.
    
    v3.0 improvements over v1.x:
    - Hamiltonian structure exposed (H, ∇H, K, Ω, dH/dt)
    - Physics-informed observations (not just Fourier modes)
    - Ready for structure-preserving RL
    
    Observation Space (23D):
    - H (1): Total Hamiltonian energy
    - K (1): Magnetic helicity  
    - Ω (1): Enstrophy (current²)
    - dH/dt (1): Dissipation rate
    - energy_drift (1): Relative energy change
    - grad_norm (1): ||∇H||
    - max_current (1): max|J|
    - psi_modes (8): Fourier modes of ψ
    - phi_modes (8): Fourier modes of φ
    
    Action Space (2D):
    - eta_multiplier ∈ [0.5, 2.0]: Resistivity control
    - nu_multiplier ∈ [0.5, 2.0]: Viscosity control
    
    Reward:
    - Encourage energy dissipation (dH/dt < 0)
    - Penalize large gradients (instability)
    - Penalize excessive actions
    
    Parameters
    ----------
    R0 : float
        Major radius [m] (default: 1.5)
    a : float
        Minor radius [m] (default: 0.5)
    nr : int
        Radial resolution (default: 32)
    ntheta : int
        Poloidal resolution (default: 64)
    dt : float
        Timestep (default: 1e-4)
    max_steps : int
        Episode length (default: 1000)
    eta : float
        Base resistivity (default: 1e-5)
    nu : float
        Base viscosity (default: 1e-4)
    normalize_obs : bool
        Use online normalization (default: True)
    """
    
    metadata = {'render_modes': []}
    
    def __init__(
        self,
        R0: float = 1.5,
        a: float = 0.5,
        nr: int = 32,
        ntheta: int = 64,
        dt: float = 1e-4,
        max_steps: int = 1000,
        eta: float = 1e-5,
        nu: float = 1e-4,
        normalize_obs: bool = True
    ):
        super().__init__()
        
        # Grid
        self.grid = ToroidalGrid(R0=R0, a=a, nr=nr, ntheta=ntheta)
        
        # Hamiltonian gradient computer (Issue #24)
        self.grad_computer = HamiltonianGradientComputer(self.grid)
        
        # Observation computer (Issue #25)
        self.obs_computer = HamiltonianObservationScalar(
            self.grid,
            self.grad_computer,
            dt=dt,
            n_modes=8
        )
        
        # Normalization
        self.normalize_obs = normalize_obs
        if normalize_obs:
            self.normalizer = ObservationNormalizer(obs_dim=23)
        
        # Episode config
        self.dt = dt
        self.max_steps = max_steps
        self.current_step = 0
        
        # Base parameters
        self.eta_base = eta
        self.nu_base = nu
        
        # Current state (will be initialized in reset())
        self.psi = None
        self.phi = None
        
        # Define spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(23,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=np.array([0.5, 0.5], dtype=np.float32),
            high=np.array([2.0, 2.0], dtype=np.float32),
            dtype=np.float32
        )
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.
        
        Returns
        -------
        obs : np.ndarray, shape (22,)
            Initial observation
        info : dict
            Additional info
        """
        super().reset(seed=seed)
        
        # Initialize state (simple perturbation)
        nr, ntheta = self.grid.nr, self.grid.ntheta
        r = np.linspace(0, 1, nr)[:, None]
        theta = np.linspace(0, 2*np.pi, ntheta)[None, :]
        
        # Initial flux (simple equilibrium + perturbation)
        self.psi = jnp.array(
            r**2 * (1 - r**2) * (1 + 0.1 * np.sin(2*theta)),
            dtype=jnp.float32
        )
        
        # Initial stream function (small)
        self.phi = jnp.array(
            0.01 * r * (1 - r) * np.cos(3*theta),
            dtype=jnp.float32
        )
        
        # Reset counters
        self.current_step = 0
        self.obs_computer.reset()
        
        # Compute observation
        obs = self._get_observation()
        
        info = {
            'step': self.current_step,
            'psi_norm': float(jnp.linalg.norm(self.psi)),
            'phi_norm': float(jnp.linalg.norm(self.phi))
        }
        
        return obs, info
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step.
        
        Parameters
        ----------
        action : np.ndarray, shape (2,)
            [eta_multiplier, nu_multiplier]
        
        Returns
        -------
        obs : np.ndarray
            Next observation
        reward : float
            Reward for this step
        terminated : bool
            Episode ended (reached goal or failed)
        truncated : bool
            Episode truncated (max steps)
        info : dict
            Additional info
        """
        # Extract action
        eta_mult, nu_mult = action
        eta = self.eta_base * float(eta_mult)
        nu = self.nu_base * float(nu_mult)
        
        # TODO: Replace with actual MHD solver step
        # For now, simple dummy step (to be replaced in Phase 3)
        self.psi, self.phi = self._dummy_solver_step(eta, nu)
        
        # Compute observation
        obs = self._get_observation()
        
        # Compute reward
        reward, reward_info = self._compute_reward(obs, action)
        
        # Check termination
        self.current_step += 1
        terminated = self._check_termination(obs)
        truncated = (self.current_step >= self.max_steps)
        
        # Info
        info = {
            'step': self.current_step,
            'reward_components': reward_info,
            'psi_norm': float(jnp.linalg.norm(self.psi)),
            'phi_norm': float(jnp.linalg.norm(self.phi)),
            'eta': eta,
            'nu': nu
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Compute Hamiltonian observation."""
        obs = self.obs_computer.compute_observation(self.psi, self.phi)
        
        if self.normalize_obs:
            obs = self.normalizer.normalize(obs)
        
        return obs
    
    def _compute_reward(
        self,
        obs: np.ndarray,
        action: np.ndarray
    ) -> Tuple[float, Dict]:
        """
        Compute reward using Hamiltonian structure.
        
        Reward components:
        1. Dissipation: encourage dH/dt < 0
        2. Gradient penalty: penalize large ||∇H||
        3. Action penalty: penalize excessive control
        """
        # Extract features from observation (before normalization)
        # Note: If normalized, need to track raw values
        # For now, assume obs is raw or we store raw values
        
        # Simple reward (to be tuned)
        # Encourage negative dH/dt (dissipation)
        # Note: obs[3] is dH/dt in the observation vector
        
        # Placeholder reward (proper implementation needs raw obs access)
        dissipation_reward = 0.0  # TODO: use raw dH/dt
        gradient_penalty = 0.0    # TODO: use raw grad_norm
        action_penalty = -0.01 * np.sum((action - 1.0)**2)
        
        reward = dissipation_reward + gradient_penalty + action_penalty
        
        reward_info = {
            'dissipation': dissipation_reward,
            'gradient': gradient_penalty,
            'action': action_penalty
        }
        
        return float(reward), reward_info
    
    def _check_termination(self, obs: np.ndarray) -> bool:
        """
        Check if episode should terminate.
        
        Terminate if:
        - Energy becomes NaN
        - Energy grows too large (unstable)
        """
        # TODO: Implement termination conditions
        return False
    
    def _dummy_solver_step(
        self,
        eta: float,
        nu: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Dummy solver step (placeholder).
        
        TODO: Replace with actual MHD solver in Phase 3.
        For now, simple decay.
        """
        # Simple exponential decay (dummy dynamics)
        decay_psi = 1.0 - eta * self.dt * 10.0
        decay_phi = 1.0 - nu * self.dt * 10.0
        
        psi_new = self.psi * decay_psi
        phi_new = self.phi * decay_phi
        
        return psi_new, phi_new
    
    def render(self):
        """Render environment (not implemented)."""
        pass
    
    def close(self):
        """Clean up resources."""
        pass


# Convenience function for creating environment
def make_hamiltonian_mhd_env(**kwargs) -> HamiltonianMHDEnv:
    """
    Create HamiltonianMHDEnv with default or custom parameters.
    
    Examples
    --------
    >>> env = make_hamiltonian_mhd_env()
    >>> env = make_hamiltonian_mhd_env(nr=64, ntheta=128, max_steps=500)
    """
    return HamiltonianMHDEnv(**kwargs)
