"""
Toroidal MHD Environment for RL (M4 Phase 3)

Integrates ToroidalMHDSolver with Gymnasium for RL training.

**Key Features:**
- Real physics via ToroidalMHDSolver
- Fourier-based observation extraction
- Multi-objective reward (energy, div_B, action)
- Action = [eta_multiplier, nu_multiplier]
- Physics constraints verification (div_B < 1e-6)

Author: 小A 🤖
Date: 2026-03-17
Phase: M4.3 - RL Integration
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Tuple, Optional

from ..geometry import ToroidalGrid
from ..solvers import ToroidalMHDSolver
from ..solvers.equilibrium import circular_equilibrium
from ..solvers.diagnostics import compute_energy
from ..operators import divergence_B_toroidal


class ToroidalMHDEnv(gym.Env):
    """
    Gymnasium environment for toroidal MHD control.
    
    **Observation (11D):**
    - psi_modes: Fourier coefficients [8D] - (cos(m*θ), sin(m*θ)) for m=1,2,3,4
    - energy: Total MHD energy [1D]
    - div_B: Divergence-free constraint violation [1D]
    - previous_action: Last action taken [1D]
    
    **Action (2D):**
    - eta_multiplier: Resistivity multiplier ∈ [0.5, 2.0]
    - nu_multiplier: Viscosity multiplier ∈ [0.5, 2.0]
    
    **Reward:**
    r = -w_E * energy - w_B * div_B - w_A * |action - 1|^2
    
    Weights:
    - w_E = 1.0 (energy minimization)
    - w_B = 0.1 (constraint enforcement)
    - w_A = 0.01 (regularization)
    
    **Physics Constraints:**
    - ∇·B < 1e-6 (verified every step)
    - CFL condition (implicit in solver)
    
    Parameters
    ----------
    grid_size : int
        Radial/poloidal resolution (default: 32)
    dt : float
        Time step (default: 0.01)
    eta_base : float
        Base resistivity (default: 1e-5)
    nu_base : float
        Base viscosity (default: 1e-4)
    max_steps : int
        Episode length (default: 200)
    R0 : float
        Major radius (default: 1.0)
    a : float
        Minor radius (default: 0.3)
    w_energy : float
        Energy weight (default: 1.0)
    w_div_B : float
        Divergence weight (default: 0.1)
    w_action : float
        Action regularization (default: 0.01)
    
    Examples
    --------
    >>> env = ToroidalMHDEnv(grid_size=32)
    >>> obs, info = env.reset()
    >>> obs.shape
    (11,)
    >>> action = env.action_space.sample()
    >>> obs, reward, terminated, truncated, info = env.step(action)
    """
    
    metadata = {'render_modes': []}
    
    def __init__(
        self,
        grid_size: int = 32,
        dt: float = 0.01,
        eta_base: float = 1e-5,
        nu_base: float = 1e-4,
        max_steps: int = 200,
        R0: float = 1.0,
        a: float = 0.3,
        w_energy: float = 1.0,
        w_div_B: float = 0.1,
        w_action: float = 0.01,
    ):
        super().__init__()
        
        # Store config
        self.grid_size = grid_size
        self.dt = dt
        self.eta_base = eta_base
        self.nu_base = nu_base
        self.max_steps = max_steps
        self.R0 = R0
        self.a = a
        
        # Reward weights
        self.w_energy = w_energy
        self.w_div_B = w_div_B
        self.w_action = w_action
        
        # Create grid
        self.grid = ToroidalGrid(
            R0=R0,
            a=a,
            nr=grid_size,
            ntheta=2 * grid_size
        )
        
        # Observation: 8 (Fourier) + 1 (energy) + 1 (div_B) + 1 (prev_action) = 11D
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(11,),
            dtype=np.float32
        )
        
        # Action: [eta_multiplier, nu_multiplier] ∈ [0.5, 2.0]^2
        self.action_space = spaces.Box(
            low=0.5,
            high=2.0,
            shape=(2,),
            dtype=np.float32
        )
        
        # Solver (initialized in reset)
        self.solver = None
        
        # State
        self.step_count = 0
        self.previous_action = np.array([1.0, 1.0], dtype=np.float32)
        self.episode_reward = 0.0
        
    def _fourier_decomposition(self, psi: np.ndarray) -> np.ndarray:
        """
        Extract Fourier modes from psi.
        
        Computes [cos(θ), sin(θ), cos(2θ), sin(2θ), ..., cos(4θ), sin(4θ)]
        
        Parameters
        ----------
        psi : np.ndarray (nr, ntheta)
        
        Returns
        -------
        modes : np.ndarray (8,)
            [c1, s1, c2, s2, c3, s3, c4, s4]
        
        Notes
        -----
        Uses radial average at mid-radius to capture dominant mode structure.
        """
        nr, ntheta = psi.shape
        theta = np.linspace(0, 2*np.pi, ntheta, endpoint=False)
        
        # Extract radial profile at mid-radius
        r_mid = nr // 2
        psi_theta = psi[r_mid, :]
        
        modes = []
        for m in range(1, 5):  # m = 1, 2, 3, 4
            # Cosine coefficient
            c_m = np.mean(psi_theta * np.cos(m * theta)) * 2
            # Sine coefficient
            s_m = np.mean(psi_theta * np.sin(m * theta)) * 2
            modes.extend([c_m, s_m])
        
        return np.array(modes, dtype=np.float32)
    
    def _compute_div_B(self, psi: np.ndarray) -> float:
        """
        Compute divergence-free constraint violation.
        
        Parameters
        ----------
        psi : np.ndarray (nr, ntheta)
        
        Returns
        -------
        div_B_norm : float
            RMS divergence ∇·B
        """
        div_B = divergence_B_toroidal(psi, self.grid)
        return float(np.sqrt(np.mean(div_B**2)))
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct observation vector.
        
        Returns
        -------
        obs : np.ndarray (11,)
            [psi_modes(8), energy(1), div_B(1), prev_action_mean(1)]
        """
        psi = self.solver.psi
        omega = self.solver.omega
        
        # Fourier modes (8D)
        psi_modes = self._fourier_decomposition(psi)
        
        # Energy (1D)
        energy = compute_energy(psi, omega, self.grid)
        energy_normalized = energy / (self.grid.nr * self.grid.ntheta)
        
        # Divergence constraint (1D)
        div_B = self._compute_div_B(psi)
        
        # Previous action (1D) - use mean of multipliers
        prev_action_mean = float(np.mean(self.previous_action))
        
        obs = np.concatenate([
            psi_modes,
            [energy_normalized, div_B, prev_action_mean]
        ])
        
        return obs.astype(np.float32)
    
    def _compute_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        """
        Multi-objective reward.
        
        r = -w_E * energy - w_B * div_B - w_A * ||action - 1||^2
        
        Parameters
        ----------
        obs : np.ndarray (11,)
        action : np.ndarray (2,)
        
        Returns
        -------
        reward : float
        """
        energy = obs[8]
        div_B = obs[9]
        
        # Action deviation from nominal (1.0, 1.0)
        action_dev = np.sum((action - 1.0)**2)
        
        reward = (
            - self.w_energy * energy
            - self.w_div_B * div_B
            - self.w_action * action_dev
        )
        
        return float(reward)
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment.
        
        Returns
        -------
        obs : np.ndarray (11,)
        info : dict
        """
        super().reset(seed=seed)
        
        # Create solver with base parameters
        self.solver = ToroidalMHDSolver(
            grid=self.grid,
            dt=self.dt,
            eta=self.eta_base,
            nu=self.nu_base
        )
        
        # Initialize with circular equilibrium
        psi0 = circular_equilibrium(self.grid)
        omega0 = np.zeros_like(psi0)
        
        # Add small perturbation
        perturbation = 0.01 * np.random.randn(*psi0.shape)
        psi0 += perturbation
        
        self.solver.initialize(psi0, omega0)
        
        # Reset counters
        self.step_count = 0
        self.previous_action = np.array([1.0, 1.0], dtype=np.float32)
        self.episode_reward = 0.0
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one timestep.
        
        Parameters
        ----------
        action : np.ndarray (2,)
            [eta_multiplier, nu_multiplier] ∈ [0.5, 2.0]^2
        
        Returns
        -------
        obs : np.ndarray (11,)
        reward : float
        terminated : bool
        truncated : bool
        info : dict
        """
        # Clip action to valid range (safety)
        action = np.clip(action, 0.5, 2.0)
        
        # Apply effective parameters
        eta_effective = self.eta_base * action[0]
        nu_effective = self.nu_base * action[1]
        
        # Update solver parameters
        self.solver.eta = eta_effective
        self.solver.nu = nu_effective
        
        # Evolve one step
        self.solver.step()
        
        # Get observation
        obs = self._get_observation()
        
        # Compute reward
        reward = self._compute_reward(obs, action)
        
        # Update state
        self.step_count += 1
        self.episode_reward += reward
        self.previous_action = action.copy()
        
        # Check termination
        terminated = False
        truncated = False
        
        # Physics constraint violation (safety)
        div_B = obs[9]
        if div_B > 1e-4:  # Relaxed threshold for stability
            terminated = True
        
        # Max steps
        if self.step_count >= self.max_steps:
            truncated = True
        
        info = self._get_info()
        info['div_B_violation'] = div_B > 1e-6
        
        return obs, reward, terminated, truncated, info
    
    def _get_info(self) -> Dict:
        """Get diagnostic info."""
        if self.solver is None or self.solver.psi is None:
            return {
                'step': self.step_count,
                'episode_reward': self.episode_reward,
            }
        
        return {
            'step': self.step_count,
            'episode_reward': self.episode_reward,
            'psi_max': float(np.max(np.abs(self.solver.psi))),
            'omega_max': float(np.max(np.abs(self.solver.omega))),
            'eta': self.solver.eta,
            'nu': self.solver.nu,
            'div_B': float(self._compute_div_B(self.solver.psi)),
        }
    
    def render(self):
        """Render (not implemented)."""
        pass
    
    def close(self):
        """Clean up."""
        pass
