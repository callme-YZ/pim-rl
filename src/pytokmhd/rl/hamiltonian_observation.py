"""
Hamiltonian-Aware Observation for RL Environment

Issue #25: Expose Hamiltonian structure to RL via observations.

Provides:
1. Hamiltonian energy H
2. Energy gradients ∇H (δH/δψ, δH/δφ)
3. Conserved quantities (helicity, enstrophy)
4. Dissipation rate dH/dt

Uses Issue #24 API (HamiltonianGradientComputer) for efficient ∇H computation.

Author: 小A 🤖
Physics guidance: 小P ⚛️
Date: 2026-03-24
"""

import jax.numpy as jnp
from jax import jit
from typing import Dict, Tuple, Optional
import numpy as np

from ..solvers.hamiltonian_mhd_grad import HamiltonianGradientComputer
from ..geometry.toroidal import ToroidalGrid
from ..operators.toroidal_operators import laplacian_toroidal


class HamiltonianObservation:
    """
    Hamiltonian-aware observation computer.
    
    Computes physics-informed observations that expose Hamiltonian structure:
    - Total energy H
    - Energy gradients δH/δψ, δH/δφ
    - Conserved quantities (helicity K, enstrophy Ω)
    - Dissipation rate dH/dt
    
    Parameters
    ----------
    grid : ToroidalGrid
        Computational grid
    grad_computer : HamiltonianGradientComputer
        Gradient computer from Issue #24
    dt : float
        Timestep for dH/dt computation (default: 0.01)
    """
    
    def __init__(
        self,
        grid: ToroidalGrid,
        grad_computer: HamiltonianGradientComputer,
        dt: float = 0.01
    ):
        self.grid = grid
        self.grad_computer = grad_computer
        self.dt = dt
        
        # State for time derivatives
        self.H_prev: Optional[float] = None
        
        # Grid spacing
        self.dV = grid.dr * grid.dtheta  # Volume element (2D)
        
    def compute_observation(
        self,
        psi: jnp.ndarray,
        phi: jnp.ndarray
    ) -> Dict[str, any]:
        """
        Compute full Hamiltonian observation.
        
        Parameters
        ----------
        psi : jnp.ndarray, shape (nr, ntheta)
            Poloidal flux
        phi : jnp.ndarray, shape (nr, ntheta)
            Stream function
            
        Returns
        -------
        obs : dict
            Hamiltonian-aware observation with keys:
            - 'hamiltonian': {'H', 'grad_psi', 'grad_phi'}
            - 'conserved': {'energy', 'helicity', 'enstrophy'}
            - 'dissipation': {'dH_dt', 'energy_drift'}
            - 'state_summary': {'grad_norm', 'max_current'}
        """
        # Use Issue #24 API for H and ∇H
        H, grad_psi, grad_phi = self.grad_computer.compute_all(psi, phi)
        
        # Compute current once (小P's optimization: avoid recomputation)
        J = laplacian_toroidal(psi, self.grid)
        
        # Conserved quantities (小P's formulas, using precomputed J)
        helicity = self._compute_helicity_from_J(psi, J)
        enstrophy = self._compute_enstrophy_from_J(J)
        
        # Dissipation rate (小P recommended: numerical)
        dH_dt = self._compute_dissipation_rate(H)
        energy_drift = abs(H - self.H_prev) / abs(H) if self.H_prev is not None else 0.0
        
        # State summary
        grad_norm = jnp.sqrt(jnp.mean(grad_psi**2 + grad_phi**2))
        max_current = jnp.max(jnp.abs(J))
        
        obs = {
            'hamiltonian': {
                'H': float(H),
                'grad_psi': grad_psi,
                'grad_phi': grad_phi,
            },
            'conserved': {
                'energy': float(H),
                'helicity': float(helicity),
                'enstrophy': float(enstrophy),
            },
            'dissipation': {
                'dH_dt': float(dH_dt),
                'energy_drift': float(energy_drift),
            },
            'state_summary': {
                'grad_norm': float(grad_norm),
                'max_current': float(max_current),
            }
        }
        
        # Update state for next call
        self.H_prev = float(H)
        
        return obs
    
    def _compute_helicity_from_J(self, psi: jnp.ndarray, J: jnp.ndarray) -> float:
        """
        Compute approximate magnetic helicity from precomputed J.
        
        小P's formula: K ≈ ∫ ψ·∇²ψ dV = ∫ ψ·J dV
        
        This is a simplified toroidal helicity.
        Not conserved in resistive MHD but slowly varying.
        
        Parameters
        ----------
        psi : jnp.ndarray
            Poloidal flux
        J : jnp.ndarray
            Precomputed current J = ∇²ψ
        """
        K = jnp.sum(psi * J) * self.dV
        return K
    
    def _compute_enstrophy_from_J(self, J: jnp.ndarray) -> float:
        """
        Compute magnetic enstrophy from precomputed J.
        
        小P's formula: Ω = ∫ J² dV
        where J = ∇²ψ (toroidal current)
        
        Physical meaning: current fluctuation magnitude.
        Dissipates in resistive MHD (dΩ/dt < 0).
        
        Parameters
        ----------
        J : jnp.ndarray
            Precomputed current J = ∇²ψ
        """
        Omega = jnp.sum(J**2) * self.dV
        return Omega
    
    def _compute_dissipation_rate(self, H: float) -> float:
        """
        Compute energy dissipation rate dH/dt.
        
        小P's recommendation: Numerical method
        dH/dt = (H - H_prev) / dt
        
        More efficient than analytical method which requires
        ∇(∇J) (2nd derivatives, expensive).
        
        Verification: Should satisfy dH/dt ≤ 0 for resistive MHD (η > 0).
        """
        if self.H_prev is None:
            return 0.0  # First step, no previous value
        
        dH_dt = (H - self.H_prev) / self.dt
        return dH_dt
    
    def reset(self):
        """Reset state (call at episode start)."""
        self.H_prev = None


class HamiltonianObservationScalar:
    """
    Scalar-only Hamiltonian observation (22D).
    
    Like HamiltonianObservation but excludes high-dimensional fields
    (grad_psi, grad_phi). Only returns scalar features suitable for
    standard RL algorithms (PPO, SAC).
    
    Observation vector (22D):
    - H (1)
    - helicity (1)
    - enstrophy (1)
    - dH_dt (1)
    - energy_drift (1)
    - grad_norm (1)
    - max_current (1)
    - psi_modes (8) - Fourier modes
    - phi_modes (8) - Fourier modes
    
    Parameters
    ----------
    grid : ToroidalGrid
    grad_computer : HamiltonianGradientComputer
    dt : float
    n_modes : int
        Number of Fourier modes to extract (default: 8)
    """
    
    def __init__(
        self,
        grid: ToroidalGrid,
        grad_computer: HamiltonianGradientComputer,
        dt: float = 0.01,
        n_modes: int = 8
    ):
        self.full_observer = HamiltonianObservation(grid, grad_computer, dt)
        self.n_modes = n_modes
        
    def compute_observation(
        self,
        psi: jnp.ndarray,
        phi: jnp.ndarray
    ) -> np.ndarray:
        """
        Compute scalar observation vector (22D).
        
        Returns
        -------
        obs : np.ndarray, shape (22,)
            Flattened observation vector
        """
        # Get full observation
        obs_dict = self.full_observer.compute_observation(psi, phi)
        
        # Extract Fourier modes
        psi_modes = self._fourier_modes(psi)
        phi_modes = self._fourier_modes(phi)
        
        # Flatten to vector
        obs_vector = np.array([
            obs_dict['hamiltonian']['H'],
            obs_dict['conserved']['helicity'],
            obs_dict['conserved']['enstrophy'],
            obs_dict['dissipation']['dH_dt'],
            obs_dict['dissipation']['energy_drift'],
            obs_dict['state_summary']['grad_norm'],
            obs_dict['state_summary']['max_current'],
            *psi_modes,
            *phi_modes,
        ], dtype=np.float32)
        
        return obs_vector
    
    def _fourier_modes(self, field: jnp.ndarray) -> np.ndarray:
        """
        Extract Fourier mode amplitudes (CORRECTED ⚛️).
        
        Returns peak amplitude for first n_modes (preserves radial structure).
        
        Fix (2026-03-24): Previous version averaged over r before FFT,
        destroying radial mode structure. Now extracts true mode amplitudes.
        """
        # 2D FFT over theta (keep radial dimension)
        fft_2d = jnp.fft.fft(field, axis=1) / field.shape[1]
        
        # Extract peak amplitude for each mode number m
        modes = []
        for m in range(self.n_modes):
            m_mode = fft_2d[:, m]  # Mode m at all radial points
            m_amp = jnp.max(jnp.abs(m_mode))  # Peak amplitude
            modes.append(m_amp)
        
        return jnp.array(modes, dtype=jnp.float32)
        # Take magnitude of first n_modes
        modes = jnp.abs(fft[:self.n_modes])
        return np.array(modes, dtype=np.float32)
    
    def reset(self):
        """Reset state."""
        self.full_observer.reset()


class ObservationNormalizer:
    """
    Online normalization for observations.
    
    Uses Welford's algorithm for running mean/std computation.
    Normalizes observations to approximately [-1, 1] range.
    
    Parameters
    ----------
    obs_dim : int
        Observation dimension
    clip : float
        Clip normalized values to [-clip, clip] (default: 10.0)
    epsilon : float
        Small constant for numerical stability (default: 1e-8)
    """
    
    def __init__(
        self,
        obs_dim: int,
        clip: float = 10.0,
        epsilon: float = 1e-8
    ):
        self.obs_dim = obs_dim
        self.clip = clip
        self.epsilon = epsilon
        
        # Running statistics
        self.mean = np.zeros(obs_dim, dtype=np.float32)
        self.var = np.ones(obs_dim, dtype=np.float32)
        self.count = 0
        
    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalize observation and update statistics.
        
        Parameters
        ----------
        obs : np.ndarray, shape (obs_dim,)
            Raw observation
            
        Returns
        -------
        obs_norm : np.ndarray, shape (obs_dim,)
            Normalized observation
        """
        # Update statistics
        self._update_stats(obs)
        
        # Normalize
        std = np.sqrt(self.var + self.epsilon)
        obs_norm = (obs - self.mean) / std
        
        # Clip
        obs_norm = np.clip(obs_norm, -self.clip, self.clip)
        
        return obs_norm
    
    def _update_stats(self, obs: np.ndarray):
        """
        Update running mean and variance (Welford's algorithm).
        
        Online update formula:
        - mean_new = mean_old + (x - mean_old) / n
        - var_new = var_old + (x - mean_old) * (x - mean_new)
        """
        self.count += 1
        delta = obs - self.mean
        self.mean += delta / self.count
        delta2 = obs - self.mean
        self.var += (delta * delta2 - self.var) / self.count
    
    def reset_stats(self):
        """Reset statistics (use carefully, breaks normalization)."""
        self.mean = np.zeros(self.obs_dim, dtype=np.float32)
        self.var = np.ones(self.obs_dim, dtype=np.float32)
        self.count = 0
