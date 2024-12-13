# modules/simulations.py
import numpy as np
from typing import Tuple
import pandas as pd

class SimulationModel:
    def __init__(self, mu: np.ndarray, sigma: np.ndarray, initial_values: np.ndarray, seed: int = None):
        """
        Initialiserer simulationsmodellen.

        Args:
            mu (np.ndarray): Driftvektor.
            sigma (np.ndarray): Kovariansmatrix.
            initial_values (np.ndarray): Initiale værdier.
            seed (int, optional): Random seed for reproducerbarhed.
        """
        self.mu = mu
        self.sigma = sigma
        self.initial_values = initial_values
        if seed is not None:
            np.random.seed(seed)
    
    def simulate_paths(self, n_steps: int, n_simulations: int, delta_t: float) -> np.ndarray:
        """
        Simulerer paths for aktiverne over tid.

        Args:
            n_steps (int): Antal tidssteg.
            n_simulations (int): Antal simuleringer.
            delta_t (float): Tidstrin størrelse.

        Returns:
            np.ndarray: Simulerede paths med form (n_simulations, n_assets, n_steps + 1).
        """
        n_assets = len(self.initial_values)
        X_t = np.zeros((n_simulations, n_assets, n_steps + 1))
        X_t[:, :, 0] = self.initial_values  # Initiale værdier

        for t in range(1, n_steps + 1):
            delta_X_t = np.random.multivariate_normal(self.mu * delta_t, self.sigma * delta_t, size=n_simulations)
            X_t[:, :, t] = X_t[:, :, t - 1] + delta_X_t
        
        return X_t

    def simulate_log_FX(self, X_t: np.ndarray) -> np.ndarray:
        """
        Ekstraherer log(FX_t) fra simulerede paths.

        Args:
            X_t (np.ndarray): Simulerede paths.

        Returns:
            np.ndarray: log(FX_t) værdier.
        """
        return X_t[:, 0, :]
