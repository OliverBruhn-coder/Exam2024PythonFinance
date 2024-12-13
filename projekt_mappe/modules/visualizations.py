# modules/visualizations.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

class Visualization:
    @staticmethod
    def plot_simulated_paths(log_FX_t: np.ndarray, n_simulations: int, n_steps: int):
        """
        Visualiserer simulerede paths for log(FX_t).

        Args:
            log_FX_t (np.ndarray): log(FX_t) værdier.
            n_simulations (int): Antal simuleringer.
            n_steps (int): Antal tidssteg.
        """
        plt.figure(figsize=(10, 6))
        for i in range(n_simulations):
            plt.plot(range(n_steps + 1), log_FX_t[i, :], alpha=0.1)
        plt.title("Simulated Evolution of log(FX_t)")
        plt.xlabel("Weeks")
        plt.ylabel("log(FX_t)")
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def plot_distribution(simulated: np.ndarray, analytical_pdf: np.ndarray, x_values: np.ndarray, title: str, xlabel: str):
        """
        Plotter histogram af simulerede data sammenlignet med analytisk PDF.

        Args:
            simulated (np.ndarray): Simulerede data.
            analytical_pdf (np.ndarray): Analytisk PDF værdier.
            x_values (np.ndarray): Værdier hvor PDF'en beregnes.
            title (str): Plot titel.
            xlabel (str): X-akse label.
        """
        plt.figure(figsize=(10, 6))
        plt.hist(simulated, bins=100, density=True, color='navy', alpha=0.6, label='Simulerede data')
        plt.plot(x_values, analytical_pdf, color='darkred', linewidth=2, label='Analytisk PDF')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def plot_heatmap(correlation_matrix: pd.DataFrame, title: str):
        """
        Plotter en korrelations heatmap.

        Args:
            correlation_matrix (pd.DataFrame): Korrelationsmatrix.
            title (str): Plot titel.
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title(title)
        plt.show()
    
    @staticmethod
    def plot_pairwise_scatter(df: pd.DataFrame, title: str):
        """
        Plotter pairwise scatter plots.

        Args:
            df (pd.DataFrame): DataFrame med data.
            title (str): Plot titel.
        """
        sns.pairplot(df)
        plt.suptitle(title, y=1.02)
        plt.show()
