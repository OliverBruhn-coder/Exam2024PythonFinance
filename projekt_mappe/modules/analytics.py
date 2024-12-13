# modules/analytics.py
import numpy as np
from scipy.stats import lognorm, skew, kurtosis

def calculate_summary_statistics(data: np.ndarray) -> dict:
    """
    Beregner summary statistics for simulerede data.

    Args:
        data (np.ndarray): Simulerede data.

    Returns:
        dict: Summary statistics.
    """
    return {
        "mean": np.mean(data),
        "variance": np.var(data),
        "median": np.median(data),
        "std_dev": np.std(data),
        "skewness": skew(data),
        "kurtosis": kurtosis(data)
    }

def calculate_lognormal_parameters(mu_annual: float, sigma_annual: float, V0: float = 1.0) -> Tuple[float, float]:
    """
    Beregner parametre til en analytisk lognormal fordeling.

    Args:
        mu_annual (float): Årlig drift.
        sigma_annual (float): Årlig volatilitet.
        V0 (float, optional): Startværdi. Defaults to 1.0.

    Returns:
        Tuple[float, float]: Scale-parameter og sigma til lognorm.
    """
    scale = V0 * np.exp(mu_annual)
    s = sigma_annual
    return scale, s

def analytical_lognormal_pdf(x: np.ndarray, scale: float, s: float) -> np.ndarray:
    """
    Beregner den analytiske PDF for en lognormal fordeling.

    Args:
        x (np.ndarray): Værdier hvor PDF'en beregnes.
        scale (float): Scale parameter.
        s (float): Sigma parameter.

    Returns:
        np.ndarray: PDF værdier.
    """
    return lognorm.pdf(x, s=s, scale=scale)
