# modules/data_loading.py
import pandas as pd

def load_data(covariance_path: str, init_values_path: str) -> (pd.DataFrame, pd.DataFrame):
    """
    Indlæser covariance matrix og initiale værdier fra Excel-filer.

    Args:
        covariance_path (str): Sti til covariance matrix Excel-fil.
        init_values_path (str): Sti til initiale værdier Excel-fil.

    Returns:
        tuple: DataFrames for covariance matrix og initiale værdier.
    """
    covariance_matrix = pd.read_excel(covariance_path, index_col=0)
    init_values = pd.read_excel(init_values_path, index_col=0)
    return covariance_matrix, init_values
