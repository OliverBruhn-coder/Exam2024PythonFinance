# main.py
import yaml
import numpy as np
from modules.data_loading import load_data
from modules.simulations import SimulationModel
from modules.analytics import calculate_summary_statistics, calculate_lognormal_parameters, analytical_lognormal_pdf
from modules.visualizations import Visualization
import pandas as pd

def main():
    # Indlæs konfiguration
    with open("config/config.yaml", 'r') as file:
        config = yaml.safe_load(file)
    
    # Load data
    covariance_path = config['data_files']['covariance_matrix']
    init_values_path = config['data_files']['init_values']
    covariance_matrix, init_values = load_data(covariance_path, init_values_path)
    
    # Parameters
    mu = np.array(config['simulation_parameters']['mu'] + [0]*(len(covariance_matrix) - len(config['simulation_parameters']['mu'])))
    sigma = covariance_matrix.values
    initial_values = init_values.values.flatten()
    seed = config.get('random_seed', None)
    
    # Initialize model
    model = SimulationModel(mu=mu, sigma=sigma, initial_values=initial_values, seed=seed)
    
    # Run simulation
    n_steps = config['simulation_parameters']['n_steps']
    n_simulations = config['simulation_parameters']['n_simulations']
    delta_t = config['simulation_parameters']['delta_t']
    X_t = model.simulate_paths(n_steps=n_steps, n_simulations=n_simulations, delta_t=delta_t)
    
    # Extract log(FX_t)
    log_FX_t = model.simulate_log_FX(X_t)
    
    # Visualize simulated paths
    Visualization.plot_simulated_paths(log_FX_t, n_simulations=n_simulations, n_steps=n_steps)
    
    # Example for Question 2: Simulation og Analyse af V_US_local
    # Antag at V_US_local er den 2. aktie (index 1)
    V_US_local_simulated = X_t[:, 1, -1]  # Værdier ved slutningen
    
    # Beregn analytiske parametre
    mu_log_annual = mu[1] * 52
    sigma_log_annual = np.sqrt(sigma[1, 1])
    scale, s = calculate_lognormal_parameters(mu_annual=mu_log_annual, sigma_annual=sigma_log_annual, V0=1)
    
    # Beregn analytisk PDF
    x_values = np.linspace(min(V_US_local_simulated), max(V_US_local_simulated), 1000)
    pdf_values = analytical_lognormal_pdf(x=x_values, scale=scale, s=s)
    
    # Visualiser distribution
    Visualization.plot_distribution(
        simulated=V_US_local_simulated,
        analytical_pdf=pdf_values,
        x_values=x_values,
        title="Comparison of Simulated and Analytical Distributions of V_US_local",
        xlabel="V_US_local"
    )
    
    # Beregn summary statistics
    stats = calculate_summary_statistics(V_US_local_simulated)
    analytical_mean = scale * np.exp((s**2)/2)
    analytical_variance = (scale**2) * (np.exp(s**2) - 1) * np.exp(s**2)
    
    print("Summary Statistics for Simulated V_US_local:")
    for key, value in stats.items():
        print(f"{key.capitalize()}: {value:.4f}")
    print(f"Analytical Mean: {analytical_mean:.4f}")
    print(f"Analytical Variance: {analytical_variance:.4f}")
    
    # Yderligere analyser kan implementeres på lignende måde

if __name__ == "__main__":
    main()
