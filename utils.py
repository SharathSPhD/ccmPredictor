# utils.py
import os
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def compute_metrics(obs, pred):
    """
    Compute comprehensive prediction metrics
    """
    # Remove NaN values
    mask = ~np.isnan(obs) & ~np.isnan(pred)
    obs, pred = obs[mask], pred[mask]
    
    # Correlation coefficient
    rho = np.corrcoef(obs, pred)[0, 1]
    
    # R-squared
    r2 = r2_score(obs, pred)
    
    # RMSE
    rmse = np.sqrt(mean_squared_error(obs, pred))
    
    # MSE
    mse = mean_squared_error(obs, pred)
    
    # MAE
    mae = mean_absolute_error(obs, pred)
    
    # MAPE
    epsilon = 1e-10  # Small constant to avoid division by zero
    mape = np.mean(np.abs((obs - pred) / (obs + epsilon))) * 100
    
    return {
        'rho': rho,
        'R2': r2,
        'RMSE': rmse,
        'MSE': mse,
        'MAE': mae,
        'MAPE': mape
    }

def format_time_axis(ax, rotation=45):
    """Format time axis for better readability"""
    ax.tick_params(axis='x', rotation=rotation)
    ax.grid(True)