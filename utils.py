import os
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def compute_metrics(obs, pred):
    """Compute comprehensive prediction metrics"""
    # Remove NaN values
    mask = ~np.isnan(obs) & ~np.isnan(pred)
    obs, pred = obs[mask], pred[mask]
    
    if len(obs) == 0:
        return {
            'rho': 0,
            'R2': 0,
            'RMSE': 0,
            'MSE': 0,
            'MAE': 0,
            'MAPE': 0
        }
    
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
    
    # MAPE (with handling for zero values)
    epsilon = np.mean(np.abs(obs)) * 1e-8  # Small constant relative to data scale
    mape = np.mean(np.abs((obs - pred) / (np.maximum(np.abs(obs), epsilon)))) * 100
    
    return {
        'rho': rho,
        'R2': r2,
        'RMSE': rmse,
        'MSE': mse,
        'MAE': mae,
        'MAPE': mape
    }

def format_time_axis(ax):
    """Format time axis with better date display"""
    ax.tick_params(axis='x', rotation=45)
    
    # Get current tick locations and labels
    locs = ax.get_xticks()
    labels = [item.get_text() for item in ax.get_xticklabels()]
    
    # If dealing with datetime objects
    if any(isinstance(label, datetime) for label in labels):
        # Format dates as needed
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    
    plt.setp(ax.get_xticklabels(), ha='right')

def normalize_time_series(data):
    """Normalize time series to [0, 1] range"""
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def detect_outliers(data, threshold=3):
    """Detect outliers using z-score method"""
    z_scores = np.abs((data - np.mean(data)) / np.std(data))
    return z_scores > threshold

def interpolate_missing(data):
    """Interpolate missing values in time series"""
    return pd.Series(data).interpolate(method='linear').values