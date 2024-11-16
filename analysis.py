import pyEDM
import pandas as pd
import numpy as np
from datetime import datetime
from utils import compute_metrics, ensure_dir
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter

def prepare_data(config):
    """
    Prepare data for CCM analysis:
    1. Read data
    2. Handle datetime
    3. Optionally apply Savitzky-Golay filtering
    4. Scale numerical features
    5. Create proper indices
    """
    # Read data
    data = pd.read_csv(config['data_file'])
    
    # Convert datetime and store
    data[config['time_column']] = pd.to_datetime(data[config['time_column']])
    datetime_values = data[config['time_column']].copy()
    
    # Select required columns
    cols_to_use = config['columns_to_keep']
    data_for_analysis = data[cols_to_use].copy()
    
    # Apply Savitzky-Golay filtering if enabled
    if config.get('savitzky_golay', {}).get('enable', False):
        window_length = config['savitzky_golay'].get('window_length', 11)
        polyorder = config['savitzky_golay'].get('polyorder', 3)
        
        # Validate window_length
        if window_length % 2 == 0:
            raise ValueError("Savitzky-Golay window_length must be an odd integer.")
        
        # Apply filter to each column
        for col in cols_to_use:
            original_length = len(data_for_analysis[col])
            try:
                filtered_data = savgol_filter(data_for_analysis[col], window_length, polyorder)
                data_for_analysis[col] = filtered_data
                print(f"Applied Savitzky-Golay filter to column '{col}' with window_length={window_length} and polyorder={polyorder}.")
            except Exception as e:
                raise RuntimeError(f"Error applying Savitzky-Golay filter to column '{col}': {e}")
    
    # Scale the data
    scaler = MinMaxScaler()
    data_scaled = pd.DataFrame(
        scaler.fit_transform(data_for_analysis),
        columns=data_for_analysis.columns
    )
    
    # Create sequential index for CCM (starting from 1)
    data_scaled.insert(0, 'time', range(1, len(data_scaled) + 1))
    data_scaled.set_index('time', inplace=True)
    
    # Store original datetime
    data_scaled['datetime'] = datetime_values
    
    return data_scaled, scaler

def perform_embedding_analysis(data, config):
    """
    Perform embedding dimension analysis to find optimal E
    """
    train_size = int(len(data) * config['train_size_ratio'])
    max_E = 10  # Maximum embedding dimension to test
    
    embedding_results = []
    for col in config['columns_to_keep']:
        rho_values = []
        for E in range(1, max_E + 1):
            simplex_out = pyEDM.Simplex(
                dataFrame=data,
                lib=f"1 {train_size}",
                pred=f"{train_size+1} {len(data)}",
                E=E,
                columns=col,
                target=col
            )
            metrics = compute_metrics(
                simplex_out['Observations'],
                simplex_out['Predictions']
            )
            rho_values.append(metrics['rho'])
        
        optimal_E = np.argmax(rho_values) + 1
        embedding_results.append({
            'variable': col,
            'optimal_E': optimal_E,
            'rho_values': rho_values
        })
    
    return embedding_results

def perform_ccm_analysis(data, config, E=None):
    """
    Perform CCM analysis using either specified E or E = vars + 1
    """
    train_size = int(len(data) * config['train_size_ratio'])
    if E is None:
        E = len(config['columns_to_keep']) + 1

    lib = f"1 {train_size}"
    pred = f"{train_size+1} {len(data)}"

    # Initialize storage
    columns = config['columns_to_keep']
    N = len(columns)
    CM_matrix = np.zeros((N, N))
    all_metrics = {}
    predictions = {}

    # Use Multiview for multivariate analysis
    multiview_output = pyEDM.Multiview(
        dataFrame=data,
        columns=','.join(columns),
        target=columns,
        lib=lib,
        pred=pred,
        E=E,
        D=len(columns),
        multiview=int(np.sqrt(len(columns))),
        verbose=True
    )

    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            if i != j:
                key = f"{col1}->{col2}"
                CM_matrix[i, j] = multiview_output['View'].loc[i, 'rho']
                all_metrics[key] = multiview_output['View'].loc[i]
                predictions[key] = multiview_output['Predictions']

                # Add datetime to predictions (only for the prediction period)
                predictions[key]['DateTime'] = data['datetime'].iloc[train_size:].reset_index(drop=True)

    return CM_matrix, all_metrics, predictions

def find_strongest_couplings(CM_matrix, columns):
    """
    Find strongest causal relationships for each target variable
    """
    strongest_couplings = {}
    causality_strengths = []
    
    for j, target in enumerate(columns):
        target_correlations = CM_matrix[:, j]
        max_idx = np.argmax(target_correlations)
        predictor = columns[max_idx]
        
        if predictor != target:
            coupling = f"{predictor}->{target}"
            strength = target_correlations[max_idx]
            
            strongest_couplings[target] = coupling
            causality_strengths.append({
                'target': target,
                'predictor': predictor,
                'strength': strength
            })
    
    # Sort by strength
    causality_strengths.sort(key=lambda x: x['strength'], reverse=True)
    
    return strongest_couplings, causality_strengths

def analyze_prediction_performance(all_metrics):
    """
    Analyze and summarize prediction performance
    """
    performance_summary = pd.DataFrame(all_metrics).T
    
    best_predictions = {
        'rho': performance_summary.loc[performance_summary['rho'].idxmax()],
        'R2': performance_summary.loc[performance_summary['R2'].idxmax()],
        'RMSE': performance_summary.loc[performance_summary['RMSE'].idxmin()]
    }
    
    worst_predictions = {
        'rho': performance_summary.loc[performance_summary['rho'].idxmin()],
        'R2': performance_summary.loc[performance_summary['R2'].idxmin()],
        'RMSE': performance_summary.loc[performance_summary['RMSE'].idxmax()]
    }
    
    return {
        'summary_stats': performance_summary.describe(),
        'best_predictions': best_predictions,
        'worst_predictions': worst_predictions
    }

def save_analysis_results(results, config):
    """
    Save analysis results to files
    """
    results_dir = config['results_dir']
    ensure_dir(results_dir)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(results['all_metrics']).T
    metrics_df.to_csv(f"{results_dir}/{config['output_prefix']}_metrics.csv")
    
    # Save causality strengths
    causality_df = pd.DataFrame(results['causality_strengths'])
    causality_df.to_csv(f"{results_dir}/{config['output_prefix']}_causality.csv")
    
    # Save prediction performance summary
    with open(f"{results_dir}/{config['output_prefix']}_summary.txt", 'w') as f:
        f.write("=== CCM Analysis Summary ===\n\n")
        f.write("Strongest Causal Relationships:\n")
        for strength in results['causality_strengths']:
            f.write(f"{strength['predictor']} -> {strength['target']}: {strength['strength']:.3f}\n")
        
        f.write("\nBest Predictions:\n")
        for metric, values in results['performance_analysis']['best_predictions'].items():
            f.write(f"Best {metric}: {values.name} ({metric}={values[metric]:.3f})\n")
        
        f.write("\nWorst Predictions:\n")
        for metric, values in results['performance_analysis']['worst_predictions'].items():
            f.write(f"Worst {metric}: {values.name} ({metric}={values[metric]:.3f})\n")

def run_full_analysis(config):
    """
    Run complete CCM analysis pipeline
    """
    # Prepare data
    data, scaler = prepare_data(config)
    
    # Perform embedding analysis
    embedding_results = perform_embedding_analysis(data, config)
    
    # Perform CCM analysis
    CM_matrix, all_metrics, predictions = perform_ccm_analysis(data, config)
    
    # Find strongest couplings
    strongest_couplings, causality_strengths = find_strongest_couplings(
        CM_matrix, config['columns_to_keep']
    )
    
    # Analyze prediction performance
    performance_analysis = analyze_prediction_performance(all_metrics)
    
    # Compile results
    results = {
        'data': data,
        'scaler': scaler,
        'embedding_results': embedding_results,
        'CM_matrix': CM_matrix,
        'all_metrics': all_metrics,
        'predictions': predictions,
        'strongest_couplings': strongest_couplings,
        'causality_strengths': causality_strengths,
        'performance_analysis': performance_analysis
    }
    
    # Save results
    save_analysis_results(results, config)
    
    return results