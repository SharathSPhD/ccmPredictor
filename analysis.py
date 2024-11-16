# analysis.py

import numpy as np
import pandas as pd
import pyEDM
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from utils import ensure_dir, compute_metrics
import os
import pickle

def prepare_data_for_analysis(data, config):
    """Prepare data for CCM analysis"""
    # Create a copy of required columns
    analysis_data = data[config['columns_to_keep']].copy()
    
    # Convert datetime if present
    if 'datetime' in data.columns:
        analysis_data['datetime'] = pd.to_datetime(data['datetime'])
    
    # Ensure data is numeric and handle any missing values
    numeric_columns = [col for col in analysis_data.columns if col != 'datetime']
    analysis_data[numeric_columns] = analysis_data[numeric_columns].astype(float)
    analysis_data[numeric_columns] = analysis_data[numeric_columns].ffill()
    
    # Apply Savitzky-Golay filter if configured
    if config.get('savitzky_golay', {}).get('enable', False):
        from scipy.signal import savgol_filter
        window_length = config['savitzky_golay']['window_length']
        polyorder = config['savitzky_golay']['polyorder']
        
        for col in numeric_columns:
            filtered_values = savgol_filter(
                analysis_data[col].values,
                window_length=window_length,
                polyorder=polyorder
            )
            analysis_data[col] = filtered_values
            print(f"Applied Savitzky-Golay filter to column '{col}' with window_length={window_length} and polyorder={polyorder}.")
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(
        scaler.fit_transform(analysis_data[numeric_columns]),
        columns=numeric_columns,
        index=analysis_data.index
    )
    
    # Add datetime back if it exists
    if 'datetime' in analysis_data.columns:
        scaled_data['datetime'] = analysis_data['datetime']
    
    print("Final columns in prepared data:", scaled_data.columns.tolist())
    return scaled_data, scaler

def perform_ccm_analysis(data, config):
    """Perform CCM analysis using pyEDM Simplex"""
    columns = config['columns_to_keep']
    
    # Create a copy without datetime for EDM analysis
    data_edm = data[columns].copy()
    data_edm.index = range(len(data_edm))
    
    # Add time column required by pyEDM
    data_edm['time'] = data_edm.index
    
    # Initialize results matrices
    n_vars = len(columns)
    CM_matrix = np.zeros((n_vars, n_vars))
    all_metrics = {}
    predictions = {}
    
    # Calculate library and prediction sets
    total_points = len(data_edm)
    train_size = int(total_points * config.get('train_size_ratio', 0.6))
    lib = f"1 {train_size}"
    pred = f"{train_size + 1} {total_points}"
    
    ccm_params = config.get('ccm', {})
    E = ccm_params.get('embedding_dimension', 3)
    tau = ccm_params.get('tau', 1)
    knn = ccm_params.get('k_nearest', E + 1)
    
    # Perform pairwise CCM analysis
    for i, target in enumerate(columns):
        for j, predictor in enumerate(columns):
            if i != j:  # Skip self-predictions
                key = f"{predictor}->{target}"
                try:
                    print(f"\nAnalyzing {key}...")
                    print(f"Data shape: {data_edm.shape}")
                    print(f"Columns: {data_edm.columns.tolist()}")
                    
                    # Try CCM analysis
                    simplex_output = pyEDM.Simplex(
                        dataFrame=data_edm,
                        lib=lib,
                        pred=pred,
                        E=E,
                        tau=tau,
                        knn=knn,
                        columns=predictor,
                        target=target,
                        embedded=False,
                        noTime=False,
                        verbose=True
                    )
                    
                    print(f"Simplex output columns: {simplex_output.columns.tolist()}")
                    
                    # Extract metrics
                    metrics = {}
                    
                    if 'rho' in simplex_output.columns:
                        metrics['rho'] = float(simplex_output['rho'].iloc[0]) if not pd.isna(simplex_output['rho'].iloc[0]) else 0
                    else:
                        metrics['rho'] = 0
                        
                    if 'MAE' in simplex_output.columns:
                        metrics['mae'] = float(simplex_output['MAE'].iloc[0]) if not pd.isna(simplex_output['MAE'].iloc[0]) else 0
                    else:
                        metrics['mae'] = 0
                        
                    if 'RMSE' in simplex_output.columns:
                        metrics['rmse'] = float(simplex_output['RMSE'].iloc[0]) if not pd.isna(simplex_output['RMSE'].iloc[0]) else 0
                    else:
                        metrics['rmse'] = 0
                    
                    # Add additional metrics
                    if 'Predictions' in simplex_output.columns:
                        pred_values = simplex_output['Predictions'].values
                        obs_values = data_edm[target].values[train_size:]
                        additional_metrics = compute_metrics(obs_values, pred_values)
                        metrics.update(additional_metrics)
                    else:
                        metrics.update({
                            'R2': 0,
                            'MSE': 0,
                            'MAPE': 0
                        })
                    
                    CM_matrix[i, j] = metrics['rho']
                    all_metrics[key] = metrics
                    
                    # Create predictions DataFrame
                    pred_df = pd.DataFrame()
                    if 'datetime' in data.columns:
                        pred_df['datetime'] = data['datetime']
                    else:
                        pred_df['datetime'] = pd.Series(range(len(data)))
                        
                    pred_df['Observations'] = data[target]
                    pred_df['Predictions'] = np.nan  # Initialize with NaN
                    
                    if 'Predictions' in simplex_output.columns:
                        pred_df.loc[train_size:, 'Predictions'] = simplex_output['Predictions'].values
                    
                    predictions[key] = pred_df
                    
                    print(f"Successfully analyzed {key} with rho: {metrics['rho']:.3f}")
                    
                except Exception as e:
                    print(f"Error in CCM analysis for {key}: {str(e)}")
                    print(f"Full error details:")
                    import traceback
                    traceback.print_exc()
                    CM_matrix[i, j] = 0
                    all_metrics[key] = {'rho': 0, 'mae': 0, 'rmse': 0, 'R2': 0, 'MSE': 0, 'MAPE': 0}
                    predictions[key] = pd.DataFrame(columns=['datetime', 'Observations', 'Predictions'])
    
    return CM_matrix, all_metrics, predictions

def analyze_full_multiview(data, config):
    """Perform full multiview analysis for all variables together"""
    columns = config['columns_to_keep']
    
    # Create a copy without datetime for EDM analysis
    data_edm = data[columns].copy()
    data_edm.index = range(len(data_edm))
    
    # Add time column
    data_edm['time'] = data_edm.index
    
    # Calculate library and prediction sets
    total_points = len(data_edm)
    train_size = int(total_points * config.get('train_size_ratio', 0.6))
    lib = f"1 {train_size}"
    pred = f"{train_size + 1} {total_points}"
    
    ccm_params = config.get('ccm', {})
    E = ccm_params.get('embedding_dimension', 3)
    tau = ccm_params.get('tau', 1)
    
    # For each target variable, try to predict it using all other variables
    multiview_results = {}
    for target in columns:
        try:
            print(f"\nPerforming Multiview analysis for target: {target}")
            print(f"Data shape: {data_edm.shape}")
            print(f"Columns: {data_edm.columns.tolist()}")
            
            # Use all other columns as predictors
            predictor_cols = [col for col in columns if col != target]
            
            mv_output = pyEDM.Multiview(
                dataFrame=data_edm,
                lib=lib,
                pred=pred,
                E=E,
                tau=tau,
                columns=','.join(predictor_cols),
                target=target,
                D=len(predictor_cols),
                multiview=int(max(1, np.sqrt(len(predictor_cols)))),
                exclusionRadius=ccm_params.get('exclusion_radius', 0),
                trainLib=False,
                verbose=True
            )
            
            multiview_results[target] = mv_output
            print(f"Successfully completed Multiview analysis for {target}")
            
        except Exception as e:
            print(f"Error in Multiview analysis for target {target}: {str(e)}")
            print("Full error details:")
            import traceback
            traceback.print_exc()
            multiview_results[target] = None
    
    return multiview_results

def analyze_causality_strengths(CM_matrix, columns):
    """Analyze causality strengths from CCM results"""
    causality_strengths = {}
    strongest_couplings = {}
    
    for i, target in enumerate(columns):
        max_strength = -1
        best_predictor = None
        
        for j, predictor in enumerate(columns):
            if i != j:
                strength = CM_matrix[j, i]  # Note the index order
                causality_strengths[f"{predictor}->{target}"] = strength
                
                if strength > max_strength:
                    max_strength = strength
                    best_predictor = predictor
        
        if best_predictor:
            strongest_couplings[target] = {
                'predictor': best_predictor,
                'strength': max_strength,
                'coupling': f"{best_predictor}->{target}"
            }
    
    return causality_strengths, strongest_couplings

def analyze_prediction_performance(all_metrics):
    """Analyze prediction performance metrics"""
    performance = {
        'best_rho': max((m['rho'] for m in all_metrics.values()), default=0),
        'mean_rho': np.mean([m['rho'] for m in all_metrics.values()]),
        'mean_rmse': np.mean([m['rmse'] for m in all_metrics.values()]),
        'mean_mae': np.mean([m['mae'] for m in all_metrics.values()]),
        'mean_r2': np.mean([m.get('R2', 0) for m in all_metrics.values()]),
        'mean_mape': np.mean([m.get('MAPE', 0) for m in all_metrics.values()]),
    }
    
    # Find best coupling based on overall performance
    best_coupling = max(all_metrics.items(), 
                       key=lambda x: (x[1]['rho'] + x[1].get('R2', 0))/2)
    performance['best_coupling'] = best_coupling[0]
    performance['best_coupling_metrics'] = best_coupling[1]
    
    return performance

def run_full_analysis(config):
    """Run complete CCM analysis pipeline"""
    # Load and prepare data
    data = pd.read_csv(config['data_file'])
    
    # Prepare data for analysis
    prepared_data, scaler = prepare_data_for_analysis(data, config)
    
    # Perform pairwise CCM analysis
    CM_matrix, all_metrics, predictions = perform_ccm_analysis(prepared_data, config)
    
    # Optionally perform full multiview analysis
    multiview_results = analyze_full_multiview(prepared_data, config)
    
    # Analyze results
    causality_strengths, strongest_couplings = analyze_causality_strengths(
        CM_matrix, config['columns_to_keep'])
    performance_analysis = analyze_prediction_performance(all_metrics)
    
    # Compile results
    results = {
        'data': prepared_data,
        'scaler': scaler,
        'CM_matrix': CM_matrix,
        'all_metrics': all_metrics,
        'predictions': predictions,
        'strongest_couplings': strongest_couplings,
        'causality_strengths': causality_strengths,
        'performance_analysis': performance_analysis,
        'multiview_results': multiview_results
    }
    
    # Save results
    save_analysis_results(results, config)
    
    return results

def save_analysis_results(results, config):
    """Save analysis results to disk"""
    ensure_dir(config['results_dir'])
    
    # Save full results to pickle
    results_file = os.path.join(config['results_dir'], 
                               f"{config['output_prefix']}_results.pkl")
    
    # Create a copy of results without the multiview_results (which might not pickle well)
    results_to_save = results.copy()
    if 'multiview_results' in results_to_save:
        del results_to_save['multiview_results']
    
    with open(results_file, 'wb') as f:
        pickle.dump(results_to_save, f)
    
    # Save summary to text file
    summary_file = os.path.join(config['results_dir'],
                               f"{config['output_prefix']}_summary.txt")
    
    with open(summary_file, 'w') as f:
        f.write("CCM Analysis Summary\n")
        f.write("===================\n\n")
        
        f.write("Performance Metrics:\n")
        f.write("-----------------\n")
        for metric, value in results['performance_analysis'].items():
            if isinstance(value, dict):
                f.write(f"{metric}:\n")
                for k, v in value.items():
                    if isinstance(v, (int, float)):
                        f.write(f"  {k}: {v:.3f}\n")
                    else:
                        f.write(f"  {k}: {v}\n")
            elif isinstance(value, (int, float)):
                f.write(f"{metric}: {value:.3f}\n")
            else:
                f.write(f"{metric}: {value}\n")
        
        f.write("\nStrongest Causal Relationships:\n")
        f.write("---------------------------\n")
        for target, info in results['strongest_couplings'].items():
            if isinstance(info, dict):  # If info is a dictionary
                strength = info.get('strength', 0)
                predictor = info.get('predictor', 'unknown')
                f.write(f"{target} is best predicted by {predictor} ")
                f.write(f"(strength: {strength:.3f})\n")
            else:  # If info is just the coupling string
                f.write(f"{target}: {info}\n")