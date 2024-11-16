import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from utils import ensure_dir, format_time_axis

def plot_time_series(data, config, original_data=None):
    """Plot individual time series for each variable"""
    ensure_dir(config['results_dir'])
    
    for col in config['columns_to_keep']:
        plt.figure(figsize=(15, 5))
        
        # Plot processed data
        if 'datetime' in data.columns:
            plt.plot(data['datetime'], data[col], 'b-', label='Processed', linewidth=1)
        else:
            plt.plot(range(len(data)), data[col], 'b-', label='Processed', linewidth=1)
        
        # Plot original data if available
        if original_data is not None:
            time_col = config.get('time_column', 'datetime')
            if time_col in original_data.columns:
                plt.plot(original_data[time_col], original_data[col], 
                        'g--', label='Original', alpha=0.5, linewidth=0.5)
            else:
                plt.plot(range(len(original_data)), original_data[col], 
                        'g--', label='Original', alpha=0.5, linewidth=0.5)
        
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(f'Time Series: {col}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(rotation=45)
        format_time_axis(plt.gca())
        plt.tight_layout()
        
        plt.savefig(os.path.join(config['results_dir'], f'timeseries_{col}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

def plot_ccm_results(results, config):
    """Plot CCM analysis results"""
    ensure_dir(config['results_dir'])
    
    # Create heatmap of causality matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(results['CM_matrix'], 
                annot=True, 
                fmt='.2f',
                xticklabels=config['columns_to_keep'],
                yticklabels=config['columns_to_keep'],
                cmap='viridis',
                center=0)
    plt.title('Cross Convergent Mapping Strength')
    plt.xlabel('Target Variable')
    plt.ylabel('Predictor Variable')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['results_dir'], 
                            f"{config['output_prefix']}_ccm_heatmap.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot predictions for strongest couplings
    for coupling, pred_df in results['predictions'].items():
        if not pred_df.empty:
            plt.figure(figsize=(15, 5))
            
            plt.plot(pred_df['datetime'], pred_df['Observations'], 
                    'k.', label='Actual', alpha=0.5, markersize=2)
            plt.plot(pred_df['datetime'], pred_df['Predictions'], 
                    'r-', label='Predicted', linewidth=1)
            
            plt.title(f'Predictions for {coupling}')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            format_time_axis(plt.gca())
            
            plt.tight_layout()
            plt.savefig(os.path.join(config['results_dir'], 
                                   f"{config['output_prefix']}_predictions_{coupling.replace('->', '_to_')}.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()

def plot_multiview_results(results, config):
    """Plot multiview analysis results"""
    ensure_dir(config['results_dir'])
    
    multiview_results = results.get('multiview_results', {})
    if not multiview_results:
        print("No multiview results to plot")
        return
    
    for target, mv_result in multiview_results.items():
        try:
            if mv_result is None:
                print(f"Skipping plot for {target}: No valid results")
                continue
                
            # Check if we have the required attributes and data
            if not hasattr(mv_result, 'topRankProjections'):
                print(f"Skipping plot for {target}: No topRankProjections available")
                continue
                
            if not mv_result.topRankProjections:
                print(f"Skipping plot for {target}: Empty topRankProjections")
                continue
            
            plt.figure(figsize=(15, 5))
            
            # Extract predictions
            predictions = mv_result.topRankProjections[0]['Predictions'].values
            observations = results['data'][target]
            dates = results['data']['datetime'] if 'datetime' in results['data'] else range(len(observations))
            
            plt.plot(dates, observations, 'k.', label='Actual', alpha=0.5, markersize=2)
            plt.plot(dates, predictions, 'r-', label='Multiview Prediction', linewidth=1)
            
            plt.title(f'Multiview Predictions for {target}')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            format_time_axis(plt.gca())
            
            plt.tight_layout()
            plt.savefig(os.path.join(config['results_dir'], 
                                   f"{config['output_prefix']}_multiview_{target}.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error plotting multiview results for {target}: {str(e)}")
            import traceback
            traceback.print_exc()

def format_time_axis(ax):
    """Format time axis for better readability"""
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), ha='right')