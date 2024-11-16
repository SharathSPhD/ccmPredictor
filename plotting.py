# plotting.py

import matplotlib.pyplot as plt
import os
import matplotlib
from utils import format_time_axis, ensure_dir
matplotlib.use('Agg')  # Use a non-interactive backend

def plot_time_series(data, config, original_data=None):
    """Plot individual time series for each variable with optional original data"""
    ensure_dir(config['results_dir'])
    
    for col in config['columns_to_keep']:
        plt.figure(figsize=(15, 5))
        
        # Plot filtered data
        plt.plot(data['datetime'], data[col], 'b-', label='Filtered')
        
        # Plot original data if available
        if original_data is not None and config['time_column'] in original_data.columns:
            plt.plot(original_data[config['time_column']], original_data[col], 'g--', label='Original')
        
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(f'Time Series: {col}')
        plt.grid(True)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(config['results_dir'], f'timeseries_{col}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

def plot_ccm_results(results, config):
    """Plot CCM analysis results"""
    num_plots = len(results['strongest_couplings']) + 1
    nrows = (num_plots + 1) // 2
    ncols = 2

    fig = plt.figure(figsize=(15, 7*nrows))

    # Plot cross mapping matrix
    ax = fig.add_subplot(nrows, ncols, 1)
    im = ax.imshow(results['CM_matrix'], cmap='viridis')
    plt.colorbar(im, ax=ax, label='ρ (Prediction Skill)')

    # Set labels
    ax.set_xticks(range(len(config['columns_to_keep'])))
    ax.set_yticks(range(len(config['columns_to_keep'])))
    ax.set_xticklabels(config['columns_to_keep'], rotation=45)
    ax.set_yticklabels(config['columns_to_keep'])
    ax.set_xlabel('Target Variable')
    ax.set_ylabel('Predictor Variable')
    ax.set_title('Cross Mapping Analysis')

    # Add correlation values
    for i in range(len(config['columns_to_keep'])):
        for j in range(len(config['columns_to_keep'])):
            ax.text(j, i, f'{results["CM_matrix"][i,j]:.2f}', 
                    ha='center', va='center',
                    color='white' if results["CM_matrix"][i,j] > 0.5 else 'black')

    # Plot predictions for strongest couplings
    for idx, (target, coupling) in enumerate(results['strongest_couplings'].items(), 2):
        ax = fig.add_subplot(nrows, ncols, idx)
        pred_df = results['predictions'][coupling]
        
        ax.plot(pred_df['DateTime'], pred_df['Observations'], 'k.', 
                label='Actual', alpha=0.5)
        ax.plot(pred_df['DateTime'], pred_df['Predictions'], 'r-', 
                label='Predicted')

        # Add vertical line to separate training and prediction periods
        ax.axvline(x=pred_df['DateTime'].iloc[0], color='gray', linestyle='--', label='Prediction Start')

        ax.set_xlabel('Time')
        ax.set_ylabel(target)
        ax.set_title(f'Best Prediction for {target}\n({coupling})')
        format_time_axis(ax)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(config['results_dir'], 
                            f"{config['output_prefix']}.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_embedding_analysis(embedding_results, config):
    """Plot embedding dimension analysis results"""
    plt.figure(figsize=(12, 6))
    
    for result in embedding_results:
        plt.plot(range(1, len(result['rho_values']) + 1), 
                result['rho_values'], 
                'o-', 
                label=result['variable'])
        
        # Mark optimal E
        plt.plot(result['optimal_E'], 
                result['rho_values'][result['optimal_E']-1], 
                'r*', 
                markersize=15)
    
    plt.xlabel('Embedding Dimension (E)')
    plt.ylabel('Prediction Skill (ρ)')
    plt.title('Embedding Dimension Analysis')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(config['results_dir'], 
                            f"{config['output_prefix']}_embedding.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
