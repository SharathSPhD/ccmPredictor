import json
from analysis import run_full_analysis
from plotting import plot_time_series, plot_ccm_results, plot_embedding_analysis
import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Run analysis
    results = run_full_analysis(config)
    
    # Optionally load original data for plotting
    if config.get('savitzky_golay', {}).get('enable', False):
        original_data = pd.read_csv(config['data_file'])
        original_data[config['time_column']] = pd.to_datetime(original_data[config['time_column']])
        # Include 'datetime' in the original_data
        original_data = original_data[[config['time_column']] + config['columns_to_keep']].copy()
    else:
        original_data = None
    
    # Create plots
    plot_time_series(results['data'], config, original_data)
    plot_ccm_results(results, config)
    plot_embedding_analysis(results['embedding_results'], config)
    
    # Print summary using logging
    logging.info("\nAnalysis Complete!")
    logging.info(f"Results saved to: {config['results_dir']}")
    logging.info("\nStrongest Causal Relationships:")
    for strength in results['causality_strengths']:
        logging.info(f"{strength['predictor']} -> {strength['target']}: {strength['strength']:.3f}")

if __name__ == "__main__":
    main()
