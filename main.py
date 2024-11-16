import json
import logging
import pandas as pd
from analysis import run_full_analysis
from plotting import (plot_time_series, plot_ccm_results, 
                     plot_multiview_results)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ccm_analysis.log')
    ]
)

def main():
    """Main execution function"""
    try:
        # Load configuration
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        # Load original data
        logging.info("Loading data from %s", config['data_file'])
        original_data = pd.read_csv(config['data_file'])
        if config['time_column'] in original_data.columns:
            original_data[config['time_column']] = pd.to_datetime(original_data[config['time_column']])
        
        # Run analysis
        logging.info("Starting CCM analysis...")
        results = run_full_analysis(config)
        
        # Create plots
        logging.info("Generating visualizations...")
        plot_time_series(results['data'], config, original_data)
        plot_ccm_results(results, config)
        plot_multiview_results(results, config)
        
        # Print summary statistics
        logging.info("\nAnalysis Complete!")
        logging.info("Results saved to: %s", config['results_dir'])
        
        if results['causality_strengths']:
            logging.info("\nPairwise Causal Relationships:")
            for key, strength in results['causality_strengths'].items():
                logging.info("%s: %.3f", key, strength)
        
        logging.info("\nOverall Performance Metrics:")
        for metric, value in results['performance_analysis'].items():
            if isinstance(value, dict):
                logging.info("%s:", metric)
                for k, v in value.items():
                    if isinstance(v, float):
                        logging.info("  %s: %.3f", k, v)
                    else:
                        logging.info("  %s: %s", k, v)
            elif isinstance(value, float):
                logging.info("%s: %.3f", metric, value)
            else:
                logging.info("%s: %s", metric, value)
            
    except Exception as e:
        logging.error("Error in analysis: %s", str(e), exc_info=True)
        raise

if __name__ == "__main__":
    main()