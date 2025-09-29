# run_evaluation.py

import os
import argparse
import config
from evaluation import load_actuals, load_forecast, generate_comparison_report

def main(exp_name: str, data_path: str):
    """
    Orchestrates the forecast evaluation process.
    1. Loads actual and forecasted data.
    2. Generates a comparison report.
    3. Saves the report to a CSV file.
    """
    print(f"📊 Running Evaluation for Experiment: {exp_name}")
    
    exp_root = os.path.join("experiments", exp_name)
    forecast_path = os.path.join(exp_root, config.OUTPUT_FILENAME) # From main config
    output_path = os.path.join(exp_root, config.EVALUATION_PARAMS["output_filename"])

    try:
        # 1. Load data sources
        print("Loading actual and forecasted data...")
        actuals_df = load_actuals(data_path, config.EVALUATION_PARAMS)
        forecast_prices = load_forecast(forecast_path, config.EVALUATION_PARAMS)
        
        # 2. Generate comparison report
        print("Generating comparison report...")
        report_df = generate_comparison_report(actuals_df, forecast_prices)
        
        # 3. Save the report
        report_df.to_csv(output_path, index=False)
        print(f"✅ Success! Evaluation report saved to: {output_path}")

    except FileNotFoundError as e:
        print(f"❌ Error: Could not find a required file. Make sure the forecast and data files exist. Details: {e}")
    except (ValueError, KeyError) as e:
        print(f"❌ Error: Data validation failed. Details: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run forecast vs. actuals evaluation.")
    parser.add_argument("--exp", type=str, default=config.EXPERIMENT_NAME, help="Name of the experiment to evaluate.")
    parser.add_argument("--data", type=str, default=config.DATA_PATH, help="Path to the original input data CSV file.")
    
    args = parser.parse_args()
    main(exp_name=args.exp, data_path=args.data)