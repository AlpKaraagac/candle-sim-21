# run_dtw_experiment.py
import os
import pandas as pd
import argparse

import config
from dtw_processing import compute_dtw_similarities

def main(exp_name: str, data_path: str):
    """
    Orchestrates the DTW similarity experiment.
    1. Loads config and data.
    2. Calls the processing function to compute distances.
    3. Saves the results.
    """
    print(f"▶️  Starting DTW Experiment: {exp_name}")
    print(f"Loading data from: {data_path}")

    df = pd.read_csv(data_path)
    print("Computing DTW distances...")
    results_df = compute_dtw_similarities(df, config.DTW_PARAMS)
    exp_root = os.path.join("experiments", exp_name)
    os.makedirs(exp_root, exist_ok=True)
    
    output_path = os.path.join(exp_root, "all_candidate_windows_sorted.csv")
    results_df.to_csv(output_path, index=False)
    
    print(f"✅ Success! Computed {len(results_df)} candidate windows.")
    print(f"Saved results to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DTW similarity search experiment.")
    parser.add_argument("--exp", type=str, default=config.EXPERIMENT_NAME, help="Name of the experiment.")
    parser.add_argument("--data", type=str, default=config.DATA_PATH, help="Path to the input data CSV file.")
    
    args = parser.parse_args()
    main(exp_name=args.exp, data_path=args.data)