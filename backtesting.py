import os
import argparse
import pandas as pd
import google.generativeai as genai
import json

import config
from dtw_processing import compute_dtw_similarities
from data_processing import prepare_experiment_data, load_and_prepare_cash_flow
from llm_interaction import get_gemini_forecast
from evaluation import load_actuals, load_forecast, generate_comparison_report
from utils import to_jsonable

def main(exp_name: str, data_path: str, use_cash_flow: bool = False):
    """
    Orchestrates a full backtesting run safely in memory.
    1. Loads the full dataset.
    2. Creates a temporary, truncated "training" version.
    3. Runs DTW search on the training data.
    4. Runs the LLM forecast on the training data.
    5. Runs evaluation by comparing the forecast against the full, original dataset.
    """

    print(f"Loading full dataset from: {data_path}")
    full_df = pd.read_csv(data_path)
    horizon = config.BACKTESTING_PARAMS['backtest_horizon']
    training_df = full_df.iloc[:-horizon].copy()
    print(f"Full dataset has {len(full_df)} rows. Training set has {len(training_df)} rows.")
    forecast_date = training_df['date'].iloc[-1]
    
    exp_root = os.path.join("experiments", exp_name)
    os.makedirs(exp_root, exist_ok=True)

    print("\n[Step 1/4] Running DTW similarity search...")
    dtw_results_df = compute_dtw_similarities(training_df, config.DTW_PARAMS)
    dtw_output_path = os.path.join(exp_root, "all_candidate_windows_sorted.csv")
    dtw_results_df.to_csv(dtw_output_path, index=False)
    print(f"DTW results saved to: {dtw_output_path}")

    print("\n[Step 2/4] Preparing data and running forecast...")
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in .env file.")
    genai.configure(api_key=config.GEMINI_API_KEY)
    
    with open(config.SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        system_prompt = f.read()
    
    cash_flow_df = None
    if use_cash_flow:
        # Load cash flow data, making sure to truncate it to the forecast date
        cash_flow_df = load_and_prepare_cash_flow(config.CASH_FLOW_DATA_PATH, end_date=forecast_date)
        
        # Dynamically add instructions to the system prompt
        cash_flow_prompt_addition = (
            "\n\n--- ADDITIONAL DATA: CASH FLOW ---\n"
            "The JSON payload includes an optional `cash_flow_series` for the query and each candidate. "
            "This series represents daily net money inflow/outflow. Positive values indicate buying pressure, "
            "and negative values indicate selling pressure. You must use this data as a key indicator of market "
            "sentiment to improve your forecast."
        )
        system_prompt += cash_flow_prompt_addition
    
    payload = prepare_experiment_data(
        config.DATA_PARAMS, config.FEATURE_PARAMS, exp_root, df=training_df, cash_flow_df = cash_flow_df
    )
    
    generation_config = {
        "temperature": config.LLM_PARAMS['temperature'],
        "response_mime_type": config.LLM_PARAMS['response_mime_type'],
    }

    client = genai.GenerativeModel(
        model_name=config.LLM_PARAMS['model_name'],
        generation_config=generation_config
    )

    forecast_result = get_gemini_forecast(
        client, to_jsonable(payload), system_prompt
    )
    
    forecast_path = os.path.join(exp_root, config.OUTPUT_FILENAME)
    with open(forecast_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(forecast_result, indent=2))
    print(f"Forecast saved to: {forecast_path}")
    
    print("\n[Step 3/4] Evaluating forecast against actual hold-out data...")
    actuals_df = load_actuals(config.EVALUATION_PARAMS, df=full_df)
    forecast_prices = load_forecast(forecast_path, config.EVALUATION_PARAMS)
    report_df = generate_comparison_report(actuals_df, forecast_prices)
    
    report_path = os.path.join(exp_root, config.EVALUATION_PARAMS["output_filename"])
    report_df.to_csv(report_path, index=False)
    print(f"Evaluation report saved to: {report_path}")
    
    print("\n[Step 4/4] ✅ Backtest complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a full backtest experiment.")
    parser.add_argument("--exp", type=str, default=config.EXPERIMENT_NAME, help="Name for the backtest experiment.")
    parser.add_argument("--data", type=str, default=config.DATA_PATH, help="Path to the full input data CSV file.")
    parser.add_argument(
        "--use-cash-flow",
        action=argparse.BooleanOptionalAction,
        default=config.USE_CASH_FLOW_DATA
    )
    
    args = parser.parse_args()
    main(exp_name=args.exp, data_path=args.data, use_cash_flow=args.use_cash_flow)