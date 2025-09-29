import os
import pandas as pd
from typing import Dict, Any, Optional

from feature_engineering import compute_window_features

def prepare_experiment_data(
    data_params: Dict,
    feature_params: Dict,
    exp_root: str,
    df: pd.DataFrame, # Changed from data_path to accept a DataFrame directly
    cash_flow_df: Optional[pd.DataFrame] = None # <-- 1. Accept the optional DataFrame
) -> Dict[str, Any]:
    """Loads all data, computes features, and builds the final payload with optional cash flow series."""
    
    # --- Load and filter candidate windows ---
    candidate_info_path = os.path.join(exp_root, "all_candidate_windows_sorted.csv")
    wins_df = pd.read_csv(candidate_info_path)
    wins_df["dtw_distance"] = pd.to_numeric(wins_df["dtw_distance"], errors="coerce")
    wins_df = wins_df.dropna(subset=["dtw_distance"])
    wins_df = wins_df[wins_df["dtw_distance"] <= data_params['cutoff']]
    wins_df = wins_df.sort_values("dtw_distance").head(data_params['max_len']).reset_index(drop=True)

    # --- Prepare Query Window ---
    t = data_params['t']
    cols = list(data_params['features'])
    q_start_idx = len(df) - t
    query_start_date = df.iloc[q_start_idx]["date"]
    query_end_date = df.iloc[-1]["date"]
    query_window_df = df.loc[q_start_idx:, cols]
    
    query_info = {
        "query_window": query_window_df.to_numpy(),
        "start_date": str(query_start_date),
        "end_date": str(query_end_date),
        "features": cols,
        "window_length": t,
    }

    # <-- 2. Conditionally slice and add cash flow for the query
    if cash_flow_df is not None:
        # Slice the cash flow data between the query's start and end dates
        query_cash_flow = cash_flow_df.loc[query_start_date:query_end_date]['pgc'].tolist()
        query_info["cash_flow_series"] = query_cash_flow
    
    # --- Prepare Candidate Windows ---
    candidates = []
    for _, row in wins_df.iterrows():
        s_idx = int(row["start_idx"])
        e_idx = s_idx + t - 1
        
        if s_idx < 0 or e_idx >= len(df) or (e_idx + t) >= len(df):
            continue

        start_date = df.iloc[s_idx]["date"]
        end_date = df.iloc[e_idx]["date"]
        x_window_df = df.loc[s_idx:e_idx, cols]
        y_window_df = df.loc[e_idx + 1 : e_idx + t, cols]
        
        candidate = {
            "x": x_window_df.to_numpy(),
            "y": y_window_df.to_numpy(),
            "start_date": str(start_date),
            "end_date": str(end_date),
            "dtw_distance": float(row["dtw_distance"]),
            "extra_features": compute_window_features(x_window_df, feature_params),
        }

        # <-- 3. Conditionally slice and add cash flow for each candidate
        if cash_flow_df is not None:
            candidate_cash_flow = cash_flow_df.loc[start_date:end_date]['pgc'].tolist()
            candidate["cash_flow_series"] = candidate_cash_flow
            
        candidates.append(candidate)
        
    return {"query": query_info, "candidates": candidates}

def load_and_prepare_cash_flow(cash_flow_path: str, end_date: pd.Timestamp = None) -> pd.DataFrame | None:
    """
    Loads and prepares cash flow data, ensuring no future data is included.
    """
    try:
        cash_flow_df = pd.read_csv(cash_flow_path)
        cash_flow_df['date'] = pd.to_datetime(cash_flow_df['date'], format='%Y%m%d')
        
        if end_date:
            cash_flow_df = cash_flow_df[cash_flow_df['date'] <= end_date]

        cash_flow_df.set_index('date', inplace=True)
        return cash_flow_df
    except FileNotFoundError:
        print(f"Warning: Cash flow data file not found at {cash_flow_path}. Proceeding without it.")
        return None