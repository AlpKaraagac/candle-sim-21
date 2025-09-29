import pandas as pd
import numpy as np
from dtaidistance import dtw_ndim
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Any

def normalize_window(window: np.ndarray) -> np.ndarray:
    """Z-score normalize each feature/column in the window."""
    mean = np.mean(window, axis=0)
    std = np.std(window, axis=0)
    std_safe = np.where(std == 0.0, 1.0, std)
    return (window - mean) / std_safe

def denoise_features(features: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Denoise the entire feature matrix using a rolling window."""
    window_size = params['denoise_window_size']
    method = params['denoise_method']
    
    df = pd.DataFrame(features)
    if method == "moving_average":
        rolled = df.rolling(window=window_size, center=True, min_periods=1).mean()
    elif method == "median":
        rolled = df.rolling(window=window_size, center=True, min_periods=1).median()
    else:
        return features
    return rolled.values

def compute_dtw_similarities(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Calculates DTW distance from the latest window to all historical windows.
    
    Returns a sorted DataFrame with candidate windows and their distances.
    """
    t = params['t']
    feature_cols = [col for col in df.columns if col.lower() not in ['date', 'time']]
    feat = df[feature_cols].values
    
    if np.isnan(feat).any():
        print("Warning: NaN values found. Forward-filling.")
        feat = pd.DataFrame(feat).fillna(method='ffill').values

    n = len(feat)
    if n < 2 * t:
        raise ValueError(f"Data has {n} rows, but needs at least {2*t} for one comparison.")

    if params.get('denoise_windows', False):
        print(f"Denoising features with method: {params['denoise_method']}")
        feat = denoise_features(feat, params)

    query = feat[-t:]
    if params.get('normalize_windows', False):
        query = normalize_window(query)

    candidate_starts = np.arange(0, n - 2 * t + 1, dtype=int)
    
    dists = []
    for s in candidate_starts:
        cand = feat[s:s+t]
        if params.get('normalize_windows', False):
            cand = normalize_window(cand)
        
        d = dtw_ndim.distance(query.T, cand.T)
        dists.append(d)

    dists = np.array(dists, dtype=float).reshape(-1, 1)
    scaled_dists = MinMaxScaler().fit_transform(dists).ravel()

    candidates_df = pd.DataFrame({
        "start_idx": candidate_starts,
        "end_idx": candidate_starts + t - 1,
        "dtw_distance": scaled_dists
    }).sort_values("dtw_distance").reset_index(drop=True)
    
    if 'date' in df.columns:
        dates = df['date'].astype(str).to_numpy()
        start_dates = dates[candidates_df["start_idx"].to_numpy()]
        end_dates = dates[candidates_df["end_idx"].to_numpy()]
        candidates_df["start_date"] = start_dates
        candidates_df["end_date"] = end_dates

    return candidates_df