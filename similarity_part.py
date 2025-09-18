import numpy as np
import os
import pandas as pd
from dtaidistance import dtw


def generate_features(candidate_window):
    """
    Expects values in this exact order:
    [close, high, low, open, volume, totalQuantity, weightedAverage]
    Returns a 1D numpy array of engineered features.
    
    TODO: expand this function with more features as needed.
    """
    close, _, _, _, _, _, _ = map(float, candidate_window)
    return np.array([close], dtype=float)


def select_non_overlapping(sorted_df, window_size, n_days, max_select=None):
    """
    Greedy selection: walk the rows (already sorted by ascending distance),
    keep a window if it doesn't overlap any previously kept window.
    """
    occupied = np.zeros(n_days, dtype=bool)
    chosen = []

    for _, row in sorted_df.iterrows():
        s = int(row["start_idx"])
        e = s + window_size - 1
        if s < 0 or e >= n_days:
            continue
        # overlap check
        if occupied[s:e+1].any():
            continue
        # keep it
        occupied[s:e+1] = True
        chosen.append(row)
        if (max_select is not None) and (len(chosen) >= max_select):
            break

    return pd.DataFrame(chosen).reset_index(drop=True)



def _attach_window_dates(df_dates, candidates_df, T):
    dates = df_dates.astype(str).to_numpy()

    x_start_idx = candidates_df["start_idx"].to_numpy(dtype=int)
    x_end_idx   = candidates_df["end_idx"].to_numpy(dtype=int)
    y_start_idx = x_end_idx + 1           # == start_idx + T
    y_end_idx   = y_start_idx + T - 1     # == start_idx + 2T - 1

    n = len(dates)
    valid = (
        (x_start_idx >= 0) & (x_end_idx < n) &
        (y_start_idx >= 0) & (y_end_idx < n)
    )

    out = candidates_df.copy()
    out["x_start_date"] = np.where(valid, dates[x_start_idx], None)
    out["x_end_date"]   = np.where(valid, dates[x_end_idx], None)
    out["y_start_date"] = np.where(valid, dates[y_start_idx], None)
    out["y_end_date"]   = np.where(valid, dates[y_end_idx], None)
    return out


def main():
    experiment_name = "baseline"
    print("Experiment Name:", experiment_name)

    data_dir = "data"
    exp_root = os.path.join("experiments", experiment_name)
    os.makedirs(exp_root, exist_ok=True)

    df = pd.read_csv(os.path.join(data_dir, "convertcsv.csv"))

    # Identify feature columns by excluding 'date' and 'time'
    feature_cols = [col for col in df.columns if col.lower() not in ['date', 'time']]
    raw = df[feature_cols].values

    if np.isnan(raw).any():
        print("Warning: NaN values found. Forward-filling NaNs.")
        raw = pd.DataFrame(raw).fillna(method='ffill').values

    # Window size for similarity computation
    T = 21
    N = len(raw)
    if N < 2*T:
        raise ValueError(f"Need at least {2*T} rows; got {N}.")

    feat = np.array([generate_features(raw[i]) for i in range(N)])

    # Build query (last T days) and all valid candidate starts that do NOT overlap query
    # Last query starts at N - T. Candidates can start at [0 .. N - 2T] inclusive.
    query_window = feat[-T:].ravel()
    candidate_starts = np.arange(0, N - 2*T + 1, dtype=int)

    # Compute DTW distance for each candidate window
    dists = []
    for s in candidate_starts:
        cand = feat[s:s+T].ravel()
        d = dtw.distance(query_window, cand)
        dists.append(d)
    dists = np.array(dists, dtype=float)

    all_candidates = pd.DataFrame({
        "start_idx": candidate_starts,
        "end_idx": candidate_starts + T - 1,
        "dtw_distance": dists
    }).sort_values("dtw_distance", ascending=True).reset_index(drop=True)

    max_select = None # limit to this many windows, or None for no limit
    selected = select_non_overlapping(all_candidates, window_size=T, n_days=N, max_select=max_select)
    
    selected = _attach_window_dates(df['date'], selected, T)
    all_candidates = _attach_window_dates(df['date'], all_candidates, T)

    # Save both the full ranking and the non-overlapping selection
    all_path = os.path.join(exp_root, "all_candidate_windows_sorted.csv")
    sel_path = os.path.join(exp_root, "top_similar_windows_nonoverlap.csv")
    all_candidates.to_csv(all_path, index=False)
    selected.to_csv(sel_path, index=False)

    print(f"Computed {len(all_candidates)} candidate windows (pre-query).")
    print(f"Selected {len(selected)} non-overlapping windows.")
    print(f"Saved full ranking to: {all_path}")
    print(f"Saved non-overlapping best set to: {sel_path}")


if __name__ == "__main__":
    main()