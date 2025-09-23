from dotenv import load_dotenv
import os
import json
import pandas as pd
import numpy as np

from google import genai
from google.genai import types


def _to_jsonable(x):
    """Recursively convert numpy types/arrays to plain Python for JSON serialization."""
    if isinstance(x, np.ndarray):
        return [_to_jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.floating, np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.integer, np.int32, np.int64)):
        return int(x)
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x


def load_data(
    data_dir,
    exp_root,
    t=21,
    cutoff=0.1,
    max_len=230,
    features=("close",),
    keepdim="auto",
):
    df = pd.read_csv(os.path.join(data_dir, "convertcsv.csv"))
    wins = pd.read_csv(os.path.join(exp_root, "all_candidate_windows_sorted.csv")).copy()

    # filter by cutoff and (optional) cap
    wins["dtw_distance"] = pd.to_numeric(wins["dtw_distance"], errors="coerce")
    wins = wins.dropna(subset=["dtw_distance"])
    wins = wins[wins["dtw_distance"] <= float(cutoff)].sort_values("dtw_distance").reset_index(drop=True)
    if max_len is not None and max_len > 0:
        wins = wins.head(int(max_len))

    D = len(features)
    def _reshape(a2d):
        if keepdim == "2d": return a2d
        if keepdim == "1d":
            if D != 1: raise ValueError("keepdim='1d' requires exactly one feature.")
            return a2d[:, 0]
        return a2d[:, 0] if D == 1 else a2d  # auto

    # build X/Y
    X_list, Y_list, kept_rows = [], [], []
    N = len(df)
    cols = list(features)

    for _, row in wins.iterrows():
        s = int(row["start_idx"])
        # bounds
        if s < 0 or s + t - 1 >= N: 
            continue
        if s + 2*t - 1 >= N: 
            continue
        x2d = df.loc[s:s+t-1, cols].to_numpy()
        y2d = df.loc[s+t:s+2*t-1, cols].to_numpy()

        X_list.append(_reshape(x2d))
        Y_list.append(_reshape(y2d))
        kept_rows.append(row)

    X = np.array(X_list, dtype=float)
    Y = np.array(Y_list, dtype=float)
    wins = pd.DataFrame(kept_rows).reset_index(drop=True)

    # query window (last T)
    q2d = df.loc[len(df)-t:len(df)-1, cols].to_numpy()
    query_window = _reshape(q2d)
    q_start = str(df.iloc[-t]["date"]) if "date" in df.columns else None
    q_end   = str(df.iloc[-1]["date"]) if "date" in df.columns else None
    returned_dim = "2d" if (keepdim == "2d" or (keepdim == "auto" and D > 1)) else "1d"

    # candidates payload
    candidates = []
    for i in range(len(wins)):
        s = int(wins.iloc[i]["start_idx"])
        e = int(wins.iloc[i]["end_idx"]) if "end_idx" in wins.columns else s + t - 1
        start_date = str(df.iloc[s]["date"]) if "date" in df.columns else None
        end_date   = str(df.iloc[e]["date"]) if "date" in df.columns else None
        candidates.append({
            "x": X[i],
            "y": Y[i],
            "start_date": start_date,
            "end_date": end_date,
            "dtw_distance": float(wins.iloc[i]["dtw_distance"]),
        })

    return {
        "query": {
            "query_window": query_window,
            "start_date": q_start,
            "end_date": q_end,
            "features": cols,
            "window_length": t,
            "returned_dim": returned_dim,
        },
        "candidates": candidates,
    }


def main(experiment_name="baseline"):
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("API_KEY not found in environment variables.")
    print("API Key loaded successfully.")

    print("Experiment Name:", experiment_name)

    data_dir = "data"
    exp_root = os.path.join("experiments", experiment_name)

    # 1) Close only (backward-compatible 1D):
    data = load_data(data_dir, exp_root, cutoff=0.05, features=("close",), keepdim="auto")

    # 2) OHLCV as 2D windows:
    """
    data = load_data(
        data_dir,
        exp_root,
        cutoff=0.05,
        features=("open", "high", "low", "close", "volume"),
        keepdim="2d",
    )
    """

    print("Prepared data for LLM with",
          len(data["candidates"]), f"candidates (each with {data['query']['window_length']} timesteps).")
    print("Feature set:", data["query"]["features"], "Returned dim:", data["query"]["returned_dim"])

    prompt_path = "system_prompt.md"
    with open(prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()
    print("System Prompt loaded successfully.")

    client = genai.Client(api_key=api_key)
    print("Gemini client initialized.")

    payload = _to_jsonable(data)
    payload_str = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    contents = [
        system_prompt.strip(),
        payload_str
    ]

    config = types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json",
    )

    model_name = "gemini-2.5-flash"

    resp = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=config,
    )

    if not hasattr(resp, "text") or not resp.text:
        raise RuntimeError("Empty response from Gemini API.")

    try:
        result = json.loads(resp.text)
    except json.JSONDecodeError as e:
        raw_path = os.path.join(exp_root, "gemini_raw_output.txt")
        os.makedirs(exp_root, exist_ok=True)
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(resp.text)
        raise RuntimeError(f"Model did not return valid JSON. Raw saved to {raw_path}") from e

    out_path = os.path.join(exp_root, "gemini_forecast.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Response received from Gemini API. Saved to: {out_path}")


if __name__ == "__main__":
    main(experiment_name="baseline-test", predict_one=False)
