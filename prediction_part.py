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


def load_data(data_dir, exp_root, top=None, t=21):
    temp_df = pd.read_csv(os.path.join(data_dir, "convertcsv.csv"))
    similar_windows = pd.read_csv(os.path.join(exp_root, "top_similar_windows_nonoverlap.csv")).sort_values(
        "dtw_distance"
    ).reset_index(drop=True)
    if top is not None:
        similar_windows = similar_windows.head(top)

    X_list, y_list = [], []
    for i in similar_windows['start_idx']:
        i = int(i)
        x = temp_df.iloc[i:i+t]['close'].values
        y = temp_df.iloc[i+t:i+(t*2)]['close'].values
        X_list.append(x)
        y_list.append(y)

    X = np.array(X_list)
    y = np.array(y_list)
    query_window = temp_df.iloc[-t:]['close'].values

    query = {
        "query_window": query_window,
        "start_date": str(temp_df.iloc[-t]['date']) if 'date' in temp_df.columns else None,
        "end_date": str(temp_df.iloc[-1]['date']) if 'date' in temp_df.columns else None,
    }

    candidates = []
    for i in range(X.shape[0]):
        start_idx = int(similar_windows.iloc[i]['start_idx'])
        end_idx = int(similar_windows.iloc[i]['end_idx']) if 'end_idx' in similar_windows.columns else start_idx + t - 1
        start_date = str(temp_df.iloc[start_idx]['date']) if 'date' in temp_df.columns else None
        end_date = str(temp_df.iloc[end_idx]['date']) if 'date' in temp_df.columns else None
        candidates.append({
            "x": X[i],
            "y": y[i],
            "start_date": start_date,
            "end_date": end_date,
            "dtw_distance": float(similar_windows.iloc[i]['dtw_distance'])
        })
    return {
        "query": query,
        "candidates": candidates,
    }


def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("API_KEY not found in environment variables.")
    print("API Key loaded successfully.")

    experiment_name = "baseline"
    print("Experiment Name:", experiment_name)

    data_dir = "data"
    exp_root = os.path.join("experiments", experiment_name)

    data = load_data(data_dir, exp_root, top=10)
    print("Prepared data for LLM with",
          len(data["candidates"]), "candidates (each with 21 x & 21 y).")

    # Load your system prompt (the instructions for direct prediction)
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
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    )

    model_name = "gemini-2.5-flash"

    # ---- Call Gemini ----
    resp = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=config,
    )

    if not hasattr(resp, "text") or not resp.text:
        raise RuntimeError("Empty response from Gemini API.")

    # Parse JSON returned by the model
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
    main()
