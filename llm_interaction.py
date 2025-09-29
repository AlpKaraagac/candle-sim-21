import json
from typing import Dict, Any, List
import google.generativeai as genai
import numpy as np

def generate_candidate_summaries(
    client: genai.GenerativeModel,
    candidates: List[Dict[str, Any]],
    summarizer_params: Dict[str, Any]
) -> List[str]:
    """Asks an LLM to produce a one-line summary for each candidate window."""
    
    input_items = []
    for idx, c in enumerate(candidates, start=1):
        item = {"window_index": idx}
        if summarizer_params["prefer_features"] and "extra_features" in c:
            item["features"] = c["extra_features"]
        else:
            close_series = np.array(c.get("x"), dtype=float)
            item["close_series"] = close_series[:, 3].tolist() # Assuming close is the 4th feature
        input_items.append(item)

    system_instr = (
        "You are a concise time-series summarization assistant. For each window supplied, "
        "produce exactly one short, human-readable line in the format: "
        "'Window N: <brief summary>.' Your summary must use percent return, volatility (low/medium/high), "
        "and trend (uptrend/flat/downtrend). Be terse and use formatted percentages (e.g., +5.2% or -3.1%). "
        "Return exactly one line per window, in order, and nothing else."
    )
    
    from utils import to_jsonable
    json_input = json.dumps(to_jsonable(input_items), indent=2)
    user_block = f"Windows (JSON):\n{json_input}"
    
    contents = [system_instr, user_block]
    generation_config = {
        "temperature": summarizer_params['temperature'],
    }

    try:
        resp = client.generate_content(
            contents=contents,
        )
        lines = [ln.strip() for ln in resp.text.strip().splitlines() if ln.strip()]

        if len(lines) != len(candidates):
            print(f"⚠️ Warning: LLM returned {len(lines)} summaries but expected {len(candidates)}. Falling back.")
            return [f"Window {i+1}: Summary generation failed." for i in range(len(candidates))]
        
        return lines

    except Exception as e:
        print(f"Error during LLM summarization call: {e}")
        return [f"Window {i+1}: Summary generation failed due to API error." for i in range(len(candidates))]


def get_gemini_forecast(
    client: genai.GenerativeModel,
    payload: Dict[str, Any],
    system_prompt: str,
) -> Dict[str, Any]:
    """Sends the final payload to the Gemini API and gets a forecast."""

    payload_str = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    contents = [system_prompt.strip(), payload_str]
    
    try:
        resp = client.generate_content(contents=contents)
        return json.loads(resp.text)
    except json.JSONDecodeError as e:
        print(f"Error: Model did not return valid JSON. Raw response: {resp.text}")
        raise RuntimeError("Failed to decode Gemini API JSON response.") from e
    except Exception as e:
        print(f"An unexpected error occurred during API call: {e}")
        raise