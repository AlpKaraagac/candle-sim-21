import os
import json
import argparse
from google import genai

import config
from data_processing import prepare_experiment_data
from llm_interaction import get_gemini_forecast, generate_candidate_summaries
from utils import to_jsonable

def main(exp_name: str, data_path: str):
    """Main execution script."""
    print(f"🚀 Running experiment: {exp_name}")

    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")
    
    genai.configure(api_key=config.GEMINI_API_KEY)
    
    exp_root = os.path.join("experiments", exp_name)
    
    print("Loading and processing data...")
    payload = prepare_experiment_data(config.DATA_PARAMS, config.FEATURE_PARAMS, exp_root, data_path)
    print(f"Prepared data with {len(payload['candidates'])} candidates.")

    # (Optional) Generate LLM-based summaries for each candidate
    if config.SUMMARIZER_PARAMS['enabled']:
        print("Generating LLM summaries for candidate windows...")
        summarizer_client = genai.GenerativeModel(config.SUMMARIZER_PARAMS['model_name'])
        
        summaries = generate_candidate_summaries(
            summarizer_client,
            payload["candidates"],
            config.SUMMARIZER_PARAMS
        )
        
        # Attach the generated summary to each candidate dictionary
        for i, candidate in enumerate(payload["candidates"]):
            candidate["llm_summary"] = summaries[i]
        print(f"Successfully generated and attached {len(summaries)} summaries.")
    
    print("Initializing main Gemini client and loading system prompt...")
    with open(config.SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        system_prompt = f.read()
    
    client = genai.GenerativeModel(config.LLM_PARAMS['model_name'])
    
    print("Sending final request to Gemini API...")
    final_payload = to_jsonable(payload) # Payload now includes summaries
    result = get_gemini_forecast(client, final_payload, system_prompt, config.LLM_PARAMS)
    
    out_path = os.path.join(exp_root, config.OUTPUT_FILENAME)
    os.makedirs(exp_root, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Success! Response saved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gemini time-series forecasting experiment.")
    parser.add_argument("--exp", type=str, default=config.EXPERIMENT_NAME, help="Name of the experiment.")
    parser.add_argument("--data", type=str, default=config.DATA_PATH, help="Path to the input data CSV file.")
    
    args = parser.parse_args()
    main(exp_name=args.exp, data_path=args.data)