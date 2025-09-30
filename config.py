import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_PARAMS = {
    "model_name": "gemini-2.5-pro",
    "temperature": 0,
    "response_mime_type": "application/json",
}

SUMMARIZER_PARAMS = {
    "enabled": True,
    "model_name": "gemini-1.5-flash-latest",
    "prefer_features": True,
    "temperature": 0.2,
    "response_mime_type": "text/plain",
}

EXPERIMENT_NAME = "test-full-kopol"
DATA_PATH = "data/kopol.csv"
CASH_FLOW_DATA_PATH = "data/KOPOLflow.csv"
USE_CASH_FLOW_DATA = True
CANDIDATES_FILENAME = "all_candidate_windows_sorted.csv"
SYSTEM_PROMPT_PATH = "system_prompt.md"
OUTPUT_FILENAME = "gemini_forecast.json"

DTW_PARAMS = {
    "t": 21,
    "normalize_windows": True,
    "denoise_windows": True,
    "denoise_method": "moving_average",
    "denoise_window_size": 3,
}

DATA_PARAMS = {
    "t": 21,
    "cutoff": 0.2,
    "max_len": 100,
    "features": ("open", "high", "low", "close", "volume"),
    "keepdim": "auto",
}

FEATURE_PARAMS = {
    "sma_periods": [5, 10, 21],
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "bollinger_period": 20,
    "atr_period": 14,
    "doji_threshold": 0.1,
    "eps": 1e-9  # Epsilon for safe division
}

EVALUATION_PARAMS = {
    "forecast_horizon": 21,
    "target_column": "close",
    "date_columns": ["date", "time"],
    "output_filename": "forecast_comparison.csv",
    "forecast_json_key": "close_prices",
}

BACKTESTING_PARAMS = {
    "enabled": True,
    "backtest_horizon": 21,
}