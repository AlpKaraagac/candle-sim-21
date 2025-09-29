import json
import pandas as pd
from typing import Dict, List, Any

def load_actuals(data_path: str, params: Dict[str, Any]) -> pd.DataFrame:
    """Loads the actual historical data for the forecast period."""
    df = pd.read_csv(data_path)
    horizon = params["forecast_horizon"]
    target_col = params["target_column"]
    
    date_col = next((c for c in params["date_columns"] if c in df.columns), None)
    
    actuals_slice = df.tail(horizon).copy()
    
    if date_col:
        dates = pd.to_datetime(actuals_slice[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    else:
        dates = [str(i) for i in actuals_slice.index]
        
    actual_prices = pd.to_numeric(actuals_slice[target_col], errors="coerce").astype(float)
    
    return pd.DataFrame({"date": dates, "actual": actual_prices}).reset_index(drop=True)

def load_forecast(forecast_path: str, params: Dict[str, Any]) -> List[float]:
    """Loads the forecasted price list from a Gemini JSON output file."""
    forecast_key = params["forecast_json_key"]
    with open(forecast_path, 'r') as f:
        data = json.load(f)
        
    forecasted_prices = data.get(forecast_key)
    if not isinstance(forecasted_prices, list):
        raise ValueError(f"Forecast JSON must contain a list under the key '{forecast_key}'.")
        
    return [float(price) for price in forecasted_prices]

def generate_comparison_report(actuals_df: pd.DataFrame, forecast_prices: List[float]) -> pd.DataFrame:
    """Compares actuals and forecasts and returns a comparison DataFrame."""
    horizon = len(actuals_df)
    if len(forecast_prices) != horizon:
        raise ValueError(f"Mismatch in length: Actuals have {horizon} entries, but forecast has {len(forecast_prices)}.")
    
    report_df = actuals_df.copy()
    report_df["forecast"] = forecast_prices
    report_df["difference"] = (report_df["actual"] - report_df["forecast"]).abs()
    
    return report_df

def load_actuals(params: Dict[str, Any], data_path: str = None, df: pd.DataFrame = None) -> pd.DataFrame:
    """Loads the actual historical data for the forecast period."""
    if df is None:
        if data_path is None:
            raise ValueError("Must provide either a data_path or a DataFrame.")
        df = pd.read_csv(data_path)
    
    horizon = params["forecast_horizon"]
    target_col = params["target_column"]
    
    date_col = next((c for c in params["date_columns"] if c in df.columns), None)
    
    actuals_slice = df.tail(horizon).copy()
    
    if date_col:
        dates = pd.to_datetime(actuals_slice[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    else:
        dates = [str(i) for i in actuals_slice.index]
        
    actual_prices = pd.to_numeric(actuals_slice[target_col], errors="coerce").astype(float)
    
    return pd.DataFrame({"date": dates, "actual": actual_prices}).reset_index(drop=True)