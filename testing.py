import json
import os
import pandas as pd

def read_gemini_forecast(file_path='gemini_forecast.json'):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def main():
    data_dir = "data"
    forecast_path = "experiments/baseline/gemini_forecast.json"
    output_dir = "experiments/baseline"
    output_file = os.path.join(output_dir, "forecast_comparison.csv")

    # Read actuals
    df = pd.read_csv(os.path.join(data_dir, "convertcsv.csv"))

    possible_date_cols = ["date","time"]
    date_col = next((c for c in possible_date_cols if c in df.columns), None)

    close_col = "close"
    last21 = df.tail(21)

    if date_col:
        dates = pd.to_datetime(last21[date_col], errors="coerce").dt.strftime("%Y-%m-%d").tolist()
    else:
        # Fallback: use the row index as the date field
        dates = [str(i) for i in last21.index.tolist()]

    actuals = pd.to_numeric(last21[close_col], errors="coerce").astype(float).tolist()

    # Read forecast
    forecast = read_gemini_forecast(forecast_path)
    forecasted_close = forecast.get("close_prices", None)
    if not isinstance(forecasted_close, list):
        raise ValueError("Forecast JSON must contain a list under 'close_prices'.")
    forecasted_close = [float(x) for x in forecasted_close]

    # Ensure we have 21 items
    if len(forecasted_close) != 21 or len(actuals) != 21:
        raise ValueError("Both actual and forecasted close prices must have exactly 21 entries.")

    # Build comparison DataFrame
    differences = [abs(a - p) for a, p in zip(actuals, forecasted_close)]
    out_df = pd.DataFrame({
        "date": dates,
        "real": actuals,
        "prediction": forecasted_close,
        "difference": differences
    })

    # Write CSV
    os.makedirs(output_dir, exist_ok=True)
    out_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()
