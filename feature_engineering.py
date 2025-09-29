# feature_engineering.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

def compute_window_features(df_window: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, float]:
    """
    Computes all technical features for a given data window.
    Flexibly handles cases where only 'close' is available, or full OHLCV.
    """
    eps = params['eps']
    features = {}

    # --- Step 1: Safely Prepare Data Series ---
    # 'close' is mandatory, as it's the basis for most features.
    if "close" not in df_window.columns:
        raise ValueError("Input DataFrame must contain a 'close' column.")
    
    close = df_window["close"].astype(float)
    
    # Safely get optional columns. They will be None if not present.
    open_ = df_window.get("open", None)
    high = df_window.get("high", None)
    low = df_window.get("low", None)
    volume = df_window.get("volume", None)
    
    if open_ is not None: open_ = open_.astype(float)
    if high is not None: high = high.astype(float)
    if low is not None: low = low.astype(float)
    if volume is not None: volume = volume.astype(float)

    # --- Features that only require 'close' (will always be calculated) ---
    ret = close.pct_change().dropna()
    if ret.empty: ret = pd.Series([0.0])
    
    features["cum_return"] = (close.iloc[-1] / (close.iloc[0] + eps)) - 1.0 if len(close) > 0 else 0.0
    features["avg_daily_return"] = ret.mean()
    features["std_daily_return"] = ret.std() if len(ret) > 1 else 0.0
    
    if len(close) >= 2:
        y = close.values
        x = np.arange(len(y))
        coef = np.polyfit(x, y, 1)
        features["slope_close"] = coef[0]
    else:
        features["slope_close"] = 0.0
    
    roll_max = close.cummax()
    drawdown = (close - roll_max) / (roll_max + eps)
    features["max_drawdown"] = drawdown.min()

    for period in params['sma_periods']:
        features[f"sma_{period}"] = close.rolling(period).mean().iloc[-1] if len(close) >= period else close.mean()
    
    features["rsi"] = _calculate_rsi(close, params['rsi_period'], eps)
    features["macd"] = _calculate_macd(close, params['macd_fast'], params['macd_slow'])
    
    # --- Step 2: Conditionally Calculate Features ---

    # Calculate volume-based features only if 'volume' is available
    if volume is not None:
        features["avg_volume"] = volume.mean()
        features["price_volume_corr"] = close.corr(volume) if len(close) >= 2 else 0.0
    else:
        features["avg_volume"] = 0.0
        features["price_volume_corr"] = 0.0

    # Calculate candlestick/range features only if OHLC are available
    if all(col is not None for col in [open_, high, low]):
        body = (close - open_).abs()
        hl_range = (high - low)
        features["avg_body_to_range"] = (body.mean() / (hl_range.mean() + eps))
        features["doji_pct"] = (body < (params['doji_threshold'] * hl_range)).mean()
        features["atr"] = _calculate_atr(high, low, close, params['atr_period'])
        features["bb_pct_b"], features["bb_bandwidth"] = _calculate_bollinger(close, params['bollinger_period'], eps)
    else:
        # Provide default values if columns are missing
        features["avg_body_to_range"] = 0.0
        features["doji_pct"] = 0.0
        features["atr"] = 0.0
        # Bollinger Bands only need 'close', but are often used with range analysis
        features["bb_pct_b"], features["bb_bandwidth"] = _calculate_bollinger(close, params['bollinger_period'], eps)


    # Final cleanup of NaN/inf values
    for k, v in features.items():
        if pd.isna(v) or not np.isfinite(v):
            features[k] = 0.0
    
    return features
# --- Helper functions for individual indicators ---

def _calculate_rsi(series: pd.Series, period: int, eps: float) -> float:
    if len(series) < period: return 50.0 # Return neutral value for short series
    delta = series.diff().dropna()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.ewm(alpha=1.0/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1.0/period, adjust=False).mean()
    rs = roll_up / (roll_down + eps)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.iloc[-1]

def _calculate_macd(series: pd.Series, fast: int, slow: int) -> float:
    if len(series) < slow: return 0.0
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    return (ema_fast - ema_slow).iloc[-1]

def _calculate_bollinger(series: pd.Series, period: int, eps: float) -> Tuple[float, float]:
    if len(series) < period: return 0.5, 0.0 # Neutral values
    sma = series.rolling(period).mean().iloc[-1]
    std = series.rolling(period).std().iloc[-1]
    if pd.isna(std) or std == 0: return 0.5, 0.0
    upper = sma + 2 * std
    lower = sma - 2 * std
    pct_b = (series.iloc[-1] - lower) / (upper - lower + eps)
    bandwidth = (upper - lower) / (sma + eps)
    return pct_b, bandwidth

def _calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> float:
    if len(close) < 2: return 0.0
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean().iloc[-1] if len(tr) >= period else tr.mean()