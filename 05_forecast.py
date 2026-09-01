"""
Forecasting layer.
"""

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")  # statsmodels convergence chatter

FORECAST_HORIZON = 5          # years ahead
BACKTEST_HOLDOUT = 5          # years held out to score accuracy
INDICATORS_TO_FORECAST = [
    "GDP_per_capita_USD",
    "unemployment_rate",
    "literacy_rate",
    "internet_users_percent",
]

PERCENT_BOUNDED = {"unemployment_rate", "literacy_rate", "internet_users_percent",
                    "electricity_access_percent", "inflation_rate"}


def _fit_ets(train, horizon):
    """Holt's linear trend (ETS with additive trend, no seasonality)."""
    model = ExponentialSmoothing(train, trend="add", damped_trend=True)
    fit = model.fit(optimized=True)
    return fit.forecast(horizon)


def _fit_arima(train, horizon, order=(1, 1, 1)):
    """Simple ARIMA(1,1,1) — adequate for short annual macro series."""
    model = ARIMA(train, order=order)
    fit = model.fit()
    return fit.forecast(horizon)


def _mape(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100)


def _rmse(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def backtest_indicator(series: pd.Series, holdout: int = BACKTEST_HOLDOUT) -> dict:
    """
    Train on all but the last `holdout` years, predict those years,
    and score both models against what actually happened.
    """
    train, test = series.iloc[:-holdout], series.iloc[-holdout:]

    results = {}
    for name, fit_fn in [("ETS", _fit_ets), ("ARIMA", _fit_arima)]:
        try:
            preds = fit_fn(train.values, holdout)
            results[name] = {
                "rmse": _rmse(test.values, preds),
                "mape": _mape(test.values, preds),
            }
        except Exception as e:
            results[name] = {"rmse": np.nan, "mape": np.nan, "error": str(e)}
    return results


def forecast_indicator(series: pd.Series, best_model: str, horizon: int = FORECAST_HORIZON):
    """Refit the better-scoring model on the FULL series and forecast forward."""
    fit_fn = _fit_ets if best_model == "ETS" else _fit_arima
    return fit_fn(series.values, horizon)


def run_forecasting_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("year").set_index("year")
    last_year = int(df.index.max())
    future_years = list(range(last_year + 1, last_year + 1 + FORECAST_HORIZON))

    comparison_rows = []
    forecast_rows = []

    for col in INDICATORS_TO_FORECAST:
        if col not in df.columns:
            print(f"  Skipping {col} — not found in dataset")
            continue

        series = df[col].dropna()
        if len(series) < BACKTEST_HOLDOUT + 8:
            print(f"  Skipping {col} — not enough history for a reliable backtest")
            continue

        scores = backtest_indicator(series)
        for model_name, m in scores.items():
            comparison_rows.append({
                "indicator": col,
                "model": model_name,
                "rmse": round(m["rmse"], 4) if not np.isnan(m["rmse"]) else None,
                "mape_pct": round(m["mape"], 2) if not np.isnan(m["mape"]) else None,
            })

        valid = {k: v for k, v in scores.items() if not np.isnan(v.get("mape", np.nan))}
        best_model = min(valid, key=lambda k: valid[k]["mape"]) if valid else "ETS"

        forecast_values = forecast_indicator(series, best_model)
        for yr, val in zip(future_years, forecast_values):
            val = float(val)
            if col in PERCENT_BOUNDED:
                val = min(max(val, 0.0), 100.0 if col != "inflation_rate" else val)
            forecast_rows.append({
                "year": yr,
                "indicator": col,
                "forecast_value": round(val, 4),
                "model_used": best_model,
            })

    comparison_df = pd.DataFrame(comparison_rows)
    forecast_df = pd.DataFrame(forecast_rows)
    return comparison_df, forecast_df


if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)

    df = pd.read_csv("data/clean_data.csv")
    print(f"Loaded {len(df)} years of data ({df['year'].min()}–{df['year'].max()})\n")

    print("Backtesting ETS vs. ARIMA on a "
          f"{BACKTEST_HOLDOUT}-year holdout window...")
    comparison_df, forecast_df = run_forecasting_pipeline(df)

    print("\nModel accuracy comparison (lower is better):")
    print(comparison_df.to_string(index=False))

    print(f"\n{FORECAST_HORIZON}-year forecast (model selected per-indicator by lowest MAPE):")
    print(forecast_df.to_string(index=False))

    comparison_df.to_csv("outputs/forecast_model_comparison.csv", index=False)
    forecast_df.to_csv("outputs/forecast_5yr.csv", index=False)
    print("\nSaved: outputs/forecast_model_comparison.csv, outputs/forecast_5yr.csv")
