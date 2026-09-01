"""
Phase 7: Anomaly detection.

Turns the "2020 was a COVID shock" narrative finding into two statistical
results:

  1. A per-indicator z-score of each year's value against that indicator's
     own trend-adjusted history (first-differenced, so a growing series
     doesn't falsely flag every later year as "extreme").
  2. An Isolation Forest run across ALL indicators simultaneously, treating
     each year as a point in multi-dimensional indicator-space, to check
     whether 2020 stands out jointly, not just on unemployment alone.

Run after 02_clean_store.py (needs data/clean_data.csv).
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

Z_FLAG_THRESHOLD = 2.0  # |z| above this is flagged as anomalous


def zscore_on_changes(series: pd.Series) -> pd.Series:
    """
    Z-score the year-over-year CHANGE in a series, not the level itself.
    For a steadily growing indicator (GDP, literacy...), the level always
    hits a new high — that's not an anomaly, that's a trend. The change
    (or % change) is what should look unusual in a shock year.
    """
    diffs = series.diff()
    mean, std = diffs.mean(), diffs.std()
    if std == 0 or np.isnan(std):
        return pd.Series(np.nan, index=series.index)
    return (diffs - mean) / std


def run_zscore_flagging(df: pd.DataFrame, indicators: list) -> pd.DataFrame:
    df = df.sort_values("year").reset_index(drop=True)
    rows = []
    for col in indicators:
        if col not in df.columns:
            continue
        z = zscore_on_changes(df[col])
        for year, z_val, raw_val in zip(df["year"], z, df[col]):
            if pd.notna(z_val):
                rows.append({
                    "year": int(year),
                    "indicator": col,
                    "value": raw_val,
                    "z_score_of_change": round(float(z_val), 3),
                    "flagged_anomaly": bool(abs(z_val) >= Z_FLAG_THRESHOLD),
                })
    return pd.DataFrame(rows)


def run_isolation_forest(df: pd.DataFrame, indicators: list, contamination: float = 0.1):
    """
    Treat each year as a row in indicator-space and ask: which years look
    jointly unusual across the full indicator set, not just one column?

    contamination=0.1 means we expect roughly 10% of years to be flagged —
    a reasonable prior for a 30-40 year window containing one major global
    shock (2020) plus normal year-to-year noise.
    """
    df = df.sort_values("year").reset_index(drop=True)
    feature_df = df[indicators].copy()

    # Use year-over-year % change so slow secular trends (GDP always rising)
    # don't dominate the anomaly signal — we want to catch *shocks*, not growth.
    pct_change = feature_df.pct_change().replace([np.inf, -np.inf], np.nan)
    pct_change = pct_change.dropna()
    years_used = df.loc[pct_change.index, "year"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(pct_change.values)

    model = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    model.fit(X)
    scores = model.decision_function(X)      # higher = more normal
    preds = model.predict(X)                 # -1 = anomaly, 1 = normal

    result = pd.DataFrame({
        "year": years_used,
        "anomaly_score": scores,
        "is_anomaly": preds == -1,
    }).sort_values("anomaly_score")

    return result


if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)

    df = pd.read_csv("data/clean_data.csv")
    indicators = [c for c in [
        "GDP_current_USD", "GDP_per_capita_USD", "inflation_rate", "literacy_rate",
        "internet_users_percent", "unemployment_rate", "electricity_access_percent",
    ] if c in df.columns]

    print(f"Loaded {len(df)} years, checking {len(indicators)} indicators\n")

    # ---- 1. Per-indicator z-score of year-over-year change ----
    print("=" * 60)
    print("PER-INDICATOR Z-SCORES (of year-over-year change)")
    print("=" * 60)
    z_df = run_zscore_flagging(df, indicators)
    flagged = z_df[z_df["flagged_anomaly"]]
    print(f"Flagged {len(flagged)} indicator-year pairs at |z| >= {Z_FLAG_THRESHOLD}:")
    print(flagged.sort_values("year").to_string(index=False))
    z_df.to_csv("outputs/anomaly_zscores.csv", index=False)

    covid_row = z_df[(z_df["year"] == 2020) & (z_df["indicator"] == "unemployment_rate")]
    if not covid_row.empty:
        z = covid_row["z_score_of_change"].iloc[0]
        print(f"\n2020 unemployment change z-score: {z:.2f} "
              f"({'flagged as anomalous' if abs(z) >= Z_FLAG_THRESHOLD else 'within normal range'})")

    # ---- 2. Multivariate Isolation Forest across all indicators jointly ----
    print("\n" + "=" * 60)
    print("MULTIVARIATE ANOMALY DETECTION (Isolation Forest, all indicators jointly)")
    print("=" * 60)
    iso_df = run_isolation_forest(df, indicators)
    print(iso_df.to_string(index=False))
    iso_df.to_csv("outputs/anomaly_isolation_forest.csv", index=False)

    if 2020 in iso_df["year"].values:
        row = iso_df[iso_df["year"] == 2020].iloc[0]
        rank = int(iso_df.reset_index(drop=True).index[iso_df["year"] == 2020][0]) + 1
        print(f"\n2020 ranks #{rank} of {len(iso_df)} years by anomaly score "
              f"(#1 = most anomalous). Flagged: {bool(row['is_anomaly'])}")

    print("\nSaved: outputs/anomaly_zscores.csv, outputs/anomaly_isolation_forest.csv")
