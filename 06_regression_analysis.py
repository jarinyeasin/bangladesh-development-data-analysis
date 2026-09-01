import pandas as pd
import numpy as np
import statsmodels.api as sm


def build_lagged_features(df: pd.DataFrame, source_col: str, target_col: str,
                           max_lag: int = 3) -> pd.DataFrame:
    out = df[["year", target_col]].copy()
    for lag in range(1, max_lag + 1):
        out[f"{source_col}_lag{lag}"] = df[source_col].shift(lag)
    return out.dropna()


def run_lagged_ols(df: pd.DataFrame, source_col: str, target_col: str, max_lag: int = 3):
    results = {}
    lagged = build_lagged_features(df, source_col, target_col, max_lag)
    for lag in range(1, max_lag + 1):
        col = f"{source_col}_lag{lag}"
        sub = lagged[["year", target_col, col]].dropna()
        if len(sub) < 8:
            continue
        X = sm.add_constant(sub[col])
        y = sub[target_col]
        model = sm.OLS(y, X).fit()
        results[lag] = model
    return results


def summarize_lag_results(results: dict, source_col: str, target_col: str) -> pd.DataFrame:
    rows = []
    for lag, model in results.items():
        coef_name = [c for c in model.params.index if c != "const"][0]
        rows.append({
            "lag_years": lag,
            "coefficient": round(model.params[coef_name], 4),
            "p_value": round(model.pvalues[coef_name], 4),
            "r_squared": round(model.rsquared, 4),
            "n_obs": int(model.nobs),
            "significant_at_05": model.pvalues[coef_name] < 0.05,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df.insert(0, "target", target_col)
        df.insert(0, "predictor", source_col)
    return df


def run_simple_ols(df: pd.DataFrame, x_col: str, y_col: str):
    sub = df[[x_col, y_col]].dropna()
    X = sm.add_constant(sub[x_col])
    y = sub[y_col]
    return sm.OLS(y, X).fit(), len(sub)


if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)

    df = pd.read_csv("data/clean_data.csv").sort_values("year")
    print(f"Loaded {len(df)} years of data\n")


    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop("year")
    corr = df[numeric_cols].corr(method="pearson")
    print("=" * 60)
    print("CORRELATION MATRIX (Pearson, contemporaneous, no lag)")
    print("=" * 60)
    print(corr.round(2).to_string())
    corr.to_csv("outputs/correlation_matrix.csv")

    print("\n" + "=" * 60)
    print("Q: Does internet penetration predict GDP growth with a lag?")
    print("=" * 60)
    if {"internet_users_percent", "GDP_growth_pct"}.issubset(df.columns):
        lag_results = run_lagged_ols(df, "internet_users_percent", "GDP_growth_pct", max_lag=3)
        lag_summary = summarize_lag_results(lag_results, "internet_users_percent", "GDP_growth_pct")
        print(lag_summary.to_string(index=False) if not lag_summary.empty
              else "Not enough overlapping observations to test.")
        if not lag_summary.empty:
            lag_summary.to_csv("outputs/regression_internet_lag_gdp_growth.csv", index=False)
    else:
        print("Required columns not found — skipping.")

    # ---- 3. Does electricity access predict literacy? ----
    print("\n" + "=" * 60)
    print("Q: Does electricity access predict literacy?")
    print("=" * 60)
    if {"electricity_access_percent", "literacy_rate"}.issubset(df.columns):
        model, n = run_simple_ols(df, "electricity_access_percent", "literacy_rate")
        print(f"n = {n}")
        print(model.summary())
        with open("outputs/regression_electricity_literacy_summary.txt", "w") as f:
            f.write(model.summary().as_text())
    else:
        print("Required columns not found — skipping.")

    print("\nSaved: outputs/correlation_matrix.csv, "
          "outputs/regression_internet_lag_gdp_growth.csv, "
          "outputs/regression_electricity_literacy_summary.txt")
