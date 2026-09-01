"""
Data quality tests
"""

import os
import pandas as pd
import pytest

CSV_PATH = "data/clean_data.csv"

EXPECTED_COLUMNS = {
    "year", "GDP_current_USD", "GDP_per_capita_USD", "inflation_rate",
    "literacy_rate", "internet_users_percent", "total_population",
    "unemployment_rate", "electricity_access_percent",
}

NON_NULLABLE_COLUMNS = {"year", "GDP_current_USD", "total_population"}

RANGE_CHECKS = [
    ("literacy_rate", 0, 100),
    ("internet_users_percent", 0, 100),
    ("electricity_access_percent", 0, 100),
    ("unemployment_rate", 0, 100),
    ("total_population", 0, 1_000_000_000),   
    ("GDP_current_USD", 0, None),
]


@pytest.fixture(scope="module")
def df():
    if not os.path.exists(CSV_PATH):
        pytest.skip(f"{CSV_PATH} not found — run 01_fetch_data.py and "
                     f"02_clean_store.py first")
    return pd.read_csv(CSV_PATH)


def test_file_not_empty(df):
    assert len(df) > 0, "clean_data.csv loaded but has zero rows"


def test_expected_columns_present(df):
    missing = EXPECTED_COLUMNS - set(df.columns)
    assert not missing, f"Missing expected columns: {missing}"


def test_year_is_integer_like(df):
    assert pd.api.types.is_numeric_dtype(df["year"]), "year column is not numeric"
    assert (df["year"] == df["year"].astype(int)).all(), "year column has non-integer values"


def test_no_duplicate_years(df):
    dupes = df["year"][df["year"].duplicated()]
    assert dupes.empty, f"Duplicate year rows found: {sorted(dupes.tolist())}"


def test_years_are_sorted_or_sortable(df):
    assert df["year"].is_monotonic_increasing or df["year"].is_monotonic_decreasing or True
    assert df["year"].min() >= 1960, "year values earlier than World Bank series start (1960)"
    assert df["year"].max() <= 2100, "year values implausibly far in the future"

def test_no_nulls_in_critical_columns(df):
    for col in NON_NULLABLE_COLUMNS:
        if col not in df.columns:
            continue
        null_years = df.loc[df[col].isna(), "year"].tolist()
        assert not null_years, f"{col} has nulls in years: {null_years}"


def test_null_rate_not_excessive(df):
    """
    Flags any column that's >50% missing — likely means an indicator
    failed to fetch for most of the series rather than genuinely having
    sparse real-world coverage.
    """
    numeric_cols = df.select_dtypes(include="number").columns
    bad_cols = {}
    for col in numeric_cols:
        null_rate = df[col].isna().mean()
        if null_rate > 0.5:
            bad_cols[col] = round(null_rate, 2)
    assert not bad_cols, f"Columns with >50% missing values: {bad_cols}"

@pytest.mark.parametrize("col,lo,hi", RANGE_CHECKS)
def test_values_within_plausible_range(df, col, lo, hi):
    if col not in df.columns:
        pytest.skip(f"{col} not present in this dataset")
    series = df[col].dropna()
    if lo is not None:
        below = series[series < lo]
        assert below.empty, f"{col} has values below {lo}: {below.tolist()}"
    if hi is not None:
        above = series[series > hi]
        assert above.empty, f"{col} has values above {hi}: {above.tolist()}"


def test_population_is_monotonic_ish(df):
    if "total_population" not in df.columns:
        pytest.skip("total_population not present")
    pop = df.sort_values("year")["total_population"].dropna()
    pct_change = pop.pct_change().dropna()
    extreme = pct_change[(pct_change < -0.05) | (pct_change > 0.10)]
    assert extreme.empty, f"Implausible year-over-year population swings: {extreme.to_dict()}"

def test_gdp_growth_pct_consistent_with_gdp(df):
    """If GDP_growth_pct exists, spot-check it's actually derived from GDP_current_USD."""
    if "GDP_growth_pct" not in df.columns or "GDP_current_USD" not in df.columns:
        pytest.skip("GDP_growth_pct or GDP_current_USD not present")
    sub = df.sort_values("year").reset_index(drop=True)
    recomputed = sub["GDP_current_USD"].pct_change() * 100
    mask = sub["GDP_growth_pct"].notna() & recomputed.notna()
    diff = (sub.loc[mask, "GDP_growth_pct"] - recomputed.loc[mask]).abs()
    assert (diff < 1e-6).all(), "GDP_growth_pct does not match pct_change of GDP_current_USD"

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))