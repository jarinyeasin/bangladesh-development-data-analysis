import os
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

BASE_URL = "https://api.worldbank.org/v2/country/BD/indicator"

EXPANDED_INDICATORS = {
    # Economy
    "NY.GDP.MKTP.CD": "gdp_current_usd",
    "NY.GDP.PCAP.CD": "gdp_per_capita_usd",
    "NY.GDP.MKTP.KD.ZG": "gdp_growth_annual_pct",
    "FP.CPI.TOTL.ZG": "inflation_pct",
    "NE.EXP.GNFS.ZS": "exports_pct_gdp",
    "NE.IMP.GNFS.ZS": "imports_pct_gdp",
    "BX.KLT.DINV.WD.GD.ZS": "fdi_inflow_pct_gdp",
    "GC.DOD.TOTL.GD.ZS": "govt_debt_pct_gdp",
    # Labour & poverty
    "SL.UEM.TOTL.ZS": "unemployment_pct",
    "SI.POV.NAHC": "poverty_headcount_national_pct",
    "SL.TLF.CACT.ZS": "labor_force_participation_pct",
    "SL.AGR.EMPL.ZS": "employment_in_agriculture_pct",
    # Human capital / education
    "SE.ADT.LITR.ZS": "literacy_rate_pct",
    "SE.PRM.ENRR": "primary_enrollment_pct",
    "SE.SEC.ENRR": "secondary_enrollment_pct",
    "SE.XPD.TOTL.GD.ZS": "education_expenditure_pct_gdp",
    # Health
    "SP.DYN.LE00.IN": "life_expectancy_years",
    "SH.DYN.MORT": "under5_mortality_per1000",
    "SH.XPD.CHEX.GD.ZS": "health_expenditure_pct_gdp",
    "SH.STA.MMRT": "maternal_mortality_per100k",
    "SP.DYN.TFRT.IN": "fertility_rate",
    # Infrastructure & digital
    "IT.NET.USER.ZS": "internet_users_pct",
    "IT.CEL.SETS.P2": "mobile_subscriptions_per100",
    "EG.ELC.ACCS.ZS": "electricity_access_pct",
    "IS.ROD.PAVE.ZS": "paved_roads_pct",
    "EG.USE.PCAP.KG.OE": "energy_use_per_capita",
    # Environment
    "EN.ATM.CO2E.PC": "co2_emissions_per_capita",
    "AG.LND.FRST.ZS": "forest_area_pct",
    "ER.H2O.FWTL.ZS": "freshwater_withdrawal_pct",
    "EN.POP.DNST": "population_density",
    # Demographics & governance context
    "SP.POP.TOTL": "total_population",
    "SP.URB.TOTL.IN.ZS": "urban_population_pct",
    "SP.POP.GROW": "population_growth_pct",
    "IC.BUS.EASE.XQ": "ease_of_doing_business_rank",
    "SI.POV.GINI": "gini_index",
    "GB.XPD.RSDV.GD.ZS": "rnd_expenditure_pct_gdp",
}


def fetch_indicator(indicator_code, indicator_name, mrv=40):
    url = f"{BASE_URL}/{indicator_code}"
    params = {"format": "json", "per_page": 100, "mrv": mrv}
    response = requests.get(url, params=params, timeout=30)
    if response.status_code != 200:
        print(f"  Error fetching {indicator_name}: HTTP {response.status_code}")
        return None
    data = response.json()
    if len(data) < 2 or data[1] is None:
        print(f"  No data for {indicator_name}")
        return None
    records = [
        {"year": int(e["date"]), "value": float(e["value"]), "indicator": indicator_name}
        for e in data[1] if e["value"] is not None
    ]
    return pd.DataFrame(records) if records else None


def fetch_expanded_indicators() -> pd.DataFrame:
    all_data = []
    for code, name in EXPANDED_INDICATORS.items():
        print(f"Fetching {name}...")
        df = fetch_indicator(code, name)
        if df is not None:
            all_data.append(df)
    combined = pd.concat(all_data, ignore_index=True)
    return combined.sort_values(["indicator", "year"])


def prepare_wide_matrix(long_df: pd.DataFrame, min_year_coverage: float = 0.6) -> pd.DataFrame:
    """
    Pivot to wide (year x indicator) and drop indicators with too much
    missing history — PCA on a matrix that's mostly imputed values isn't
    telling you about Bangladesh, it's telling you about the imputer.
    """
    wide = long_df.pivot(index="year", columns="indicator", values="value")
    coverage = wide.notna().mean()
    keep_cols = coverage[coverage >= min_year_coverage].index
    dropped = set(wide.columns) - set(keep_cols)
    if dropped:
        print(f"Dropping {len(dropped)} indicators with <{min_year_coverage:.0%} year coverage: "
              f"{sorted(dropped)}")
    return wide[keep_cols].sort_index()


def run_pca(wide_df: pd.DataFrame, n_components: int = 3):
    """
    Standardize (mean 0, var 1) so indicators on wildly different scales
    (GDP in billions vs. literacy in percent) don't dominate the components
    just by virtue of their units. Missing values are median-imputed per
    column — acceptable for a handful of gaps, not for sparse columns
    (those should already have been dropped by prepare_wide_matrix).
    """
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(wide_df.values)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(X_scaled)

    scores_df = pd.DataFrame(
        scores,
        index=wide_df.index,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )

    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=wide_df.columns,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )

    explained = pd.Series(
        pca.explained_variance_ratio_,
        index=[f"PC{i+1}" for i in range(n_components)],
        name="explained_variance_ratio",
    )

    return scores_df, loadings_df, explained


def top_loadings(loadings_df: pd.DataFrame, component: str, n: int = 5) -> pd.Series:
    """Which original indicators define a component most strongly, either direction."""
    return loadings_df[component].reindex(
        loadings_df[component].abs().sort_values(ascending=False).index
    ).head(n)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    raw_path = "data/raw_expanded_indicators.csv"
    if not os.path.exists(raw_path):
        print(f"Fetching {len(EXPANDED_INDICATORS)} indicators from the World Bank API...\n")
        long_df = fetch_expanded_indicators()
        long_df.to_csv(raw_path, index=False)
        print(f"\nSaved {len(long_df)} records to {raw_path}")
    else:
        print(f"Found existing {raw_path}, skipping fetch (delete the file to re-fetch)")
        long_df = pd.read_csv(raw_path)

    wide_df = prepare_wide_matrix(long_df)
    print(f"\nWide matrix: {wide_df.shape[0]} years x {wide_df.shape[1]} indicators")

    n_components = 3
    scores_df, loadings_df, explained = run_pca(wide_df, n_components=n_components)

    print(f"\nExplained variance by component:")
    print(explained.round(3).to_string())
    print(f"Cumulative variance explained by {n_components} components: "
          f"{explained.sum():.1%}")

    print("\nTop indicators defining each component:")
    for pc in scores_df.columns:
        print(f"\n  {pc}:")
        for ind, loading in top_loadings(loadings_df, pc).items():
            print(f"    {ind:35s} {loading:+.3f}")

    scores_df.to_csv("outputs/pca_development_dimensions.csv")
    loadings_df.to_csv("outputs/pca_loadings.csv")
    explained.to_csv("outputs/pca_explained_variance.csv")
    print("\nSaved: outputs/pca_development_dimensions.csv, "
          "outputs/pca_loadings.csv, outputs/pca_explained_variance.csv")
