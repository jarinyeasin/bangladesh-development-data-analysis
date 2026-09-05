import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Bangladesh Development Dashboard",
    page_icon="🇧🇩",
    layout="wide",
)

DB_PATH = "data/bangladesh_development.db"

@st.cache_data
def load_data():
    import requests
    BASE_URL = "https://api.worldbank.org/v2/country/BD/indicator"
    INDICATORS = {
        "NY.GDP.MKTP.CD": "GDP_current_USD",
        "NY.GDP.PCAP.CD": "GDP_per_capita_USD",
        "FP.CPI.TOTL.ZG": "inflation_rate",
        "SE.ADT.LITR.ZS": "literacy_rate",
        "IT.NET.USER.ZS": "internet_users_percent",
        "SP.POP.TOTL": "total_population",
        "SL.UEM.TOTL.ZS": "unemployment_rate",
        "EG.ELC.ACCS.ZS": "electricity_access_percent",
    }

    all_data = []
    for code, name in INDICATORS.items():
        url = f"{BASE_URL}/{code}"
        params = {"format": "json", "per_page": 100, "mrv": 30}
        try:
            response = requests.get(url, params=params, timeout=15)
            data = response.json()
            if len(data) >= 2 and data[1]:
                for entry in data[1]:
                    if entry["value"] is not None:
                        all_data.append({
                            "year": int(entry["date"]),
                            name: float(entry["value"])
                        })
        except Exception:
            pass

    df = pd.DataFrame(all_data)
    df = df.groupby("year").first().reset_index().sort_values("year")

    if "GDP_current_USD" in df.columns:
        df["GDP_billions"] = df["GDP_current_USD"] / 1e9
        df["GDP_growth_pct"] = df["GDP_current_USD"].pct_change() * 100

    return df

df = load_data()

FLAG_URL = "https://upload.wikimedia.org/wikipedia/commons/f/f9/Flag_of_Bangladesh.svg"

st.markdown(f"""
<style>
    .block-container {{ padding-top: 2rem; }}
    .hero {{
        position: relative;
        border-radius: 14px;
        overflow: hidden;
        padding: 2.6rem 2.4rem;
        margin-bottom: 1.6rem;
        background-color: #0B1410;
        background-image:
            linear-gradient(100deg, rgba(11,20,16,0.95) 0%, rgba(11,20,16,0.88) 100%),
            url('{FLAG_URL}');
        background-size: cover, cover;
        background-position: center, center;
        background-repeat: no-repeat, no-repeat;
        border: 1px solid rgba(0,106,78,0.35);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 2rem;
        flex-wrap: wrap;
    }}
.hero-text {{ flex: 1 1 380px; }}
.hero-video {{
    flex: 0 0 auto;
    position: relative;
    display: block;
    width: 320px;
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid rgba(0,106,78,0.4);
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
}}
.hero-video img {{ width: 100%; display: block; }}
.hero-video::after {{
    content: "▶";
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    width: 56px; height: 56px;
    background: rgba(0,0,0,0.65);
    border-radius: 50%;
    color: #fff;
    font-size: 1.4rem;
    display: flex;
    align-items: center;
    justify-content: center;
    padding-left: 4px;
}}
    .hero h1 {{
        font-size: 2.15rem;
        font-weight: 650;
        color: #F4F6F5;
        margin: 0 0 0.55rem 0;
        line-height: 1.2;
        letter-spacing: -0.01em;
    }}
    .hero p.sub {{
        font-size: 1.02rem;
        color: #C7D1CC;
        max-width: 640px;
        margin: 0 0 1.2rem 0;
        line-height: 1.55;
    }}
    .hero .meta {{
        font-size: 0.82rem;
        color: #82988B;
        border-top: 1px solid rgba(126,145,134,0.25);
        padding-top: 0.85rem;
        margin-top: 0.1rem;
    }}
    .hero .meta a {{ color: #4CAF7D; text-decoration: none; }}
    .hero .meta a:hover {{ text-decoration: underline; }}
</style>

<div class="hero">
    <div class="hero-text">
        <h1>Analysis of Bangladesh Development Indicators</h1>
        <p class="sub">
            Economic and social development data from the World Bank, explored,
            forecast, and tested for statistical significance.
        </p>
        <div class="meta">
            Data from <a href="https://data.worldbank.org/country/BD" target="_blank">World Bank Open Data</a>
        </div>
    </div>
    <a class="hero-video" href="https://youtu.be/rJ2z3GSdVak" target="_blank">
        <img src="https://img.youtube.com/vi/rJ2z3GSdVak/maxresdefault.jpg" alt="Watch the project walkthrough on YouTube">
        <div style="color:#9AA3C7; font-size:0.72rem; text-align:center; margin-top:6px;
                            text-transform:uppercase; letter-spacing:0.05em;">
                    ▶ Watch: Project Walkthrough
        </div>
    </a>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

def latest_value(col_name):
    data = df.dropna(subset=[col_name])
    if data.empty:
        return None, None
    row = data.iloc[-1]
    return row[col_name], int(row["year"])

gdp, gdp_yr = latest_value("GDP_billions")
lit, lit_yr = latest_value("literacy_rate")
inet, inet_yr = latest_value("internet_users_percent")
elec, elec_yr = latest_value("electricity_access_percent")

with col1:
    if gdp is not None:
        st.metric(f"GDP ({gdp_yr})", f"${gdp:.1f}B")
with col2:
    if lit is not None:
        st.metric(f"Literacy Rate ({lit_yr})", f"{lit:.1f}%")
with col3:
    if inet is not None:
        st.metric(f"Internet Users ({inet_yr})", f"{inet:.1f}%")
with col4:
    if elec is not None:
        st.metric(f"Electricity Access ({elec_yr})", f"{elec:.1f}%")

st.markdown("---")

st.subheader("Explore Any Indicator Over Time")

INDICATOR_OPTIONS = {
    "GDP (USD Billions)": "GDP_billions",
    "GDP Per Capita (USD)": "GDP_per_capita_USD",
    "GDP Annual Growth (%)": "GDP_growth_pct",
    "Literacy Rate (%)": "literacy_rate",
    "Internet Users (%)": "internet_users_percent",
    "Inflation Rate (%)": "inflation_rate",
    "Unemployment Rate (%)": "unemployment_rate",
    "Electricity Access (%)": "electricity_access_percent",
    "Population (Millions)": "total_population",
}

left, right = st.columns([1, 3])

with left:
    selected_label = st.selectbox("Select indicator:", list(INDICATOR_OPTIONS.keys()))
    selected_col = INDICATOR_OPTIONS[selected_label]
    chart_type = st.radio("Chart type:", ["Line", "Bar", "Area"])

    data_for_slider = df.dropna(subset=[selected_col])
    if not data_for_slider.empty:
        min_yr = int(data_for_slider["year"].min())
        max_yr = int(data_for_slider["year"].max())
        year_range = st.slider("Year range:", min_yr, max_yr, (min_yr, max_yr))
    else:
        year_range = (1990, 2024)

with right:
    plot_data = df.dropna(subset=[selected_col])
    plot_data = plot_data[
        (plot_data["year"] >= year_range[0]) &
        (plot_data["year"] <= year_range[1])
    ]

    y_col = selected_col
    if selected_col == "total_population":
        plot_data = plot_data.copy()
        plot_data["total_population"] = plot_data["total_population"] / 1e6

    if not plot_data.empty:
        title = f"{selected_label} — Bangladesh {year_range[0]}–{year_range[1]}"
        if chart_type == "Line":
            fig = px.line(plot_data, x="year", y=y_col, title=title, markers=True)
        elif chart_type == "Bar":
            fig = px.bar(plot_data, x="year", y=y_col, title=title)
        else:
            fig = px.area(plot_data, x="year", y=y_col, title=title)

        fig.update_layout(
            xaxis_title="Year",
            yaxis_title=selected_label,
            hovermode="x unified",
            height=420,
        )
        st.plotly_chart(fig, width='stretch')
    else:
        st.warning("No data available for the selected indicator and year range.")

st.markdown("---")

st.subheader("Overview Charts")

chart_files = {
    "GDP Trend": "outputs/fig1_gdp_trend.png",
    "Development Indicators": "outputs/fig2_development_indicators.png",
    "GDP Per Capita": "outputs/fig3_gdp_per_capita.png",
    "Annual GDP Growth": "outputs/fig4_gdp_growth.png",
    "Population Growth": "outputs/fig5_population.png",
}

available = {k: v for k, v in chart_files.items() if os.path.exists(v)}

if available:
    tabs = st.tabs(list(available.keys()))
    for tab, (name, path) in zip(tabs, available.items()):
        with tab:
            st.image(path, width='stretch')
else:
    st.info("Run 03_visualize.py to generate the overview charts.")

st.markdown("---")

st.subheader("Advanced Analytics")
st.markdown(
    "Predictive and inferential layers on top of the descriptive charts above — "
    "forecasting, correlation/regression, anomaly detection, and PCA-derived "
    "development dimensions."
)

PERCENT_BOUNDED = {"unemployment_rate", "literacy_rate", "internet_users_percent",
                    "electricity_access_percent"}


@st.cache_data
def fit_ets_forecast(values, holdout_or_horizon):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    fit = ExponentialSmoothing(np.array(values), trend="add", damped_trend=True).fit(optimized=True)
    return fit.forecast(holdout_or_horizon).tolist()


@st.cache_data
def fit_arima_forecast(values, holdout_or_horizon, order=(1, 1, 1)):
    from statsmodels.tsa.arima.model import ARIMA
    fit = ARIMA(np.array(values), order=order).fit()
    return fit.forecast(holdout_or_horizon).tolist()


def _mape(a, p):
    a, p = np.asarray(a), np.asarray(p)
    return float(np.mean(np.abs((a - p) / a)) * 100)


def _rmse(a, p):
    a, p = np.asarray(a), np.asarray(p)
    return float(np.sqrt(np.mean((a - p) ** 2)))


@st.cache_data
def backtest_and_forecast(series_values, series_years, col_name, holdout=5, horizon=5):
    series = pd.Series(series_values, index=series_years)
    train, test = series.iloc[:-holdout], series.iloc[-holdout:]

    scores = {}
    for name, fn in [("ETS", fit_ets_forecast), ("ARIMA", fit_arima_forecast)]:
        try:
            preds = fn(train.values.tolist(), holdout)
            scores[name] = {"rmse": _rmse(test.values, preds), "mape": _mape(test.values, preds)}
        except Exception:
            scores[name] = {"rmse": np.nan, "mape": np.nan}

    valid = {k: v for k, v in scores.items() if not np.isnan(v["mape"])}
    best = min(valid, key=lambda k: valid[k]["mape"]) if valid else "ETS"
    fn = fit_ets_forecast if best == "ETS" else fit_arima_forecast
    forecast_vals = fn(series.values.tolist(), horizon)

    if col_name in PERCENT_BOUNDED:
        forecast_vals = [min(max(float(v), 0.0), 100.0) for v in forecast_vals]

    return scores, best, forecast_vals


@st.cache_data
def compute_zscores(_df, col):
    diffs = _df.set_index("year")[col].diff()
    m, s = diffs.mean(), diffs.std()
    z = (diffs - m) / s if s and not np.isnan(s) else diffs * np.nan
    return z.reset_index().rename(columns={col: "change"}).assign(z=z.values)


@st.cache_data
def run_isolation_forest(_df, indicators, contamination=0.1):
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    feat = _df[indicators].copy()
    pct = feat.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    years_used = _df.loc[pct.index, "year"].values
    X = StandardScaler().fit_transform(pct.values)
    model = IsolationForest(contamination=contamination, random_state=42, n_estimators=200).fit(X)
    return pd.DataFrame({
        "year": years_used,
        "anomaly_score": model.decision_function(X),
        "is_anomaly": model.predict(X) == -1,
    }).sort_values("anomaly_score")


@st.cache_data
def lagged_ols_summary(_df, source, target, max_lag=3):
    import statsmodels.api as sm
    out = _df[["year", target]].copy()
    for lag in range(1, max_lag + 1):
        out[f"lag{lag}"] = _df[source].shift(lag)
    out = out.dropna()
    rows = []
    for lag in range(1, max_lag + 1):
        col = f"lag{lag}"
        sub = out[["year", target, col]].dropna()
        if len(sub) < 8:
            continue
        X = sm.add_constant(sub[col])
        model = sm.OLS(sub[target], X).fit()
        rows.append({
            "lag_years": lag,
            "coefficient": round(model.params[col], 4),
            "p_value": round(model.pvalues[col], 4),
            "r_squared": round(model.rsquared, 4),
            "n_obs": int(model.nobs),
            "significant_at_05": bool(model.pvalues[col] < 0.05),
        })
    return pd.DataFrame(rows)


@st.cache_data
def simple_ols_fit(_df, x_col, y_col):
    import statsmodels.api as sm
    sub = _df[[x_col, y_col]].dropna()
    X = sm.add_constant(sub[x_col])
    model = sm.OLS(sub[y_col], X).fit()
    return {
        "coef": float(model.params[x_col]),
        "intercept": float(model.params["const"]),
        "r_squared": float(model.rsquared),
        "p_value": float(model.pvalues[x_col]),
        "n": int(model.nobs),
    }


analytics_tabs = st.tabs([
    "Forecast", "Regression & Correlation", "Anomaly Detection", "PCA Dimensions"
])

with analytics_tabs[0]:
    st.markdown(
        "ETS (Holt's damped trend) and ARIMA(1,1,1) are backtested on a 5-year "
        "holdout; the model with the lower MAPE on that holdout is refit on the "
        "full series and used for the forward forecast."
    )
    forecast_indicator_options = {
        "GDP Per Capita (USD)": "GDP_per_capita_USD",
        "Unemployment Rate (%)": "unemployment_rate",
        "Literacy Rate (%)": "literacy_rate",
        "Internet Users (%)": "internet_users_percent",
    }
    fc_label = st.selectbox("Indicator to forecast:", list(forecast_indicator_options.keys()), key="fc_select")
    fc_col = forecast_indicator_options[fc_label]

    series_df = df.dropna(subset=[fc_col]).sort_values("year")
    if len(series_df) >= 13:
        scores, best_model, forecast_vals = backtest_and_forecast(
            series_df[fc_col].values.tolist(), series_df["year"].values.tolist(), fc_col
        )

        acc_col1, acc_col2 = st.columns(2)
        with acc_col1:
            st.markdown("**Backtest accuracy (5-year holdout)**")
            acc_df = pd.DataFrame([
                {"model": k, "RMSE": round(v["rmse"], 3) if not np.isnan(v["rmse"]) else None,
                 "MAPE (%)": round(v["mape"], 2) if not np.isnan(v["mape"]) else None}
                for k, v in scores.items()
            ])
            st.dataframe(acc_df, hide_index=True, width='stretch')
            st.caption(f"Selected model: **{best_model}** (lower MAPE)")

        with acc_col2:
            last_year = int(series_df["year"].max())
            future_years = list(range(last_year + 1, last_year + 6))
            fc_table = pd.DataFrame({"year": future_years, fc_label: [round(v, 2) for v in forecast_vals]})
            st.markdown("**5-year forecast**")
            st.dataframe(fc_table, hide_index=True, width='stretch')

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series_df["year"], y=series_df[fc_col],
                                  mode="lines+markers", name="Historical"))
        fig.add_trace(go.Scatter(x=future_years, y=forecast_vals,
                                  mode="lines+markers", name=f"Forecast ({best_model})",
                                  line=dict(dash="dash")))
        fig.update_layout(title=f"{fc_label} — Historical + 5-Year Forecast",
                           xaxis_title="Year", yaxis_title=fc_label, height=420,
                           hovermode="x unified")
        st.plotly_chart(fig, width='stretch')
    else:
        st.warning("Not enough historical data to backtest reliably.")

with analytics_tabs[1]:
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in ("year", "GDP_billions")]
    corr = df[numeric_cols].corr()

    st.markdown("**Correlation matrix** (Pearson, contemporaneous — no lag)")
    fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
                          zmin=-1, zmax=1)
    fig_corr.update_layout(height=450)
    st.plotly_chart(fig_corr, width='stretch')

    reg_col1, reg_col2 = st.columns(2)

    with reg_col1:
        st.markdown("**Does internet penetration predict GDP growth, with a lag?**")
        if {"internet_users_percent", "GDP_growth_pct"}.issubset(df.columns):
            lag_summary = lagged_ols_summary(df, "internet_users_percent", "GDP_growth_pct")
            if not lag_summary.empty:
                st.dataframe(lag_summary, hide_index=True, width='stretch')
                any_sig = lag_summary["significant_at_05"].any()
                st.caption("At least one lag is significant at p<0.05." if any_sig
                           else "No lag reaches significance at p<0.05 in this sample.")
            else:
                st.info("Not enough overlapping observations.")

    with reg_col2:
        st.markdown("**Does electricity access predict literacy?**")
        if {"electricity_access_percent", "literacy_rate"}.issubset(df.columns):
            result = simple_ols_fit(df, "electricity_access_percent", "literacy_rate")
            st.metric("R²", f"{result['r_squared']:.3f}")
            st.write(f"Coefficient: **{result['coef']:.3f}** (p = {result['p_value']:.4f}, n = {result['n']})")
            st.caption(
                f"Each 1-point rise in electricity access is associated with a "
                f"{result['coef']:.2f}-point rise in literacy rate, holding the "
                f"linear trend fixed."
            )

            scatter_df = df.dropna(subset=["electricity_access_percent", "literacy_rate"])
            fig_sc = px.scatter(scatter_df, x="electricity_access_percent", y="literacy_rate",
                                 trendline="ols", trendline_color_override="firebrick")
            fig_sc.update_layout(height=320)
            st.plotly_chart(fig_sc, width='stretch')

with analytics_tabs[2]:
    st.markdown(
        "Z-scores are computed on the **year-over-year change** in each indicator "
        "(not the level), so a steadily rising series like GDP doesn't get flagged "
        "just for hitting a new high."
    )

    anomaly_indicator_options = {
        "Unemployment Rate": "unemployment_rate",
        "GDP Per Capita": "GDP_per_capita_USD",
        "Literacy Rate": "literacy_rate",
        "Internet Users": "internet_users_percent",
    }
    az_label = st.selectbox("Indicator:", list(anomaly_indicator_options.keys()), key="az_select")
    az_col = anomaly_indicator_options[az_label]

    if az_col in df.columns:
        z_df = compute_zscores(df, az_col).dropna()
        fig_z = px.bar(z_df, x="year", y="z", title=f"{az_label} — Z-score of Year-over-Year Change")
        fig_z.add_hline(y=2, line_dash="dash", line_color="firebrick")
        fig_z.add_hline(y=-2, line_dash="dash", line_color="firebrick")
        fig_z.update_layout(height=380, yaxis_title="Z-score")
        st.plotly_chart(fig_z, width='stretch')

        row_2020 = z_df[z_df["year"] == 2020]
        if not row_2020.empty:
            z_val = row_2020["z"].iloc[0]
            st.metric("2020 z-score", f"{z_val:.2f}",
                      delta="Flagged anomalous" if abs(z_val) >= 2 else "Within normal range")

    st.markdown("---")
    st.markdown("**Multivariate anomaly detection** (Isolation Forest, all indicators jointly)")
    iso_indicators = [c for c in [
        "GDP_current_USD", "literacy_rate", "internet_users_percent",
        "unemployment_rate", "electricity_access_percent", "inflation_rate",
    ] if c in df.columns]

    if len(iso_indicators) >= 3:
        iso_df = run_isolation_forest(df, iso_indicators)
        fig_iso = px.bar(iso_df, x="year", y="anomaly_score", color="is_anomaly",
                          color_discrete_map={True: "firebrick", False: "steelblue"},
                          title="Anomaly Score by Year (lower = more anomalous)")
        fig_iso.update_layout(height=380)
        st.plotly_chart(fig_iso, width='stretch')

        if 2020 in iso_df["year"].values:
            rank = int(iso_df.reset_index(drop=True).index[iso_df["year"] == 2020][0]) + 1
            flagged = bool(iso_df.loc[iso_df["year"] == 2020, "is_anomaly"].iloc[0])
            st.caption(f"2020 ranks #{rank} of {len(iso_df)} years by joint anomaly score "
                       f"across {len(iso_indicators)} indicators simultaneously. "
                       f"Flagged: **{flagged}**.")

with analytics_tabs[3]:
    st.markdown(
        "Data-driven development dimensions from ~35 World Bank indicators, reduced "
        "via PCA — the same technique used for the PCA work in the wellbeing-survey "
        "project, applied here to macro data. This runs offline "
        "(`08_expand_indicators_pca.py`) since it needs a separate 35-indicator API "
        "pull that's too heavy to repeat on every dashboard load; commit its outputs "
        "to `outputs/` to show them here."
    )

    pca_scores_path = "outputs/pca_development_dimensions.csv"
    pca_loadings_path = "outputs/pca_loadings.csv"
    pca_variance_path = "outputs/pca_explained_variance.csv"

    if all(os.path.exists(p) for p in [pca_scores_path, pca_loadings_path, pca_variance_path]):
        pca_scores = pd.read_csv(pca_scores_path, index_col=0)
        pca_loadings = pd.read_csv(pca_loadings_path, index_col=0)
        pca_variance = pd.read_csv(pca_variance_path, index_col=0)

        var_col1, var_col2 = st.columns([1, 2])
        with var_col1:
            st.markdown("**Variance explained**")
            st.dataframe(pca_variance, width='stretch')
        with var_col2:
            if {"PC1", "PC2"}.issubset(pca_scores.columns):
                fig_pca = px.scatter(
                    pca_scores.reset_index(), x="PC1", y="PC2",
                    color=pca_scores.index if pca_scores.index.name else None,
                    text=pca_scores.index,
                    title="Bangladesh's Development Trajectory in PCA Space",
                )
                fig_pca.update_traces(textposition="top center")
                fig_pca.update_layout(height=420)
                st.plotly_chart(fig_pca, width='stretch')

        st.markdown("**Top indicators defining each component**")
        for pc in pca_loadings.columns:
            top5 = pca_loadings[pc].reindex(pca_loadings[pc].abs().sort_values(ascending=False).index).head(5)
            with st.expander(f"{pc}"):
                st.dataframe(top5.rename("loading").reset_index().rename(columns={"index": "indicator"}),
                             hide_index=True, width='stretch')
    else:
        st.info(
            "PCA outputs not found. Run `python 08_expand_indicators_pca.py` locally, "
            "then commit the three files it writes to `outputs/` "
            "(`pca_development_dimensions.csv`, `pca_loadings.csv`, "
            "`pca_explained_variance.csv`) so they render here."
        )

st.markdown("---")

with st.expander("View Raw Data Table"):
    display_cols = ["year"] + [
        c for c in df.columns
        if c not in ("year", "country") and not df[c].isna().all()
    ]
    st.dataframe(
        df[display_cols].sort_values("year", ascending=False),
        width='stretch',
        height=350,
    )

    csv = df[display_cols].to_csv(index=False)
    st.download_button(
        label="⬇ Download CSV",
        data=csv,
        file_name="bangladesh_development_data.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption(
    "Data: World Bank Open Data (CC BY 4.0) · "
    "Built by Jarin Binta Yeasin · "
    "An Individual Project on Bangladesh Data Pipeline"
)
