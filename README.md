# Bangladesh Development Data Pipeline

**Jarin Binta Yeasin** | Department of Mass Communication & Journalism | University of Dhaka
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org) [![pandas](https://img.shields.io/badge/pandas-2.x-150458?logo=pandas)](https://pandas.pydata.org) [![statsmodels](https://img.shields.io/badge/statsmodels-0.14-8A2BE2)](https://www.statsmodels.org) [![scikit--learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org) [![Streamlit](https://img.shields.io/badge/Streamlit-1.36-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io) [![World Bank](https://img.shields.io/badge/Data-World%20Bank%20Open%20Data-009FDA)](https://data.worldbank.org)

> 🔗 **[View Live Dashboard](https://bangladesh-development-dashboard-mneydmy24fk5pew5drbtkt.streamlit.app/)**

---

## Research Motivation

Bangladesh presents one of the most analytically compelling development stories of the past three decades, a country that has sustained GDP growth rates averaging 6–7% annually while simultaneously navigating acute structural challenges: a population of 174 million in one of the world's most densely settled territories, persistent digital inequality between urban and rural areas, and an economy structurally exposed to climate risk through its low-lying delta geography.

Standard economic reporting on Bangladesh tends toward either uncritical optimism ("the Bengal Tiger economy") or crisis framing around political instability and climate vulnerability. Neither framing is analytically adequate. This project takes a different approach: letting longitudinal World Bank data speak across eight core development dimensions. Across 25 further indicators reduced via PCA, making visible the relationships, and the tensions, that single-indicator reporting obscures.

Three research questions motivate the indicator selection:

1. **Does Bangladesh's aggregate GDP growth translate into proportional gains in human development indicators** — literacy, electricity access, employment or does growth remain structurally concentrated?
2. **How did the COVID-19 shock of 2020 propagate across economic and social indicators**, and what does the recovery trajectory reveal about the resilience of Bangladesh's development model?
3. **Is digital inclusion (internet penetration) tracking with or lagging behind economic growth**, and what does the gap suggest about the distributional reach of the country's "Digital Bangladesh" agenda?

---

## Key Indicators & Analytical Rationale

| Indicator               | World Bank Code | Why Included                                                                   |
| ------------------------ | ---------------- | -------------------------------------------------------------------------------- |
| GDP (current USD)        | NY.GDP.MKTP.CD   | Primary economic output measure; baseline for all relative comparisons          |
| GDP Per Capita (USD)     | NY.GDP.PCAP.CD   | Distributional proxy — divergence from total GDP signals concentration          |
| Inflation Rate (%)       | FP.CPI.TOTL.ZG   | Structural economic health; COVID-era spike is analytically significant         |
| Literacy Rate (%)        | SE.ADT.LITR.ZS   | Human capital formation; lagging indicator of long-run development investment   |
| Internet Users (%)       | IT.NET.USER.ZS   | Digital inclusion proxy; tested against GDP growth for co-movement              |
| Total Population         | SP.POP.TOTL      | Denominator for per-capita metrics; context for unemployment interpretation     |
| Unemployment Rate (%)    | SL.UEM.TOTL.ZS   | Labour market resilience; COVID shock visibility                                |
| Electricity Access (%)   | EG.ELC.ACCS.ZS   | Infrastructure floor for both digital inclusion and economic participation      |

The PCA analysis draws on a further 35 World Bank indicators spanning economy, labour, education, health, infrastructure, and environment.

---

## Selected Findings

### 1. GDP Growth Without Proportional Employment Absorption

Bangladesh's GDP grew from approximately $30B in 1990 to over $400B by 2024, a 13× increase over three decades, placing it among the fastest-growing economies in South Asia. Yet the unemployment rate across the same period remained relatively stable in the 4–5% band, rather than declining as classical development theory would predict. Interpreted alongside population growth from roughly 108 million (1990) to 174 million (2024), it suggests the economy is absorbing a rapidly expanding labour force without generating proportionally more formal employment. The informal sector, not captured in these ILO-modelled estimates, likely explains much of this divergence.

### 2. The COVID-19 Unemployment Signal and Recovery

The 2020 COVID-19 shock is visible in the data as a clear unemployment spike, the rate reached approximately 5.3% in 2020, its highest point in the observed period, rising from 4.22% in 2019. The spike was driven primarily by suspension of non-critical services, particularly hospitality, with partial recovery beginning as early as mid-2020 as informal sector activity resumed. By 2022–2023, the rate had returned to its pre-COVID band. The recovery is real, but its speed reflects the informality of Bangladesh's employment structure as much as it reflects resilience: workers re-entered informal arrangements, not formal employment recovery.

On a year-over-year change basis, 2020's unemployment z-score is **1.85** , below the |z| ≥ 2.0 threshold used elsewhere in this project to flag statistical anomalies. A multivariate Isolation Forest run jointly across seven indicators ranks 2020 **6th of 11** scored years by anomaly score and does not flag it at all. The year that *does* clear both thresholds is **2016**, flagged individually on GDP, GDP per capita, and literacy (three simultaneous z-score flags, the only year with more than two) and the sole year flagged by the joint Isolation Forest. The 2020 unemployment rise is visible and directionally consistent with the COVID narrative, but by these two statistical tests it is not the most anomalous year in the dataset. The year is 2016, and doesn't yet have a story attached to it.

### 3. Digital Inclusion: Rapid Growth, Persistent Structural Gap

Internet penetration in Bangladesh rose from near-zero in 2000 to approximately 40–47% by 2023–2025, driven almost entirely by mobile internet expansion following 4G rollout in 2018. The digital economy contributed an estimated 4.2% of GDP by 2025, exceeding earlier projections. However, the aggregate penetration figure conceals a significant urban-rural divide: urban penetration reached approximately 78% by 2025, while rural penetration lagged at 49%. Electricity access reached approximately 88% nationally by the early 2020s, functions as the infrastructural floor beneath digital inclusion; the remaining 12% without electricity access represents a population structurally excluded from digital participation regardless of mobile network availability.

Electricity access is a statistically significant predictor of literacy (R² = 0.942, coefficient = 0.476, p < 0.001), reinforcing the "infrastructural floor" framing above with a formal test.

### 4. The Literacy-Growth Relationship

Adult literacy improved from approximately 35% in 1990 to over 74% by the early 2020s, a 40-percentage-point gain over three decades. The trajectory tracks closely with GDP per capita growth, suggesting genuine co-movement rather than literacy gains lagging growth (as occurs in economies where growth is primarily resource-extraction driven). This co-movement is consistent with Bangladesh's growth model, which is human-capital intensive through the garment sector and remittance economy, both of which reward basic literacy and numeracy.

---

### Forecasting

Each indicator is backtested on a 5-year holdout using two models, Holt's damped-trend Exponential Smoothing (ETS) and ARIMA(1,1,1) and the model with the lower MAPE on that holdout is refit on the full 34-year series (1981–2025) to produce the forward forecast.

| Indicator | Best Model | RMSE | MAPE |
|---|---|---|---|
| GDP Per Capita (USD) | ARIMA | 163.07 | 5.76% |
| Unemployment Rate (%) | ETS | 0.86 | 19.79% |
| Literacy Rate (%) | ETS | 1.46 | **1.43%** |
| Internet Users (%) | ETS | 2.68 | 4.99% |

Literacy is by far the most forecastable series here, a 1.43% MAPE reflects how slow and structurally steady its historical trend is. Unemployment is the least forecastable (MAPE 19.79–30.48% depending on model), consistent with it being the noisiest, most shock-sensitive indicator in the set.

5-year forecasts (2026–2030): GDP per capita rises from roughly **$2,618 to $2,682**; unemployment holds essentially flat around **3.79%**; literacy climbs steadily to **~79.0%**. Internet penetration is projected to reach **~97.4% by 2030**. ETS extrapolates the strong recent growth trend linearly without any awareness that adoption curves saturate; a 5-year jump from ~54% (2024 actual) to near-total penetration should be read as a model limitation, not a forecast to plan around.

### Regression & Correlation

**Correlation matrix (Pearson, contemporaneous):** Level variables — GDP, GDP per capita, internet users, literacy, electricity access are all pairwise correlated above 0.9, which mostly reflects a shared secular upward trend across three decades rather than a specific causal relationship between any pair. `GDP_growth_pct` (which removes the trend) correlates far more weakly with everything at most 0.44 with unemployment and 0.31 with literacy and this weaker, trend-adjusted correlation is the more informative one.

**Does internet penetration predict GDP growth, with a lag?** Tested at 1, 2, and 3-year lags via separate univariate OLS models (n = 26 for each):

| Lag (years) | Coefficient | p-value | R² |
|---|---|---|---|
| 1 | −0.0996 | 0.267 | 0.051 |
| 2 | −0.1278 | 0.203 | 0.067 |
| 3 | −0.1614 | 0.143 | 0.087 |

**Null result:** none of the three lags reach significance at p < 0.05. The coefficients trend negative and the fit improves slightly as the lag lengthens, but with R² under 0.09 throughout, this is not strong enough to read as a real effect — only as "not detectable at this sample size." A useful null result for a research question the project set out to test, not a finding to force into a positive story.

**Does electricity access predict literacy?** Yes — R² = 0.942, coefficient = 0.476 (p < 0.001, F = 179.1). Each 1-point rise in electricity access is associated with a 0.48-point rise in literacy rate. The fit is strong, but n = 13 (the number of years with non-null data for both variables simultaneously) is thin for an OLS regression; treat the strength of this relationship as suggestive and trend-consistent rather than a settled causal estimate.

### Anomaly Detection

Two independent tests, both run across the 34-year series:

**Z-score of year-over-year change** 11 indicator-year pairs were flagged at |z| ≥ 2.0:

| Year | Indicator | z-score |
|---|---|---|
| 2010 | Unemployment | −3.11 |
| 2012 | Inflation | −2.54 |
| 2015 | Electricity access | +2.37 |
| 2016 | GDP (current USD) | +3.19 |
| 2016 | GDP per capita | +3.29 |
| 2016 | Literacy rate | +2.38 |
| 2017 | Electricity access | +2.50 |
| 2023 | GDP (current USD) | −2.10 |
| 2023 | GDP per capita | −2.29 |
| 2023 | Unemployment | −2.40 |
| 2024 | Internet users | +3.05 |

**2020's unemployment change z-score: 1.85, below threshold.**

**Isolation Forest**, run jointly across all 7 indicators simultaneously (treating each year as a point in indicator-space, using year-over-year % change so the secular growth trend doesn't dominate): only **2016** is flagged as anomalous. **2020 ranks 6th of 11** scored years by anomaly score and is not flagged.

Both tests agree: **2016, not 2020, is the year that stands out most rigorously in this dataset** , individually (three separate indicators flagged the same year, more than any other year) and jointly (the only year the Isolation Forest flags at all). This directly complicates the "clear COVID spike" framing in Finding 2 above. The 2020 unemployment rise is real and visible on a raw chart, but it doesn't clear either statistical bar used elsewhere in this project. What happened in Bangladesh's economy in 2016 that would explain a joint disruption across GDP, GDP per capita, and literacy simultaneously is an open question this project doesn't currently answer.

### PCA-Derived Development Dimensions

35 World Bank indicators across economy, labour, education, health, infrastructure, and environment were pulled and reduced via PCA. 7 indicators were dropped for having less than 60% year coverage in this series, notably including **internet users (%)** and **literacy rate (%)**, meaning the dimensions below are built *without* two of the project's central indicators, since their coverage windows in the expanded 35-indicator pull didn't overlap sufficiently with the rest. (The other 5 dropped: freshwater withdrawal %, Gini index, government debt % GDP, health expenditure % GDP, poverty headcount %.) The remaining 25 indicators, over 56 years, were standardized and reduced to 3 principal components explaining **79.7%** of total variance:

| Component | Variance Explained | Top Loadings |
|---|---|---|
| PC1 | 59.3% | + life expectancy, + urban population %, − maternal mortality, + total population, + population density |
| PC2 | 12.0% | + exports % GDP, + imports % GDP, + FDI inflow % GDP, − labor force participation %, − GDP (current USD) |
| PC3 | 8.4% | + education expenditure % GDP, + secondary enrollment %, + employment in agriculture %, − unemployment %, + forest area % |

Tentative labels, offered as a starting point: **PC1** reads as a general demographic-scale-and-health axis; **PC2** as trade and investment openness, contrasted against labour force participation and raw economic size; **PC3** as an education-investment-and-agrarian-structure axis. Unlike the Human Development Index, which hand-picks three dimensions and a fixed weighting formula, these three emerged from the correlation structure of the data itself, the tradeoff being that they're harder to name cleanly and, per the caveat above, don't directly incorporate two of this project's core indicators.

---

## Pipeline Architecture

```
World Bank Open Data API
         │
         ▼
01_fetch_data.py          ← REST API calls, long-format DataFrame
         │
         ▼
data/raw_world_bank_data.csv
         │
         ▼
02_clean_store.py         ← cleaning, pivot to wide format,
         │                   derived columns, SQLite storage
         ├─► data/clean_data.csv
         └─► data/bangladesh_development.db   (SQLite)
                   │
                   ├─► 03_visualize.py        ← 5 publication-quality PNG charts
                   │         └─► outputs/fig*.png
                   │
                   ├─► 04_dashboard.py        ← Interactive Streamlit dashboard
                   │         └─► live at Streamlit Cloud
                   │           
                   │
                   ├─► 05_forecast.py             ← ETS/ARIMA backtest + 5yr forecast
                   ├─► 06_regression_analysis.py  ← correlation matrix + OLS regressions
                   └─► 07_anomaly_detection.py    ← z-score + Isolation Forest

World Bank Open Data API
         │
         ▼
08_expand_indicators_pca.py
         ├─► data/raw_expanded_indicators.csv
         └─► outputs/pca_development_dimensions.csv, pca_loadings.csv, pca_explained_variance.csv

test_data.py               ← pytest suite: schema, nulls, value ranges, derived-column
                               consistency, run against data/clean_data.csv
```

---

## Project Structure

```
bangladesh-development-data-analysis/
│
├── 01_fetch_data.py             # Phase 1: World Bank API → CSV
├── 02_clean_store.py            # Phase 2: Clean → SQLite database
├── 03_visualize.py              # Phase 3: Generate PNG charts
├── 04_dashboard.py              # Phase 4: Streamlit dashboard
├── 05_forecast.py               # Phase 5: ETS/ARIMA forecasting
├── 06_regression_analysis.py    # Phase 6: OLS regression + correlation
├── 07_anomaly_detection.py      # Phase 7: Z-score + Isolation Forest
├── 08_expand_indicators_pca.py  # Phase 8: Expanded indicators + PCA
├── test_data.py                 # pytest data-quality suite
│
├── data/                        # Created automatically on first run
│   ├── raw_world_bank_data.csv
│   ├── clean_data.csv
│   ├── bangladesh_development.db
│   └── raw_expanded_indicators.csv
│
├── outputs/                     # Created automatically on first run
│   ├── fig1_gdp_trend.png
│   ├── fig2_development_indicators.png
│   ├── fig3_gdp_per_capita.png
│   ├── fig4_gdp_growth.png
│   ├── fig5_population.png
│   ├── forecast_model_comparison.csv
│   ├── forecast_5yr.csv
│   ├── correlation_matrix.csv
│   ├── regression_internet_lag_gdp_growth.csv
│   ├── regression_electricity_literacy_summary.txt
│   ├── anomaly_zscores.csv
│   ├── anomaly_isolation_forest.csv
│   ├── pca_development_dimensions.csv
│   ├── pca_loadings.csv
│   └── pca_explained_variance.csv
│
├── requirements.txt
└── README.md
```
---

## How to Run

### 1. Clone the repository

```
git clone https://github.com/jarinyeasin/bangladesh-development-data-analysis.git
cd bangladesh-development-data-analysis
```

### 2. Create a virtual environment

```
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Run the core pipeline in order

```
python 01_fetch_data.py        # fetch from World Bank API
python 02_clean_store.py       # clean + store in SQLite
python 03_visualize.py         # generate charts
python 05_forecast.py               # ETS/ARIMA backtest + 5-year forecast
python 06_regression_analysis.py    # correlation matrix + OLS regressions
python 07_anomaly_detection.py      # z-score + Isolation Forest anomaly detection
python 08_expand_indicators_pca.py  # 35-indicator pull + PCA
streamlit run 04_dashboard.py  # launch dashboard at localhost:8501
```

### 5. Run the data-quality tests

```
pytest test_data.py -v

```

## Limitations

- **ILO-modelled unemployment estimates** smooth over informal sector volatility, likely underestimating true COVID-era labour market disruption given Bangladesh's large informal economy.
- **Sample resolution** is annual. Quarterly or monthly data would reveal shock propagation and recovery dynamics more precisely, particularly around the 2020 COVID period.
- **Electricity access figures** are self-reported national estimates and may overstate rural access quality.
- **Internet penetration** counts unique SIM connections rather than individual users, potentially double-counting in a market with high multi-SIM usage.
- **Forecasting is trained on ~29–34 annual observations** — enough to backtest reasonably for slow-moving structural indicators like literacy, but the internet-penetration forecast (projecting ~97% by 2030) shows how easily ETS/ARIMA overshoot when a series' recent growth rate can't continue indefinitely. Treat 5-year forecasts as trend extrapolations, not predictions.
- **The lagged internet→GDP-growth regression has low statistical power** (n = 26, R² < 0.09 at every lag tested) — its null result should be read as "not detectable at this sample size and resolution," not as proof of no relationship.
- **The electricity→literacy regression relies on only 13 overlapping non-null years.** The R² = 0.94 fit is genuine but should be weighted against that small n, not treated as a precise causal estimate.
- **The PCA-derived development dimensions exclude internet penetration and literacy rate** — two of this project's three core research indicators because their coverage in the expanded 35-indicator pull fell below the 60% threshold used to filter sparse series. The dimensions describe the other 25 indicators well; they should not be read as a complete development index.
- **Anomaly detection surfaced a genuine tension with this project's own narrative:** 2020 does not clear either statistical anomaly threshold used here, while 2016 clears both. That's reported above rather than reconciled, a fuller explanation of 2016 is a natural next step for this project.

---

## Data Source

World Bank Open Data · [data.worldbank.org/country/BD](https://data.worldbank.org/country/BD) · CC BY 4.0 License

---

## Author

**Jarin Binta Yeasin** | Final-year undergraduate, Mass Communication & Journalism, University of Dhaka
📧 <jarinyeasin@gmail.com> · 🔗 [LinkedIn](https://www.linkedin.com/in/jarin-binta-yeasin-b61b88278) · 🐙 [GitHub](https://github.com/jarinyeasin)
