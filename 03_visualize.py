"""
Reads:  data/clean_data.csv
Writes: outputs/fig1_gdp_trend.png
        outputs/fig2_development_indicators.png
        outputs/fig3_gdp_per_capita.png
        outputs/fig4_gdp_growth.png
        outputs/fig5_population.png

"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

COLORS  = ["#1565C0", "#2E7D32", "#E65100", "#6A1B9A", "#00695C"]
STYLE   = "seaborn-v0_8-whitegrid"
DPI     = 150
OUT_DIR = "outputs"


def load_data(path="data/clean_data.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found. "
            "Please run 02_clean_store.py first."
        )
    return pd.read_csv(path)

def fig1_gdp_trend(df):
    data = df.dropna(subset=["GDP_billions"])

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(
        data["year"], data["GDP_billions"],
        color=COLORS[0], linewidth=2.5, marker="o", markersize=4, label="GDP"
    )
    ax.fill_between(data["year"], data["GDP_billions"], alpha=0.08, color=COLORS[0])

    ax.set_title("Bangladesh GDP 1990–2024", fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("GDP (USD Billions)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0fB"))

    last = data.iloc[-1]
    ax.annotate(
        f"  ${last['GDP_billions']:.1f}B ({int(last['year'])})",
        xy=(last["year"], last["GDP_billions"]),
        fontsize=9, color=COLORS[0],
    )

    plt.tight_layout()
    path = f"{OUT_DIR}/fig1_gdp_trend.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def fig2_development_indicators(df):
    indicators = [
        ("literacy_rate",              "Literacy Rate (%)",        COLORS[0]),
        ("internet_users_percent",     "Internet Users (%)",       COLORS[1]),
        ("electricity_access_percent", "Electricity Access (%)",   COLORS[2]),
        ("unemployment_rate",          "Unemployment Rate (%)",    COLORS[3]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "Bangladesh Development Indicators", fontsize=16, fontweight="bold", y=1.01
    )

    for ax, (col, label, color) in zip(axes.flat, indicators):
        data = df.dropna(subset=[col])
        ax.plot(
            data["year"], data[col],
            color=color, linewidth=2, marker="o", markersize=3
        )
        ax.fill_between(data["year"], data[col], alpha=0.08, color=color)
        ax.set_title(label, fontweight="bold", fontsize=11)
        ax.set_xlabel("Year", fontsize=9)
        ax.set_ylabel("%", fontsize=9)

    plt.tight_layout()
    path = f"{OUT_DIR}/fig2_development_indicators.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def fig3_gdp_per_capita(df):
    data = df.dropna(subset=["GDP_per_capita_USD"])

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.fill_between(data["year"], data["GDP_per_capita_USD"],
                    alpha=0.2, color=COLORS[1])
    ax.plot(data["year"], data["GDP_per_capita_USD"],
            color=COLORS[1], linewidth=2.5, marker="o", markersize=4)

    ax.set_title("Bangladesh GDP Per Capita Growth", fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("USD Per Person", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))

    plt.tight_layout()
    path = f"{OUT_DIR}/fig3_gdp_per_capita.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def fig4_gdp_growth(df):
    data = df.dropna(subset=["GDP_growth_pct"])

    colors = [COLORS[1] if v >= 0 else COLORS[2] for v in data["GDP_growth_pct"]]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(data["year"], data["GDP_growth_pct"], color=colors, width=0.7, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    ax.set_title("Bangladesh Annual GDP Growth Rate (%)", fontsize=15,
                 fontweight="bold", pad=14)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Growth (%)", fontsize=11)

    plt.tight_layout()
    path = f"{OUT_DIR}/fig4_gdp_growth.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def fig5_population(df):
    data = df.dropna(subset=["total_population"])

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(
        data["year"], data["total_population"] / 1e6,
        color=COLORS[4], linewidth=2.5, marker="o", markersize=4
    )
    ax.fill_between(data["year"], data["total_population"] / 1e6,
                    alpha=0.1, color=COLORS[4])

    ax.set_title("Bangladesh Population Growth", fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Population (Millions)", fontsize=11)

    plt.tight_layout()
    path = f"{OUT_DIR}/fig5_population.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")

if __name__ == "__main__":
    print("=" * 55)
    print("  Generating visualizations")
    print("=" * 55)

    os.makedirs(OUT_DIR, exist_ok=True)
    plt.style.use(STYLE)

    df = load_data()

    print(f"\nGenerating {5} charts ...")
    fig1_gdp_trend(df)
    fig2_development_indicators(df)
    fig3_gdp_per_capita(df)
    fig4_gdp_growth(df)
    fig5_population(df)

    print()
    print(f"Charts are in the '{OUT_DIR}/' folder.")
