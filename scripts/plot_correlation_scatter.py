import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.io.eweld_loader import (
    load_electricity,
    load_weather,
    get_user_cluster,
    get_weather_station,
    load_extreme_weather,
)
from src.io.eweld_merger import merge_electricity_weather, aggregate_daily

USER_ID = "U317"

SEASON_COLORS = {
    "Spring": "#2ecc71",
    "Summer": "#e74c3c",
    "Fall": "#e67e22",
    "Winter": "#3498db",
}

DAYTYPE_COLORS = {
    "Weekday": "#e74c3c",
    "Weekend": "#3498db",
    "Holiday": "#2ecc71",
}

CONDITION_ORDER = [
    "Normal",
    "Low Temp",
    "High Humidity",
    "High Temp",
    "High Heat+Humidity",
    "Storm/Typhoon",
]
CONDITION_COLORS = {
    "Normal": "#95a5a6",
    "Low Temp": "#1abc9c",
    "High Humidity": "#3498db",
    "High Temp": "#e74c3c",
    "High Heat+Humidity": "#8e44ad",
    "Storm/Typhoon": "#2c3e50",
}

CONDITION_PRIORITY = {
    "Storm/Typhoon": 5,
    "High Heat+Humidity": 4,
    "High Temp": 3,
    "High Humidity": 2,
    "Low Temp": 1,
    "Normal": 0,
}

PREFIX_TO_CONDITION = {}
for p in ("08", "09", "10", "11"):
    PREFIX_TO_CONDITION[p] = "Storm/Typhoon"
PREFIX_TO_CONDITION["04"] = "High Heat+Humidity"
PREFIX_TO_CONDITION["02"] = "High Temp"
PREFIX_TO_CONDITION["03"] = "High Humidity"
PREFIX_TO_CONDITION["01"] = "Low Temp"


def _assign_season(daily: pd.DataFrame) -> pd.DataFrame:
    month = daily.index.month
    season = pd.Series("Winter", index=daily.index)
    season[(month >= 3) & (month <= 5)] = "Spring"
    season[(month >= 6) & (month <= 8)] = "Summer"
    season[(month >= 9) & (month <= 11)] = "Fall"
    daily["season"] = season
    return daily


def _assign_weather_condition(
    daily: pd.DataFrame, ew_df: pd.DataFrame
) -> pd.DataFrame:
    day_condition: dict[pd.Timestamp, str] = {}

    for _, row in ew_df.iterrows():
        weather_str = str(row["Weather"])
        prefix = weather_str[:2]
        condition = PREFIX_TO_CONDITION.get(prefix)
        if condition is None:
            continue

        start = pd.Timestamp(row["Start Time"]).normalize()
        end = pd.Timestamp(row["End Time"]).normalize()
        for d in pd.date_range(start, end, freq="D"):
            existing = day_condition.get(d, "Normal")
            if CONDITION_PRIORITY[condition] > CONDITION_PRIORITY[existing]:
                day_condition[d] = condition

    daily["weather_condition"] = daily.index.normalize().map(
        lambda d: day_condition.get(d, "Normal")
    )
    return daily


def plot_kwh_over_time(daily: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.scatter(daily.index, daily["kWh"], alpha=0.4, s=6, c="steelblue", zorder=2)
    rolling = daily["kWh"].rolling(30, center=True).mean()
    ax.plot(daily.index, rolling, color="#e74c3c", linewidth=1.5,
            label="30-day rolling mean", zorder=3)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Daily Electricity Consumption (kWh)", fontsize=12)
    ax.set_title(f"Daily Electricity Consumption Over Time — {USER_ID}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "scatter_kwh_over_time.png", dpi=150)
    plt.close(fig)
    print("Saved scatter_kwh_over_time.png")


def plot_kwh_vs_temp(daily: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    for season, color in SEASON_COLORS.items():
        mask = daily["season"] == season
        subset = daily[mask]
        ax.scatter(subset["temp_f"], subset["kWh"], alpha=0.35, s=10,
                   c=color, label=f"{season} ({len(subset)} days)", zorder=2)
    ax.set_xlabel("Daily Mean Temperature (°F)", fontsize=12)
    ax.set_ylabel("Daily Electricity Consumption (kWh)", fontsize=12)
    ax.set_title(f"Temperature vs Consumption by Season — {USER_ID}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "scatter_kwh_vs_temp.png", dpi=150)
    plt.close(fig)
    print("Saved scatter_kwh_vs_temp.png")


def plot_kwh_vs_humidity(daily: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(daily["humidity"], daily["kWh"], alpha=0.3, s=8, c="steelblue")
    corr = daily["humidity"].corr(daily["kWh"])
    ax.text(0.02, 0.97, f"r = {corr:.3f}", transform=ax.transAxes,
            va="top", fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.set_xlabel("Daily Mean Humidity (%)", fontsize=12)
    ax.set_ylabel("Daily Electricity Consumption (kWh)", fontsize=12)
    ax.set_title(f"Humidity vs Consumption — {USER_ID}", fontsize=13)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "scatter_kwh_vs_humidity.png", dpi=150)
    plt.close(fig)
    print("Saved scatter_kwh_vs_humidity.png")


def plot_kwh_vs_daytype(daily: pd.DataFrame, output_dir: Path) -> None:
    np.random.seed(42)

    groups = {
        "Weekday": daily[(daily["day_of_week"] < 5) & (daily["is_holiday"] == 0)],
        "Weekend": daily[(daily["day_of_week"] >= 5) & (daily["is_holiday"] == 0)],
        "Holiday": daily[daily["is_holiday"] == 1],
    }

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, (label, subset) in enumerate(groups.items()):
        jitter = np.random.uniform(-0.2, 0.2, len(subset))
        marker = "D" if label == "Holiday" else "o"
        alpha = 0.6 if label == "Holiday" else 0.2
        s = 15 if label == "Holiday" else 8
        edgecolors = "black" if label == "Holiday" else "none"
        linewidths = 0.3 if label == "Holiday" else 0
        ax.scatter(i + jitter, subset["kWh"], alpha=alpha, s=s, marker=marker,
                   c=DAYTYPE_COLORS[label], edgecolors=edgecolors,
                   linewidths=linewidths, zorder=2)
        median_val = subset["kWh"].median()
        ax.hlines(median_val, i - 0.3, i + 0.3, colors=DAYTYPE_COLORS[label],
                  linewidths=2.5, zorder=4)
        ax.text(i + 0.35, median_val, f"{median_val:.0f}",
                va="center", fontsize=9, color=DAYTYPE_COLORS[label], fontweight="bold")

    labels = [f"{name}\n(n={len(groups[name])})" for name in groups]
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Daily Electricity Consumption (kWh)", fontsize=12)
    ax.set_title(f"Electricity Consumption by Day Type — {USER_ID}", fontsize=13)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "scatter_kwh_vs_daytype.png", dpi=150)
    plt.close(fig)
    print("Saved scatter_kwh_vs_daytype.png")


def plot_kwh_vs_weather_condition(daily: pd.DataFrame, output_dir: Path) -> None:
    np.random.seed(42)

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, cond in enumerate(CONDITION_ORDER):
        subset = daily[daily["weather_condition"] == cond]
        if subset.empty:
            continue
        jitter = np.random.uniform(-0.25, 0.25, len(subset))
        ax.scatter(i + jitter, subset["kWh"], alpha=0.25, s=8,
                   c=CONDITION_COLORS[cond], zorder=2)
        median_val = subset["kWh"].median()
        ax.hlines(median_val, i - 0.35, i + 0.35, colors=CONDITION_COLORS[cond],
                  linewidths=2.5, zorder=4)
        ax.text(i + 0.4, median_val, f"{median_val:.0f}",
                va="center", fontsize=9, color=CONDITION_COLORS[cond], fontweight="bold")

    labels = []
    for cond in CONDITION_ORDER:
        n = (daily["weather_condition"] == cond).sum()
        labels.append(f"{cond}\n(n={n})")
    ax.set_xticks(range(len(CONDITION_ORDER)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Daily Electricity Consumption (kWh)", fontsize=12)
    ax.set_title(
        f"Electricity Consumption by Weather Condition (EWELD Events) — {USER_ID}",
        fontsize=13,
    )
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "scatter_kwh_vs_weather_condition.png", dpi=150)
    plt.close(fig)
    print("Saved scatter_kwh_vs_weather_condition.png")


if __name__ == "__main__":
    output_dir = Path("results/eweld/degree_day")
    output_dir.mkdir(parents=True, exist_ok=True)

    cluster = get_user_cluster(USER_ID)
    station = get_weather_station(cluster)
    elec_df = load_electricity(USER_ID)
    weather_df = load_weather(station)
    merged = merge_electricity_weather(elec_df, weather_df)
    daily = aggregate_daily(merged)

    _assign_season(daily)

    ew_df = load_extreme_weather(cluster)
    _assign_weather_condition(daily, ew_df)

    plot_kwh_over_time(daily, output_dir)
    plot_kwh_vs_temp(daily, output_dir)
    plot_kwh_vs_humidity(daily, output_dir)
    plot_kwh_vs_daytype(daily, output_dir)
    plot_kwh_vs_weather_condition(daily, output_dir)

    print(f"\nAll 5 scatter plots saved to {output_dir}/")
