import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.io.eweld_loader import load_electricity, load_weather, get_user_cluster, get_weather_station
from src.io.eweld_merger import merge_electricity_weather, aggregate_daily
from src.features.degree_day import find_balance_point, build_dd_features, build_raw_features
from src.models.baseline import (
    OLSModel, RidgeModel, LGBMModel, ChangePoint5PModel, iterative_clean,
)
import pandas as pd

user_id = "U317"
eta = 3.0
output_dir = Path("results/eweld/degree_day")

cluster = get_user_cluster(user_id)
station = get_weather_station(cluster)
elec_df = load_electricity(user_id)
weather_df = load_weather(station)
merged = merge_electricity_weather(elec_df, weather_df)
daily = aggregate_daily(merged)

bp = find_balance_point(daily)
print(f"Balance point: {bp:.1f}°F ({(bp - 32) * 5/9:.1f}°C)")

X_dd, dd_names = build_dd_features(daily, bp)
X_raw, raw_names, cat_idx = build_raw_features(daily)
y = daily["kWh"].values
temps = daily["temp_f"].values

models = {}

# OLS
m = OLSModel()
m.fit(X_dd, y)
models["OLS"] = m.predict(X_dd)

# Ridge
m = RidgeModel()
m.fit(X_dd, y)
models["Ridge"] = m.predict(X_dd)

# LightGBM
m = LGBMModel()
m.fit(X_raw, y, categorical_indices=cat_idx)
models["LightGBM"] = m.predict(X_raw)

# 5P
dow = pd.get_dummies(daily["day_of_week"], prefix="dow", dtype=float).iloc[:, :-1].values
hol = daily["is_holiday"].values.reshape(-1, 1)
hum = daily["humidity"].values.reshape(-1, 1)
res_X = np.column_stack([dow, hol, hum])

m = ChangePoint5PModel()
m.fit(temps, y, residual_X=res_X)
models["ASHRAE 5P"] = m.predict(temps, residual_X=res_X)

# --- Plot ---
sort_idx = np.argsort(temps)
temps_sorted = temps[sort_idx]

fig, ax = plt.subplots(figsize=(12, 7))

ax.scatter(temps, y, alpha=0.15, s=6, c="silver", label="Actual data", zorder=1)

colors = {"OLS": "#e74c3c", "Ridge": "#3498db", "LightGBM": "#2ecc71", "ASHRAE 5P": "#9b59b6"}

for name, y_pred in models.items():
    pred_sorted = y_pred[sort_idx]
    # Smooth with rolling mean for cleaner lines
    window = 30
    if len(pred_sorted) > window:
        smoothed = np.convolve(pred_sorted, np.ones(window) / window, mode="valid")
        t_smooth = temps_sorted[(window - 1) // 2 : (window - 1) // 2 + len(smoothed)]
    else:
        smoothed = pred_sorted
        t_smooth = temps_sorted
    ax.plot(t_smooth, smoothed, color=colors[name], linewidth=2, label=name, zorder=3)

ax.axvline(bp, color="black", linestyle=":", alpha=0.5, linewidth=1,
           label=f"Balance point ({bp:.0f}°F / {(bp-32)*5/9:.0f}°C)")

ax.set_xlabel("Daily Mean Temperature (°F)", fontsize=12)
ax.set_ylabel("Daily Electricity Consumption (kWh)", fontsize=12)
ax.set_title("Data Points vs Model Baselines — U317", fontsize=14)
ax.legend(fontsize=10, loc="upper left")
ax.grid(True, alpha=0.2)

fig.tight_layout()
fig.savefig(output_dir / "baselines_comparison.png", dpi=200)
plt.close(fig)
print(f"Saved to {output_dir / 'baselines_comparison.png'}")
