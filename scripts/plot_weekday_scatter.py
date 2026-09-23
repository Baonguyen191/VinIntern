import sys
sys.path.insert(0, ".")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.io.eweld_loader import load_electricity, load_weather, get_user_cluster, get_weather_station
from src.io.eweld_merger import merge_electricity_weather, aggregate_daily

output_dir = Path("results/eweld/degree_day")

cluster = get_user_cluster("U317")
station = get_weather_station(cluster)
elec_df = load_electricity("U317")
weather_df = load_weather(station)
merged = merge_electricity_weather(elec_df, weather_df)
daily = aggregate_daily(merged)

weekday = daily[daily["day_of_week"] < 5]
weekend = daily[daily["day_of_week"] >= 5]
holiday = daily[daily["is_holiday"] == 1]

fig, ax = plt.subplots(figsize=(12, 7))

ax.scatter(weekday["temp_f"], weekday["kWh"], alpha=0.25, s=10,
           c="#e74c3c", label=f"Weekday ({len(weekday)} days)", zorder=2)
ax.scatter(weekend["temp_f"], weekend["kWh"], alpha=0.25, s=10,
           c="#3498db", label=f"Weekend ({len(weekend)} days)", zorder=2)
ax.scatter(holiday["temp_f"], holiday["kWh"], alpha=0.7, s=25,
           c="#2ecc71", marker="D", edgecolors="black", linewidths=0.3,
           label=f"Holiday ({len(holiday)} days)", zorder=4)

ax.set_xlabel("Daily Mean Temperature (°F)", fontsize=12)
ax.set_ylabel("Daily Electricity Consumption (kWh)", fontsize=12)
ax.set_title("Temperature vs Consumption — Weekday / Weekend / Holiday — U317", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)

fig.tight_layout()
fig.savefig(output_dir / "scatter_weekday_weekend.png", dpi=200)
plt.close(fig)
print("Saved scatter_weekday_weekend.png")
