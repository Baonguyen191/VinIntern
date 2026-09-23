import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def plot_rolling_baseline(
    dates: pd.DatetimeIndex,
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    residuals: np.ndarray,
    threshold: float,
    model_name: str,
    output_dir: Path,
    fold_num: int,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})

    ax1 = axes[0]
    ax1.plot(dates, y_actual, alpha=0.6, linewidth=0.8, label="Actual", color="steelblue")
    ax1.plot(dates, y_pred, alpha=0.8, linewidth=1.2, label="Baseline", color="red")
    upper = y_pred + threshold
    lower = y_pred - threshold
    ax1.fill_between(dates, lower, upper, alpha=0.15, color="red", label=f"+/-{threshold:.0f} kWh")
    ax1.set_ylabel("Daily kWh")
    ax1.set_title(f"Fold {fold_num} -- {model_name}: Actual vs Statistical Baseline")
    ax1.legend()

    ax2 = axes[1]
    colors = np.where(np.abs(residuals) > threshold, "red", "steelblue")
    ax2.bar(dates, residuals, color=colors, alpha=0.6, width=1.0)
    ax2.axhline(threshold, color="red", linestyle="--", alpha=0.5)
    ax2.axhline(-threshold, color="red", linestyle="--", alpha=0.5)
    ax2.axhline(0, color="gray", alpha=0.3)
    ax2.set_ylabel("Residual (kWh)")
    ax2.set_xlabel("Date")

    fig.tight_layout()
    fig.savefig(output_dir / f"fold_{fold_num}_{model_name}_baseline.png", dpi=150)
    plt.close(fig)


def plot_scatter_actual_vs_pred(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    day_of_week: np.ndarray,
    metrics: dict,
    model_name: str,
    output_dir: Path,
    fold_num: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))

    weekday_mask = day_of_week < 5
    weekend_mask = ~weekday_mask

    ax.scatter(y_actual[weekday_mask], y_pred[weekday_mask],
               alpha=0.4, s=15, c="steelblue", label="Weekday", edgecolors="none")
    ax.scatter(y_actual[weekend_mask], y_pred[weekend_mask],
               alpha=0.4, s=15, c="coral", label="Weekend", edgecolors="none")

    all_vals = np.concatenate([y_actual, y_pred])
    lo, hi = np.min(all_vals) * 0.95, np.max(all_vals) * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, linewidth=1, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    cv = metrics.get("cv_rmse", 0)
    r2 = metrics.get("r_squared", 0)
    nmbe = metrics.get("nmbe", 0)
    ax.text(0.05, 0.95,
            f"CV(RMSE) = {cv:.2f}%\nNMBE = {nmbe:.2f}%\nR2 = {r2:.4f}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Actual (kWh)")
    ax.set_ylabel("Predicted (kWh)")
    ax.set_title(f"Fold {fold_num} -- {model_name}: Actual vs Predicted")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / f"fold_{fold_num}_{model_name}_scatter.png", dpi=150)
    plt.close(fig)


def plot_dow_profiles(
    train_daily: pd.DataFrame, output_dir: Path
) -> None:
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    groups = [train_daily[train_daily["day_of_week"] == d]["kWh"].values for d in range(7)]

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(groups, tick_labels=dow_labels, patch_artist=True)
    weekday_color = "steelblue"
    weekend_color = "coral"
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(weekend_color if i >= 5 else weekday_color)
        patch.set_alpha(0.6)

    medians = [np.median(g) for g in groups if len(g) > 0]
    overall_median = np.median(train_daily["kWh"].values)
    ax.axhline(overall_median, color="gray", linestyle="--", alpha=0.5,
               label=f"Overall median = {overall_median:.0f}")

    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Daily kWh")
    ax.set_title("Consumption Profile by Day of Week (Training)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "dow_profiles.png", dpi=150)
    plt.close(fig)
