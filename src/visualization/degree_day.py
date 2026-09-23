import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def plot_scatter_temp_kwh(
    daily: pd.DataFrame, balance_point: float, output_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(daily["temp_f"], daily["kWh"], alpha=0.3, s=8, c="steelblue")
    ax.axvline(balance_point, color="red", linestyle="--", label=f"Balance point = {balance_point:.1f}°F")
    ax.set_xlabel("Daily Mean Temperature (°F)")
    ax.set_ylabel("Daily kWh")
    ax.set_title("Temperature vs Electricity Consumption")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "scatter_temp_kwh.png", dpi=150)
    plt.close(fig)


def plot_actual_vs_predicted(
    dates: pd.DatetimeIndex,
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_dir: Path,
    fold_num: int,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, y_actual, alpha=0.7, linewidth=0.8, label="Actual")
    ax.plot(dates, y_pred, alpha=0.7, linewidth=0.8, label="Predicted")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily kWh")
    ax.set_title(f"Fold {fold_num} — {model_name}: Actual vs Predicted")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"fold_{fold_num}_{model_name}_actual_vs_pred.png", dpi=150)
    plt.close(fig)


def plot_residuals(
    dates: pd.DatetimeIndex,
    residuals: np.ndarray,
    threshold: float,
    anomaly_mask: np.ndarray,
    model_name: str,
    output_dir: Path,
    fold_num: int,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, residuals, alpha=0.6, linewidth=0.8, color="steelblue")
    ax.axhline(threshold, color="red", linestyle="--", alpha=0.7, label=f"+{threshold:.1f}")
    ax.axhline(-threshold, color="red", linestyle="--", alpha=0.7, label=f"-{threshold:.1f}")
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    if np.any(anomaly_mask):
        ax.scatter(
            dates[anomaly_mask], residuals[anomaly_mask],
            color="red", s=20, zorder=5, label="Anomaly",
        )
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual (kWh)")
    ax.set_title(f"Fold {fold_num} — {model_name}: Residuals")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"fold_{fold_num}_{model_name}_residuals.png", dpi=150)
    plt.close(fig)


def plot_cusum(
    dates: pd.DatetimeIndex,
    s_plus: np.ndarray,
    s_minus: np.ndarray,
    h_sigma: float,
    model_name: str,
    output_dir: Path,
    fold_num: int,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, s_plus, label="S+", color="orangered", linewidth=0.8)
    ax.plot(dates, s_minus, label="S−", color="royalblue", linewidth=0.8)
    ax.axhline(h_sigma, color="red", linestyle="--", alpha=0.7, label=f"h = {h_sigma:.1f}")
    ax.set_xlabel("Date")
    ax.set_ylabel("CUSUM Statistic")
    ax.set_title(f"Fold {fold_num} — {model_name}: CUSUM Drift Detection")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"fold_{fold_num}_{model_name}_cusum.png", dpi=150)
    plt.close(fig)


def plot_anomaly_calendar(
    anomaly_df: pd.DataFrame, output_dir: Path
) -> None:
    anom = anomaly_df[anomaly_df["is_anomaly"]].copy()
    if anom.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 4))
    colors = {
        "point": "red",
        "drift_up": "orange",
        "drift_down": "blue",
        "context": "green",
    }
    for _, row in anom.iterrows():
        types = row["anomaly_type"].split(",")
        c = colors.get(types[0], "gray")
        ax.axvline(row["date"], color=c, alpha=0.5, linewidth=0.5)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=c, label=k) for k, c in colors.items()]
    ax.legend(handles=handles, loc="upper right")
    ax.set_xlabel("Date")
    ax.set_title("Anomaly Timeline")
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_dir / "anomaly_calendar.png", dpi=150)
    plt.close(fig)


def plot_model_comparison(
    all_fold_metrics: dict[str, list[dict]], output_dir: Path
) -> None:
    model_names = list(all_fold_metrics.keys())
    metrics = ["cv_rmse", "nmbe", "r_squared"]
    labels = ["CV(RMSE) %", "NMBE %", "R²"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(model_names))
    width = 0.6

    for ax, metric, label in zip(axes, metrics, labels):
        means, stds = [], []
        for name in model_names:
            vals = [f[metric] for f in all_fold_metrics[name]]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        bars = ax.bar(x, means, width, yerr=stds, capsize=4, color="steelblue", alpha=0.8)
        if metric == "cv_rmse":
            ax.axhline(25, color="red", linestyle="--", alpha=0.5, label="ASHRAE 25%")
            ax.legend()
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=15)
        ax.set_ylabel(label)
        ax.set_title(label)

    fig.suptitle("Model Comparison (mean ± std across folds)")
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=150)
    plt.close(fig)


def plot_metric_trend(
    all_fold_metrics: dict[str, list[dict]], output_dir: Path
) -> None:
    metrics = ["cv_rmse", "nmbe", "r_squared"]
    labels = ["CV(RMSE) %", "NMBE %", "R²"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric, label in zip(axes, metrics, labels):
        for model_name, fold_list in all_fold_metrics.items():
            vals = [f[metric] for f in fold_list]
            folds = list(range(1, len(vals) + 1))
            ax.plot(folds, vals, marker="o", label=model_name)
        ax.set_xlabel("Fold")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend()
        ax.set_xticks(range(1, max(len(v) for v in all_fold_metrics.values()) + 1))

    fig.suptitle("Metric Trend Across Folds")
    fig.tight_layout()
    fig.savefig(output_dir / "metric_trend.png", dpi=150)
    plt.close(fig)
