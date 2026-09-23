import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from ..features.tier_b import SHAPE_STATS


def plot_tier_a_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: pd.DatetimeIndex,
    fold_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, y_true, color="steelblue", alpha=0.4, linewidth=0.3, label="Actual")
    ax.plot(
        dates, y_pred, color="darkorange", alpha=0.6, linewidth=0.3, label="Predicted"
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("kWh")
    ax.set_title("Tier A — Actual vs Predicted (15-min)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fold_dir / "tier_a_actual_vs_predicted.png", dpi=150)
    plt.close(fig)


def plot_tier_a_residuals(
    residuals: np.ndarray,
    dates: pd.DatetimeIndex,
    eta: float,
    fold_dir: Path,
) -> None:
    mu = np.mean(residuals)
    sigma = np.std(residuals)
    ucl = mu + eta * sigma
    lcl = mu - eta * sigma

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.scatter(dates, residuals, s=0.3, alpha=0.3, color="steelblue")
    ax.axhline(ucl, color="red", linestyle="--", label=f"UCL ({ucl:.2f})")
    ax.axhline(lcl, color="red", linestyle="--", label=f"LCL ({lcl:.2f})")
    ax.axhline(mu, color="gray", linestyle="-", alpha=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Residual (kWh)")
    ax.set_title("Tier A — Residuals with Control Limits")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fold_dir / "tier_a_residual_plot.png", dpi=150)
    plt.close(fig)


def plot_tier_a_histogram(residuals: np.ndarray, fold_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=100, density=True, alpha=0.7, color="steelblue")
    ax.set_xlabel("Residual (kWh)")
    ax.set_ylabel("Density")
    ax.set_title("Tier A — Residual Distribution")
    fig.tight_layout()
    fig.savefig(fold_dir / "tier_a_residual_histogram.png", dpi=150)
    plt.close(fig)


def plot_tier_b_stat(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: pd.DatetimeIndex,
    stat_name: str,
    eta: float,
    fold_dir: Path,
) -> None:
    residuals = y_true - y_pred
    mu = np.mean(residuals)
    sigma = np.std(residuals)
    ucl = mu + eta * sigma
    lcl = mu - eta * sigma

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(dates, y_true, s=5, alpha=0.5, color="steelblue", label="Actual")
    axes[0].scatter(
        dates, y_pred, s=5, alpha=0.5, color="darkorange", label="Predicted"
    )
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel(stat_name)
    axes[0].set_title(f"Tier B — {stat_name}: Actual vs Predicted")
    axes[0].legend()

    axes[1].scatter(dates, residuals, s=5, alpha=0.5, color="steelblue")
    axes[1].axhline(ucl, color="red", linestyle="--", label=f"UCL ({ucl:.2f})")
    axes[1].axhline(lcl, color="red", linestyle="--", label=f"LCL ({lcl:.2f})")
    axes[1].axhline(mu, color="gray", linestyle="-", alpha=0.5)
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Residual")
    axes[1].set_title(f"Tier B — {stat_name}: Residuals")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(fold_dir / f"tier_b_{stat_name}.png", dpi=150)
    plt.close(fig)


def plot_metric_trend(all_fold_metrics: list[dict], output_dir: Path) -> None:
    folds = list(range(1, len(all_fold_metrics) + 1))

    tier_a_metrics = ["cv_rmse", "nmbe", "r_squared"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, metric in zip(axes, tier_a_metrics):
        vals = [fm[f"tier_a_{metric}"] for fm in all_fold_metrics]
        ax.plot(folds, vals, marker="o", color="steelblue", label="Tier A")
        ax.set_xlabel("Fold")
        ax.set_ylabel(metric)
        ax.set_title(f"Tier A — {metric}")
        ax.set_xticks(folds)

    fig.tight_layout()
    fig.savefig(output_dir / "metric_trend_tier_a.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for idx, stat in enumerate(SHAPE_STATS):
        ax = axes[idx // 3, idx % 3]
        for metric in tier_a_metrics:
            key = f"tier_b_{stat}_{metric}"
            vals = [fm[key] for fm in all_fold_metrics]
            ax.plot(folds, vals, marker="o", label=metric)
        ax.set_xlabel("Fold")
        ax.set_title(f"Tier B — {stat}")
        ax.legend(fontsize=7)
        ax.set_xticks(folds)

    fig.tight_layout()
    fig.savefig(output_dir / "metric_trend_tier_b.png", dpi=150)
    plt.close(fig)
