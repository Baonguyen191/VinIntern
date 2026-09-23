import json
import numpy as np
import pandas as pd
from pathlib import Path

from ..io.bdg2_loader import get_bdg2_site, get_bdg2_country, load_bdg2_metadata, load_bdg2_weather
from ..pipelines.bdg2 import _load_daily_for_variant
from ..models.statistical import compute_dow_baseline, compute_overall_baseline
from ..core.metrics import compute_all_metrics
from ..core.anomaly_detector import (
    detect_point_anomalies, detect_drift_cusum, classify_anomalies,
)
from ..core.cross_validation import generate_folds
from ..visualization.degree_day import (
    plot_actual_vs_predicted, plot_residuals, plot_cusum, plot_anomaly_calendar,
)
from ..visualization.statistical import plot_rolling_baseline, plot_dow_profiles, plot_scatter_actual_vs_pred

MODEL_NAMES = ["dow_median", "overall_median"]
VARIANT_NAMES = ["raw", "bdg2_cleaned", "our_cleaned"]

_MODEL_FNS = {
    "dow_median": compute_dow_baseline,
    "overall_median": compute_overall_baseline,
}


def _run_single_variant(
    daily: pd.DataFrame,
    variant: str,
    output_dir: Path,
    k: float,
    window: int,
) -> list[dict]:
    var_dir = output_dir / variant
    var_dir.mkdir(parents=True, exist_ok=True)

    folds = generate_folds(
        daily.index[0], daily.index[-1],
        initial_train_years=1, test_years=1,
    )
    print(f"  {len(folds)} fold(s)")

    all_fold_metrics: list[dict] = []

    for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(folds):
        fold_num = fold_idx + 1
        fold_dir = var_dir / f"fold_{fold_num}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_daily = daily[(daily.index >= train_start) & (daily.index < train_end)]
        test_daily = daily[(daily.index >= test_start) & (daily.index < test_end)]
        print(f"  Fold {fold_num}: Train {train_start.date()}->{train_end.date()} ({len(train_daily)}d), "
              f"Test {test_start.date()}->{test_end.date()} ({len(test_daily)}d)")

        if fold_num == 1:
            plot_dow_profiles(train_daily, var_dir)

        best_model_name = None
        best_cv_rmse = np.inf
        best_y_test = None
        best_y_pred = None
        fold_metrics: dict[str, dict] = {}

        for model_name in MODEL_NAMES:
            fn = _MODEL_FNS[model_name]
            y_test, y_pred, model_info = fn(train_daily, test_daily, window=window)
            metrics = compute_all_metrics(y_test, y_pred)
            fold_metrics[model_name] = metrics
            print(f"    {model_name}: CV(RMSE)={metrics['cv_rmse']:.2f}%, "
                  f"NMBE={metrics['nmbe']:.2f}%, R2={metrics['r_squared']:.4f}")

            weights = {
                **model_info,
                "fold": fold_num,
                "train_period": f"{train_start.date()} to {train_end.date()}",
                "test_period": f"{test_start.date()} to {test_end.date()}",
            }
            with open(fold_dir / f"fold_{fold_num}_{model_name}_weights.json", "w") as wf:
                json.dump(weights, wf, indent=2)

            if metrics["cv_rmse"] < best_cv_rmse:
                best_cv_rmse = metrics["cv_rmse"]
                best_model_name = model_name
                best_y_test = y_test
                best_y_pred = y_pred

            residuals = y_test - y_pred
            point_mask, threshold = detect_point_anomalies(residuals, k=k)
            s_plus, s_minus, drift_up, drift_down = detect_drift_cusum(residuals)

            plot_actual_vs_predicted(
                test_daily.index, y_test, y_pred, model_name, fold_dir, fold_num,
            )
            plot_residuals(
                test_daily.index, residuals, threshold, point_mask,
                model_name, fold_dir, fold_num,
            )
            h_sigma = 5.0 * np.std(residuals)
            plot_cusum(
                test_daily.index, s_plus, s_minus, h_sigma,
                model_name, fold_dir, fold_num,
            )
            plot_rolling_baseline(
                test_daily.index, y_test, y_pred, residuals, threshold,
                model_name, fold_dir, fold_num,
            )
            plot_scatter_actual_vs_pred(
                y_test, y_pred, test_daily["day_of_week"].values, metrics,
                model_name, fold_dir, fold_num,
            )

        with open(fold_dir / f"fold_{fold_num}_metrics.json", "w") as f:
            json.dump(fold_metrics, f, indent=2)

        all_fold_metrics.append(fold_metrics)

    return all_fold_metrics


def run_statistical_pipeline(
    building_id: str = "Rat_office_Colby",
    output_dir: str | Path = "results/bdg2/statistical",
    k: float = 3.0,
    window: int = 28,
) -> dict:
    output_dir = Path(output_dir) / building_id
    output_dir.mkdir(parents=True, exist_ok=True)

    site = get_bdg2_site(building_id)
    country = get_bdg2_country(building_id)
    meta = load_bdg2_metadata(building_id)

    print(f"Building: {building_id}")
    print(f"  Site: {site}, Country: {country}")
    print(f"  Type: {meta.get('primaryspaceusage', '?')}, sqm: {meta.get('sqm', '?')}")
    print(f"  Pipeline: statistical baseline (window={window})")

    print("\nLoading weather...")
    weather_df = load_bdg2_weather(site)

    comparison_rows: list[dict] = []

    for variant in VARIANT_NAMES:
        print(f"\n{'=' * 60}")
        print(f"Variant: {variant}")
        print(f"{'=' * 60}")

        daily = _load_daily_for_variant(building_id, variant, weather_df, country)
        print(f"  Daily data: {len(daily)} days, "
              f"{daily.index[0].date()} -> {daily.index[-1].date()}")
        print(f"  kWh range: {daily['kWh'].min():.0f} - {daily['kWh'].max():.0f}")

        fold_metrics_list = _run_single_variant(
            daily, variant, output_dir, k, window,
        )

        for fold_idx, fold_metrics in enumerate(fold_metrics_list):
            for model_name in MODEL_NAMES:
                m = fold_metrics.get(model_name, {})
                comparison_rows.append({
                    "variant": variant,
                    "fold": fold_idx + 1,
                    "model": model_name,
                    "cv_rmse": m.get("cv_rmse", np.nan),
                    "nmbe": m.get("nmbe", np.nan),
                    "r_squared": m.get("r_squared", np.nan),
                })

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(output_dir / "comparison.csv", index=False)

    print(f"\n{'=' * 60}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    for variant in VARIANT_NAMES:
        vdf = comp_df[comp_df["variant"] == variant]
        print(f"\n  {variant}:")
        for _, row in vdf.iterrows():
            print(f"    {row['model']:16s}: CV(RMSE)={row['cv_rmse']:.2f}%, "
                  f"NMBE={row['nmbe']:.2f}%, R2={row['r_squared']:.4f}")

    print(f"\nResults saved to {output_dir}")
    return comp_df.to_dict("records")
