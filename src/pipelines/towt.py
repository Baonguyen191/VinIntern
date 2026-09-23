import json
import numpy as np
import pandas as pd
from pathlib import Path

from ..io.eweld_loader import load_electricity, load_weather, get_user_cluster, get_weather_station
from ..io.eweld_merger import merge_electricity_weather
from ..core.metrics import compute_all_metrics
from ..features.tier_a import build_tier_a_features
from ..features.tier_b import (
    SHAPE_STATS,
    compute_daily_shape_stats,
    compute_daily_weather,
    build_tier_b_features,
)
from ..core.phase1_cleaning import phase1_clean_tier_a, phase1_clean_tier_b
from ..core.cross_validation import generate_folds
from ..visualization.towt import (
    plot_tier_a_actual_vs_predicted,
    plot_tier_a_residuals,
    plot_tier_a_histogram,
    plot_tier_b_stat,
    plot_metric_trend,
)


def run_pipeline(
    user_id: str = "U317",
    output_dir: str | Path = "results/eweld/towt",
    eta: float = 3.0,
    n_breakpoints: int = 6,
    alpha: float = 1.0,
) -> list[dict]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load & merge ---
    print(f"Loading data for {user_id}...")
    cluster = get_user_cluster(user_id)
    station = get_weather_station(cluster)
    elec_df = load_electricity(user_id)
    weather_df = load_weather(station)
    merged = merge_electricity_weather(elec_df, weather_df)
    print(f"  Merged: {len(merged)} rows, {merged.index[0]} -> {merged.index[-1]}")

    # --- Generate folds ---
    folds = generate_folds(merged.index[0], merged.index[-1])
    print(f"  {len(folds)} folds generated")

    all_fold_metrics = []

    for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(folds):
        fold_num = fold_idx + 1
        fold_dir = output_dir / f"fold_{fold_num}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Fold {fold_num} ===")
        print(f"  Train: {train_start.date()} -> {train_end.date()}")
        print(f"  Test:  {test_start.date()} -> {test_end.date()}")

        train = merged[(merged.index >= train_start) & (merged.index < train_end)]
        test = merged[(merged.index >= test_start) & (merged.index < test_end)]
        print(f"  Train rows: {len(train)}, Test rows: {len(test)}")

        fold_metrics = {}

        # ========== Tier A (TOWT) ==========
        print("  Tier A: building features...")
        X_train_a, bp_a = build_tier_a_features(
            train, n_breakpoints=n_breakpoints
        )
        X_test_a, _ = build_tier_a_features(test, breakpoints=bp_a)
        y_train_a = train["Value"].values
        y_test_a = test["Value"].values

        print("  Tier A: Phase I cleaning...")
        model_a, mask_a = phase1_clean_tier_a(
            X_train_a, y_train_a, alpha=alpha, eta=eta
        )
        n_removed_a = int(np.sum(~mask_a))
        print(f"  Tier A: removed {n_removed_a} outliers ({n_removed_a/len(mask_a)*100:.1f}%)")

        print("  Tier A: predicting test set...")
        y_pred_a = model_a.predict(X_test_a)
        metrics_a = compute_all_metrics(y_test_a, y_pred_a)
        for k, v in metrics_a.items():
            fold_metrics[f"tier_a_{k}"] = v
        print(f"  Tier A: CV(RMSE)={metrics_a['cv_rmse']:.2f}%, NMBE={metrics_a['nmbe']:.2f}%, R²={metrics_a['r_squared']:.4f}")

        print("  Tier A: generating plots...")
        residuals_a = y_test_a - y_pred_a
        plot_tier_a_actual_vs_predicted(y_test_a, y_pred_a, test.index, fold_dir)
        plot_tier_a_residuals(residuals_a, test.index, eta, fold_dir)
        plot_tier_a_histogram(residuals_a, fold_dir)

        # ========== Tier B (Zhu shape stats) ==========
        print("  Tier B: computing daily shape stats...")
        train_daily_stats = compute_daily_shape_stats(train)
        test_daily_stats = compute_daily_shape_stats(test)
        train_daily_weather = compute_daily_weather(train)
        test_daily_weather = compute_daily_weather(test)

        X_train_b, y_train_b, bp_b, train_dates_b = build_tier_b_features(
            train_daily_stats, train_daily_weather, n_breakpoints=n_breakpoints
        )
        X_test_b, y_test_b, _, test_dates_b = build_tier_b_features(
            test_daily_stats, test_daily_weather, breakpoints=bp_b
        )

        print("  Tier B: Phase I cleaning...")
        model_b, masks_b = phase1_clean_tier_b(
            X_train_b, y_train_b, alpha=alpha, eta=eta
        )

        print("  Tier B: predicting test set...")
        y_pred_b = model_b.predict(X_test_b)

        for stat in SHAPE_STATS:
            metrics_b = compute_all_metrics(y_test_b[stat], y_pred_b[stat])
            for k, v in metrics_b.items():
                fold_metrics[f"tier_b_{stat}_{k}"] = v
            print(f"    {stat}: CV(RMSE)={metrics_b['cv_rmse']:.2f}%, R²={metrics_b['r_squared']:.4f}")

        print("  Tier B: generating plots...")
        for stat in SHAPE_STATS:
            plot_tier_b_stat(
                y_test_b[stat], y_pred_b[stat], test_dates_b, stat, eta, fold_dir
            )

        # Save fold metrics
        with open(fold_dir / f"fold_{fold_num}_metrics.json", "w") as f:
            json.dump(fold_metrics, f, indent=2)

        all_fold_metrics.append(fold_metrics)

    # ========== Summary ==========
    print("\n=== Summary ===")
    _save_summary(all_fold_metrics, output_dir)
    plot_metric_trend(all_fold_metrics, output_dir)
    print(f"Results saved to {output_dir}")

    return all_fold_metrics


def _save_summary(all_fold_metrics: list[dict], output_dir: Path) -> None:
    df = pd.DataFrame(all_fold_metrics)
    df.index = [f"fold_{i+1}" for i in range(len(df))]

    mean_row = df.mean()
    std_row = df.std()
    summary = pd.concat(
        [df, mean_row.to_frame("mean").T, std_row.to_frame("std").T]
    )
    summary.to_csv(output_dir / "summary_table.csv")

    # ASHRAE pass/fail
    ashrae = []
    for fold_name, row in df.iterrows():
        entry = {"fold": fold_name}
        entry["tier_a_cv_rmse_pass"] = row["tier_a_cv_rmse"] <= 25.0
        entry["tier_a_nmbe_pass"] = abs(row["tier_a_nmbe"]) <= 10.0
        for stat in SHAPE_STATS:
            entry[f"tier_b_{stat}_cv_rmse_pass"] = (
                row[f"tier_b_{stat}_cv_rmse"] <= 25.0
            )
            entry[f"tier_b_{stat}_nmbe_pass"] = (
                abs(row[f"tier_b_{stat}_nmbe"]) <= 10.0
            )
        ashrae.append(entry)

    pd.DataFrame(ashrae).set_index("fold").to_csv(
        output_dir / "ashrae_pass_fail.csv"
    )

    # Print summary
    print("\nTier A (mean ± std across folds):")
    for metric in ("cv_rmse", "nmbe", "r_squared"):
        key = f"tier_a_{metric}"
        print(f"  {metric}: {mean_row[key]:.2f} ± {std_row[key]:.2f}")

    print("\nTier B (mean ± std across folds):")
    for stat in SHAPE_STATS:
        vals = [f"{mean_row[f'tier_b_{stat}_{m}']:.2f}±{std_row[f'tier_b_{stat}_{m}']:.2f}"
                for m in ("cv_rmse", "nmbe", "r_squared")]
        print(f"  {stat}: CV(RMSE)={vals[0]}, NMBE={vals[1]}, R²={vals[2]}")
