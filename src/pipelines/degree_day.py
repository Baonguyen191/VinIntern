import json
import numpy as np
import pandas as pd
from pathlib import Path

from ..io.eweld_loader import load_electricity, load_weather, load_extreme_weather, get_user_cluster, get_weather_station
from ..io.eweld_merger import merge_electricity_weather, aggregate_daily, build_ew_daily_flags
from ..core.data_cleaning import clean_daily_consumption
from ..core.metrics import compute_all_metrics
from ..features.degree_day import find_balance_point, build_dd_features, build_raw_features
from ..models.baseline import (
    OLSModel, RidgeModel, LGBMModel, ChangePoint5PModel, iterative_clean,
)
from ..core.anomaly_detector import (
    detect_point_anomalies, detect_drift_cusum,
    detect_context_anomalies, classify_anomalies,
)
from ..core.cross_validation import generate_folds
from ..visualization.degree_day import (
    plot_scatter_temp_kwh, plot_actual_vs_predicted, plot_residuals,
    plot_cusum, plot_anomaly_calendar, plot_model_comparison, plot_metric_trend,
)

MODEL_NAMES = ["ols", "ridge", "lgbm", "5p"]


def _fit_predict_model(model_name, train_daily, test_daily, balance_point, eta):
    if model_name in ("ols", "ridge"):
        X_train, feat_names = build_dd_features(train_daily, balance_point)
        X_test, _ = build_dd_features(test_daily, balance_point)
        y_train = train_daily["kWh"].values
        y_test = test_daily["kWh"].values

        model = OLSModel() if model_name == "ols" else RidgeModel()

        def _fit(X, y, mask):
            model.fit(X[mask], y[mask])

        def _predict(X):
            return model.predict(X)

        mask = iterative_clean(_fit, _predict, X_train, y_train, eta=eta)
        model.fit(X_train[mask], y_train[mask])
        y_pred = model.predict(X_test)
        return y_test, y_pred, model

    elif model_name == "lgbm":
        X_train, feat_names, cat_idx = build_raw_features(train_daily)
        X_test, _, _ = build_raw_features(test_daily)
        y_train = train_daily["kWh"].values
        y_test = test_daily["kWh"].values

        model = LGBMModel()

        def _fit(X, y, mask):
            model.fit(X[mask], y[mask], categorical_indices=cat_idx)

        def _predict(X):
            return model.predict(X)

        mask = iterative_clean(_fit, _predict, X_train, y_train, eta=eta)
        model.fit(X_train[mask], y_train[mask], categorical_indices=cat_idx)
        y_pred = model.predict(X_test)
        return y_test, y_pred, model

    elif model_name == "5p":
        temp_train = train_daily["temp_f"].values
        temp_test = test_daily["temp_f"].values
        y_train = train_daily["kWh"].values
        y_test = test_daily["kWh"].values

        dow_train = pd.get_dummies(train_daily["day_of_week"], prefix="dow", dtype=float).iloc[:, :-1].values
        hol_train = train_daily["is_holiday"].values.reshape(-1, 1)
        hum_train = train_daily["humidity"].values.reshape(-1, 1)

        ref_date = train_daily.index[0]
        m_tr = train_daily.index.month.values.astype(float)
        month_sin_tr = np.sin(2 * np.pi * m_tr / 12).reshape(-1, 1)
        month_cos_tr = np.cos(2 * np.pi * m_tr / 12).reshape(-1, 1)
        train_days = (train_daily.index - ref_date).days.values.astype(float)
        trend_tr = (train_days / max(train_days.max(), 1)).reshape(-1, 1)

        res_X_train = np.column_stack([dow_train, hol_train, hum_train,
                                       month_sin_tr, month_cos_tr, trend_tr])

        dow_test = pd.get_dummies(test_daily["day_of_week"], prefix="dow", dtype=float).iloc[:, :-1].values
        hol_test = test_daily["is_holiday"].values.reshape(-1, 1)
        hum_test = test_daily["humidity"].values.reshape(-1, 1)

        m_te = test_daily.index.month.values.astype(float)
        month_sin_te = np.sin(2 * np.pi * m_te / 12).reshape(-1, 1)
        month_cos_te = np.cos(2 * np.pi * m_te / 12).reshape(-1, 1)
        test_days = (test_daily.index - ref_date).days.values.astype(float)
        trend_te = (test_days / max(train_days.max(), 1)).reshape(-1, 1)

        res_X_test = np.column_stack([dow_test, hol_test, hum_test,
                                      month_sin_te, month_cos_te, trend_te])

        model = ChangePoint5PModel()

        def _fit(X_unused, y, mask):
            model.fit(temp_train[mask], y[mask], residual_X=res_X_train[mask])

        def _predict(X_unused):
            return model.predict(temp_train, residual_X=res_X_train)

        mask = iterative_clean(_fit, _predict, temp_train, y_train, eta=eta)
        model.fit(temp_train[mask], y_train[mask], residual_X=res_X_train[mask])
        y_pred = model.predict(temp_test, residual_X=res_X_test)
        return y_test, y_pred, model


def run_pipeline(
    user_id: str = "U317",
    output_dir: str | Path = "results/eweld/degree_day",
    k: float = 3.0,
    eta: float = 3.0,
    balance_step: float = 0.5,
) -> dict[str, list[dict]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data for {user_id}...")
    cluster = get_user_cluster(user_id)
    station = get_weather_station(cluster)
    elec_df = load_electricity(user_id)
    weather_df = load_weather(station)
    merged = merge_electricity_weather(elec_df, weather_df)
    daily_raw = aggregate_daily(merged)

    print(f"  Raw daily: {len(daily_raw)} days")
    daily_cleaned, outlier_report = clean_daily_consumption(
        daily_raw, country="TW",
    )
    n_outliers = daily_cleaned["kWh"].isna().sum()
    daily = daily_cleaned.dropna(subset=["kWh"])
    print(f"  Cleaning removed {n_outliers} days ({len(outlier_report)} outlier records)")
    print(f"  Cleaned daily: {len(daily)} days, {daily.index[0].date()} -> {daily.index[-1].date()}")

    outlier_report.to_csv(output_dir / "outlier_report.csv", index=False)

    ew_df = load_extreme_weather(cluster)
    daily["is_storm"] = build_ew_daily_flags(ew_df, daily.index)
    n_storm = int(daily["is_storm"].sum())
    print(f"  Storm/typhoon days excluded from training: {n_storm}")

    folds = generate_folds(daily.index[0], daily.index[-1])
    print(f"  {len(folds)} folds generated")

    all_metrics: dict[str, list[dict]] = {name: [] for name in MODEL_NAMES}

    for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(folds):
        fold_num = fold_idx + 1
        fold_dir = output_dir / f"fold_{fold_num}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Fold {fold_num} ===")
        print(f"  Train: {train_start.date()} -> {train_end.date()}")
        print(f"  Test:  {test_start.date()} -> {test_end.date()}")

        train_all = daily[(daily.index >= train_start) & (daily.index < train_end)]
        train_daily = train_all[~train_all["is_storm"]]
        test_daily = daily[(daily.index >= test_start) & (daily.index < test_end)]
        n_filtered = len(train_all) - len(train_daily)
        print(f"  Train days: {len(train_daily)} ({n_filtered} storm days filtered), Test days: {len(test_daily)}")

        bp = find_balance_point(train_daily, step=balance_step)
        print(f"  Balance point: {bp:.1f}°F ({(bp - 32) * 5/9:.1f}°C)")

        if fold_num == 1:
            plot_scatter_temp_kwh(train_daily, bp, output_dir)

        best_model_name = None
        best_cv_rmse = np.inf
        best_y_test = None
        best_y_pred = None

        fold_metrics_all = {}

        for model_name in MODEL_NAMES:
            try:
                y_test, y_pred, model = _fit_predict_model(
                    model_name, train_daily, test_daily, bp, eta
                )
                metrics = compute_all_metrics(y_test, y_pred)
                all_metrics[model_name].append(metrics)
                fold_metrics_all[model_name] = metrics
                print(f"  {model_name}: CV(RMSE)={metrics['cv_rmse']:.2f}%, NMBE={metrics['nmbe']:.2f}%, R²={metrics['r_squared']:.4f}")

                if metrics["cv_rmse"] < best_cv_rmse:
                    best_cv_rmse = metrics["cv_rmse"]
                    best_model_name = model_name
                    best_y_test = y_test
                    best_y_pred = y_pred
            except Exception as e:
                print(f"  {model_name}: FAILED — {e}")
                all_metrics[model_name].append({"cv_rmse": np.nan, "nmbe": np.nan, "r_squared": np.nan})

        with open(fold_dir / f"fold_{fold_num}_metrics.json", "w") as f:
            json.dump(fold_metrics_all, f, indent=2)

        if best_model_name is not None:
            residuals = best_y_test - best_y_pred
            point_mask, threshold = detect_point_anomalies(residuals, k=k)
            s_plus, s_minus, drift_up, drift_down = detect_drift_cusum(residuals)
            context_mask = detect_context_anomalies(test_daily, residuals, best_y_pred)

            plot_actual_vs_predicted(
                test_daily.index, best_y_test, best_y_pred, best_model_name, fold_dir, fold_num
            )
            plot_residuals(
                test_daily.index, residuals, threshold, point_mask, best_model_name, fold_dir, fold_num
            )
            h_sigma = 5.0 * np.std(residuals)
            plot_cusum(
                test_daily.index, s_plus, s_minus, h_sigma, best_model_name, fold_dir, fold_num
            )

            n_point = int(np.sum(point_mask))
            n_drift = int(np.sum(drift_up | drift_down))
            n_context = int(np.sum(context_mask))
            print(f"  Best model: {best_model_name} | Anomalies: {n_point} point, {n_drift} drift, {n_context} context")

    print("\n=== Summary ===")
    _save_summary(all_metrics, output_dir)
    plot_model_comparison(all_metrics, output_dir)
    plot_metric_trend(all_metrics, output_dir)

    print("\n=== Full-dataset anomaly detection ===")
    best_overall = _select_best_model(all_metrics)
    print(f"  Best overall model: {best_overall}")
    _run_full_anomaly_detection(daily, best_overall, bp, eta, k, output_dir)

    print(f"\nResults saved to {output_dir}")
    return all_metrics


def _select_best_model(all_metrics: dict[str, list[dict]]) -> str:
    best_name, best_mean = MODEL_NAMES[0], np.inf
    for name in MODEL_NAMES:
        vals = [m["cv_rmse"] for m in all_metrics[name] if not np.isnan(m["cv_rmse"])]
        if vals:
            mean_cv = np.mean(vals)
            if mean_cv < best_mean:
                best_mean = mean_cv
                best_name = name
    return best_name


def _run_full_anomaly_detection(daily, model_name, balance_point, eta, k, output_dir):
    train_clean = daily[~daily["is_storm"]]
    y_test, y_pred, model = _fit_predict_model(
        model_name, train_clean, daily, balance_point, eta
    )
    residuals = y_test - y_pred

    point_mask, threshold = detect_point_anomalies(residuals, k=k)
    s_plus, s_minus, drift_up, drift_down = detect_drift_cusum(residuals)
    context_mask = detect_context_anomalies(daily, residuals, y_pred)

    anomaly_df = classify_anomalies(
        daily.index, residuals, point_mask, drift_up, drift_down, context_mask
    )
    anomaly_df.to_csv(output_dir / "anomaly_report.csv", index=False)
    plot_anomaly_calendar(anomaly_df, output_dir)

    n_anom = int(anomaly_df["is_anomaly"].sum())
    print(f"  Total anomalous days: {n_anom}/{len(daily)} ({n_anom/len(daily)*100:.1f}%)")


def _save_summary(all_metrics: dict[str, list[dict]], output_dir: Path) -> None:
    rows = []
    for model_name in MODEL_NAMES:
        for fold_idx, metrics in enumerate(all_metrics[model_name]):
            row = {"model": model_name, "fold": fold_idx + 1}
            row.update(metrics)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "summary_table.csv", index=False)

    ashrae_rows = []
    for model_name in MODEL_NAMES:
        for fold_idx, metrics in enumerate(all_metrics[model_name]):
            ashrae_rows.append({
                "model": model_name,
                "fold": fold_idx + 1,
                "cv_rmse_pass": metrics["cv_rmse"] <= 25.0 if not np.isnan(metrics["cv_rmse"]) else False,
                "nmbe_pass": abs(metrics["nmbe"]) <= 10.0 if not np.isnan(metrics["nmbe"]) else False,
            })
    pd.DataFrame(ashrae_rows).to_csv(output_dir / "ashrae_pass_fail.csv", index=False)

    print("\nModel Comparison (mean ± std across folds):")
    for model_name in MODEL_NAMES:
        vals = all_metrics[model_name]
        if not vals:
            continue
        cv = [v["cv_rmse"] for v in vals if not np.isnan(v["cv_rmse"])]
        nb = [v["nmbe"] for v in vals if not np.isnan(v["nmbe"])]
        r2 = [v["r_squared"] for v in vals if not np.isnan(v["r_squared"])]
        if cv:
            print(
                f"  {model_name:6s}: CV(RMSE)={np.mean(cv):.2f}±{np.std(cv):.2f}%, "
                f"NMBE={np.mean(nb):.2f}±{np.std(nb):.2f}%, "
                f"R²={np.mean(r2):.4f}±{np.std(r2):.4f}"
            )
