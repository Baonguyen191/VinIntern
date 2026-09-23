import json
import numpy as np
import pandas as pd
from pathlib import Path

from ..io.bdg2_loader import (
    load_bdg2_electricity, load_bdg2_weather, load_bdg2_metadata,
    get_bdg2_site, get_bdg2_country,
)
from ..io.bdg2_merger import merge_bdg2_electricity_weather, aggregate_bdg2_daily
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
    plot_cusum, plot_anomaly_calendar, plot_model_comparison,
)

MODEL_NAMES = ["ols", "ridge", "lgbm", "5p"]
VARIANT_NAMES = ["raw", "bdg2_cleaned", "our_cleaned"]


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
        res_X_train = np.column_stack([dow_train, hol_train, hum_train])

        dow_test = pd.get_dummies(test_daily["day_of_week"], prefix="dow", dtype=float).iloc[:, :-1].values
        hol_test = test_daily["is_holiday"].values.reshape(-1, 1)
        hum_test = test_daily["humidity"].values.reshape(-1, 1)
        res_X_test = np.column_stack([dow_test, hol_test, hum_test])

        model = ChangePoint5PModel()

        def _fit(X_unused, y, mask):
            model.fit(temp_train[mask], y[mask], residual_X=res_X_train[mask])

        def _predict(X_unused):
            return model.predict(temp_train, residual_X=res_X_train)

        mask = iterative_clean(_fit, _predict, temp_train, y_train, eta=eta)
        model.fit(temp_train[mask], y_train[mask], residual_X=res_X_train[mask])
        y_pred = model.predict(temp_test, residual_X=res_X_test)
        return y_test, y_pred, model


def _load_daily_for_variant(
    building_id: str, variant: str, weather_df: pd.DataFrame, country: str,
) -> pd.DataFrame:
    if variant == "raw":
        elec = load_bdg2_electricity(building_id, variant="raw")
    elif variant == "bdg2_cleaned":
        elec = load_bdg2_electricity(building_id, variant="cleaned")
    elif variant == "our_cleaned":
        elec = load_bdg2_electricity(building_id, variant="raw")
    else:
        raise ValueError(f"Unknown variant: {variant}")

    merged = merge_bdg2_electricity_weather(elec, weather_df)
    daily = aggregate_bdg2_daily(merged, country=country)

    if variant == "our_cleaned":
        daily, report = clean_daily_consumption(daily, country=country)
        daily = daily.dropna(subset=["kWh"])
        n_cleaned = len(report)
        print(f"    Our cleaning removed {n_cleaned} days")

    return daily


def _run_single_variant(
    daily: pd.DataFrame,
    variant: str,
    output_dir: Path,
    k: float,
    eta: float,
    balance_step: float,
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

        bp = find_balance_point(train_daily, step=balance_step)
        print(f"    Balance point: {bp:.1f}°F ({(bp - 32) * 5 / 9:.1f}°C)")

        if fold_num == 1:
            plot_scatter_temp_kwh(train_daily, bp, var_dir)

        best_model_name = None
        best_cv_rmse = np.inf
        best_y_test = None
        best_y_pred = None
        fold_metrics: dict[str, dict] = {}

        for model_name in MODEL_NAMES:
            try:
                y_test, y_pred, model = _fit_predict_model(
                    model_name, train_daily, test_daily, bp, eta
                )
                metrics = compute_all_metrics(y_test, y_pred)
                fold_metrics[model_name] = metrics
                print(f"    {model_name}: CV(RMSE)={metrics['cv_rmse']:.2f}%, "
                      f"NMBE={metrics['nmbe']:.2f}%, R²={metrics['r_squared']:.4f}")

                if metrics["cv_rmse"] < best_cv_rmse:
                    best_cv_rmse = metrics["cv_rmse"]
                    best_model_name = model_name
                    best_y_test = y_test
                    best_y_pred = y_pred
            except Exception as e:
                print(f"    {model_name}: FAILED — {e}")
                fold_metrics[model_name] = {
                    "cv_rmse": np.nan, "nmbe": np.nan, "r_squared": np.nan
                }

        with open(fold_dir / f"fold_{fold_num}_metrics.json", "w") as f:
            json.dump(fold_metrics, f, indent=2)

        if best_model_name is not None:
            residuals = best_y_test - best_y_pred
            point_mask, threshold = detect_point_anomalies(residuals, k=k)
            s_plus, s_minus, drift_up, drift_down = detect_drift_cusum(residuals)
            context_mask = detect_context_anomalies(test_daily, residuals, best_y_pred)

            plot_actual_vs_predicted(
                test_daily.index, best_y_test, best_y_pred,
                best_model_name, fold_dir, fold_num,
            )
            plot_residuals(
                test_daily.index, residuals, threshold, point_mask,
                best_model_name, fold_dir, fold_num,
            )
            h_sigma = 5.0 * np.std(residuals)
            plot_cusum(
                test_daily.index, s_plus, s_minus, h_sigma,
                best_model_name, fold_dir, fold_num,
            )

        all_fold_metrics.append(fold_metrics)

    return all_fold_metrics


def run_bdg2_pipeline(
    building_id: str = "Rat_office_Colby",
    output_dir: str | Path = "results/bdg2/degree_day",
    k: float = 3.0,
    eta: float = 3.0,
    balance_step: float = 0.5,
) -> dict:
    output_dir = Path(output_dir) / building_id
    output_dir.mkdir(parents=True, exist_ok=True)

    site = get_bdg2_site(building_id)
    country = get_bdg2_country(building_id)
    meta = load_bdg2_metadata(building_id)

    print(f"Building: {building_id}")
    print(f"  Site: {site}, Country: {country}")
    print(f"  Type: {meta.get('primaryspaceusage', '?')}, sqm: {meta.get('sqm', '?')}")

    print("\nLoading weather...")
    weather_df = load_bdg2_weather(site)
    print(f"  Weather: {len(weather_df)} hourly records")

    comparison_rows: list[dict] = []

    for variant in VARIANT_NAMES:
        print(f"\n{'=' * 60}")
        print(f"Variant: {variant}")
        print(f"{'=' * 60}")

        daily = _load_daily_for_variant(building_id, variant, weather_df, country)
        print(f"  Daily data: {len(daily)} days, "
              f"{daily.index[0].date()} -> {daily.index[-1].date()}")
        print(f"  kWh range: {daily['kWh'].min():.0f} – {daily['kWh'].max():.0f}")

        daily["is_storm"] = False

        fold_metrics_list = _run_single_variant(
            daily, variant, output_dir, k, eta, balance_step,
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
            print(f"    {row['model']:6s}: CV(RMSE)={row['cv_rmse']:.2f}%, "
                  f"NMBE={row['nmbe']:.2f}%, R²={row['r_squared']:.4f}")

    print(f"\nResults saved to {output_dir}")
    return comp_df.to_dict("records")
