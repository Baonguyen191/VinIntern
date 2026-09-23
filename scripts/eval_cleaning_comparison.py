"""Compare cleaning-only vs cleaning+seasonal+trend using CLEANING.md fold structure.

Fold 1: Train 2017, Test 2018
Fold 2: Train 2017-2018, Test 2019
"""
import sys
sys.path.insert(0, ".")

import json
import numpy as np
import pandas as pd
from pathlib import Path

from src.io.eweld_loader import load_electricity, load_weather, get_user_cluster, get_weather_station
from src.io.eweld_merger import merge_electricity_weather, aggregate_daily
from src.core.data_cleaning import clean_daily_consumption
from src.features.degree_day import find_balance_point, build_dd_features, build_raw_features
from src.models.baseline import (
    OLSModel, RidgeModel, LGBMModel, ChangePoint5PModel, iterative_clean,
)
from src.core.metrics import compute_all_metrics

MODEL_NAMES = ["ols", "ridge", "lgbm", "5p"]

FOLDS = [
    ("2017-01-01", "2018-01-01", "2018-01-01", "2019-01-01"),
    ("2017-01-01", "2019-01-01", "2019-01-01", "2020-01-01"),
]


def _fit_predict(model_name, train_daily, test_daily, balance_point, eta):
    if model_name in ("ols", "ridge"):
        X_train, _ = build_dd_features(train_daily, balance_point)
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
        return y_test, model.predict(X_test)

    elif model_name == "lgbm":
        X_train, _, cat_idx = build_raw_features(train_daily)
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
        return y_test, model.predict(X_test)

    else:  # 5p
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

        mask = iterative_clean(_fit, _predict, temp_train, y_train, eta=3.0)
        model.fit(temp_train[mask], y_train[mask], residual_X=res_X_train[mask])
        return y_test, model.predict(temp_test, residual_X=res_X_test)


def main():
    print("Loading & cleaning U317...")
    cluster = get_user_cluster("U317")
    station = get_weather_station(cluster)
    elec_df = load_electricity("U317")
    weather_df = load_weather(station)
    merged = merge_electricity_weather(elec_df, weather_df)
    daily_raw = aggregate_daily(merged)
    daily_cleaned, report = clean_daily_consumption(daily_raw, country="TW")
    daily = daily_cleaned.dropna(subset=["kWh"])
    print(f"  Cleaned: {len(daily)} days, removed {len(report)}")

    for fold_idx, (ts, te, vs, ve) in enumerate(FOLDS):
        fold_num = fold_idx + 1
        train_start, train_end = pd.Timestamp(ts), pd.Timestamp(te)
        test_start, test_end = pd.Timestamp(vs), pd.Timestamp(ve)

        train_daily = daily[(daily.index >= train_start) & (daily.index < train_end)]
        test_daily = daily[(daily.index >= test_start) & (daily.index < test_end)]

        print(f"\n{'='*60}")
        print(f"Fold {fold_num}: Train {ts} -> {te}, Test {vs} -> {ve}")
        print(f"  Train: {len(train_daily)} days, Test: {len(test_daily)} days")

        bp = find_balance_point(train_daily)
        print(f"  Balance point: {bp:.1f}F ({(bp-32)*5/9:.1f}C)")

        for model_name in MODEL_NAMES:
            try:
                y_test, y_pred = _fit_predict(model_name, train_daily, test_daily, bp, eta=3.0)
                m = compute_all_metrics(y_test, y_pred)
                tag = ""
                if m["cv_rmse"] <= 25 and abs(m["nmbe"]) <= 10:
                    tag = " PASS"
                print(f"  {model_name:6s}: CV(RMSE)={m['cv_rmse']:.2f}%  NMBE={m['nmbe']:+.2f}%  R2={m['r_squared']:.4f}{tag}")
            except Exception as e:
                print(f"  {model_name:6s}: FAILED - {e}")


if __name__ == "__main__":
    main()
