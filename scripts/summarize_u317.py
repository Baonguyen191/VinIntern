"""Tong hop du lieu cua mot user EWELD (mac dinh U317) dung trong danh gia.

Script nay chi DOC du lieu, khong ghi de len bat cu thu gi trong EWELD/.
No dung lai dung cac module trong src/ ma pipeline su dung, nen moi con so
in ra phan anh chinh xac nhung gi model nhin thay.

Chay:
    python scripts/summarize_u317.py
    python scripts/summarize_u317.py --user U290
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.io.eweld_loader import (  # noqa: E402
    load_electricity,
    load_weather,
    get_user_cluster,
    get_weather_station,
)
from src.io.eweld_merger import merge_electricity_weather  # noqa: E402
from src.core.cross_validation import generate_folds  # noqa: E402
from src.core.metrics import cv_rmse, nmbe, r_squared  # noqa: E402


def rule(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def pct(series: pd.Series, points) -> dict:
    return {p: round(float(np.percentile(series, p)), 4) for p in points}


def zero_streaks(values: np.ndarray) -> dict:
    """Phan loai cac doan zero lien tiep theo do dai.

    Zero KHONG phai missing data - no la shutdown that (bao tri, le, dong cua).
    Doan ngan (1-4 khung = 15-60 phut) thuong la nhieu cam bien;
    doan dai (>96 khung = >1 ngay) la dung may that.
    """
    is_zero = values == 0
    if not is_zero.any():
        return {"total_zero": 0}

    # tim do dai cac doan True lien tiep
    idx = np.flatnonzero(np.diff(np.concatenate(([0], is_zero.view(np.int8), [0]))))
    lengths = idx[1::2] - idx[::2]

    lo, hi = lengths.min(), lengths.max()
    return {
        "total_zero": int(is_zero.sum()),
        "zero_pct": round(float(is_zero.mean() * 100), 3),
        "runs": int(len(lengths)),
        "run_min": int(lo),
        "run_median": float(np.median(lengths)),
        "run_max": int(hi),
        "runs_1_4": int(((lengths >= 1) & (lengths <= 4)).sum()),
        "runs_5_96": int(((lengths >= 5) & (lengths <= 96)).sum()),
        "runs_gt_96": int((lengths > 96).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="U317")
    args = ap.parse_args()
    user = args.user

    cluster = get_user_cluster(user)
    station = get_weather_station(cluster)
    elec = load_electricity(user)
    weather = load_weather(station)
    merged = merge_electricity_weather(elec, weather)
    v = merged["Value"]

    print(f"USER {user}  |  cluster {cluster}  |  weather station {station}")

    # ---------------------------------------------------------------- 1
    rule("1. RAW ELECTRICITY FILE")
    print(f"rows              : {len(elec)}")
    print(f"range             : {elec.index[0]}  ->  {elec.index[-1]}")
    print(f"span              : {(elec.index[-1] - elec.index[0]).days / 365.25:.2f} years")
    print(f"dtypes            : {dict(elec.dtypes.astype(str))}")
    print(f"nulls             : {int(elec.isna().sum().sum())}")
    print(f"duplicate ts      : {int(elec.index.duplicated().sum())}")
    print(f"negative values   : {int((elec['Value'] < 0).sum())}")
    print(f"timestamp step    : {elec.index.to_series().diff().value_counts().to_dict()}")

    # ---------------------------------------------------------------- 2
    rule("2. MERGED DATASET (electricity joined with weather, -15 min shift)")
    print(f"raw elec rows     : {len(elec)}")
    print(f"merged rows       : {len(merged)}")
    print(f"dropped by merge  : {len(elec) - len(merged)}   (dropna on Temperature(F))")
    print(f"merged range      : {merged.index[0]}  ->  {merged.index[-1]}")
    print(f"weather {station} range : {weather.index[0]}  ->  {weather.index[-1]}")
    print(f"weather columns   : {list(weather.columns)}")

    # ---------------------------------------------------------------- 3
    rule("3. VALUE DISTRIBUTION (merged)")
    print(f"count  : {len(v)}")
    print(f"mean   : {v.mean():.4f}")
    print(f"std    : {v.std():.4f}")
    print(f"min    : {v.min():.4f}")
    print(f"max    : {v.max():.4f}")
    print(f"CV = std/mean : {v.std() / v.mean():.4f}")
    print(f"percentiles   : {pct(v, [0, 1, 2.5, 5, 25, 50, 75, 95, 97.5, 99, 100])}")
    print()
    print("zero analysis:")
    for k, val in zero_streaks(v.values).items():
        print(f"  {k:14s}: {val}")

    # ---------------------------------------------------------------- 4
    rule("4. FOLD STRUCTURE (expanding window, 2y initial train, 1y test)")
    folds = generate_folds(merged.index[0], merged.index[-1])
    print(f"{len(folds)} folds generated")
    print()
    print(f"{'fold':<6}{'train start':<12}{'train end':<12}{'n_train':>9}"
          f"{'tr_mean':>9}{'tr_std':>8}   {'test start':<12}{'test end':<12}"
          f"{'n_test':>8}{'te_mean':>9}{'te_std':>8}{'te/tr':>7}")
    for i, (a, b, c, d) in enumerate(folds):
        tr = merged[(merged.index >= a) & (merged.index < b)]
        te = merged[(merged.index >= c) & (merged.index < d)]
        print(f"{i + 1:<6}{str(a.date()):<12}{str(b.date()):<12}{len(tr):>9}"
              f"{tr['Value'].mean():>9.2f}{tr['Value'].std():>8.2f}   "
              f"{str(c.date()):<12}{str(d.date()):<12}{len(te):>8}"
              f"{te['Value'].mean():>9.2f}{te['Value'].std():>8.2f}"
              f"{te['Value'].mean() / tr['Value'].mean():>7.3f}")

    # ---------------------------------------------------------------- 5
    rule("5. YEAR-OVER-YEAR (drift check)")
    g = merged.groupby(merged.index.year)["Value"]
    yearly = pd.DataFrame({
        "n": g.count(), "mean": g.mean(), "std": g.std(),
        "median": g.median(), "p95": g.quantile(0.95), "max": g.max(),
    }).round(2)
    print(yearly.to_string())
    if len(yearly) > 1:
        first, last = yearly["mean"].iloc[0], yearly["mean"].iloc[-1]
        print(f"\nmean {yearly.index[0]} -> {yearly.index[-1]}: "
              f"{first:.2f} -> {last:.2f}  ({last / first - 1:+.1%})")
        print(f"p95  {yearly.index[0]} -> {yearly.index[-1]}: "
              f"{yearly['p95'].iloc[0]:.2f} -> {yearly['p95'].iloc[-1]:.2f}  "
              f"({yearly['p95'].iloc[-1] / yearly['p95'].iloc[0] - 1:+.1%})")

    # ---------------------------------------------------------------- 6
    rule("6. SEASONALITY (mean Value by month, averaged over years)")
    mo = merged.groupby(merged.index.month)["Value"].mean().round(2)
    print(" ".join(f"{m:>6}" for m in range(1, 13)))
    print(" ".join(f"{mo[m]:>6.2f}" for m in range(1, 13)))
    print(f"peak month {int(mo.idxmax())} = {mo.max():.2f} | "
          f"trough month {int(mo.idxmin())} = {mo.min():.2f} | "
          f"ratio = {mo.max() / mo.min():.2f}")

    # ---------------------------------------------------------------- 7
    rule("7. INTRADAY PROFILE (mean Value by hour)")
    wd_mask = merged.index.dayofweek < 5
    wd = merged[wd_mask].groupby(merged[wd_mask].index.hour)["Value"].mean()
    we = merged[~wd_mask].groupby(merged[~wd_mask].index.hour)["Value"].mean()
    print("hour   : " + "".join(f"{h:>7}" for h in range(24)))
    print("weekday: " + "".join(f"{wd[h]:>7.2f}" for h in range(24)))
    print("weekend: " + "".join(f"{we[h]:>7.2f}" for h in range(24)))
    print()
    print(f"weekday peak {int(wd.idxmax()):>2}h = {wd.max():.2f} | "
          f"trough {int(wd.idxmin()):>2}h = {wd.min():.2f} | ratio {wd.max() / wd.min():.2f}")
    print(f"weekend peak {int(we.idxmax()):>2}h = {we.max():.2f} | "
          f"trough {int(we.idxmin()):>2}h = {we.min():.2f} | ratio {we.max() / we.min():.2f}")
    print(f"weekday/weekend overall ratio = {wd.mean() / we.mean():.3f}")
    print(f"share of rows that are weekday = {wd_mask.mean():.3f}")

    # ---------------------------------------------------------------- 8
    rule("8. TEMPERATURE AND LOAD-TEMPERATURE RELATION")
    t = merged["Temperature(F)"]
    print(f"temp min/mean/max : {t.min():.2f} / {t.mean():.2f} / {t.max():.2f} (F)")
    print(f"temp percentiles  : {pct(t, [1, 5, 25, 50, 75, 95, 99])}")
    print(f"humidity max      : {merged['Humidity(%)'].max():.1f} %")
    print()
    print(f"corr(Value, Temp)           : {v.corr(t):+.4f}")
    day = merged[(merged.index.hour >= 9) & (merged.index.hour <= 18)]
    print(f"corr(Value, Temp) 09-18h    : "
          f"{day['Value'].corr(day['Temperature(F)']):+.4f}")
    night = merged[(merged.index.hour < 6)]
    print(f"corr(Value, Temp) 00-06h    : "
          f"{night['Value'].corr(night['Temperature(F)']):+.4f}")

    # ---------------------------------------------------------------- 9
    rule("9. METRIC IDENTITY CHECK  CV(RMSE) = sqrt(1 - R2) x CV(data)")
    print("Ap dung cho Tier A tren tung fold, de doi chieu summary_table.csv.")
    print("Duoi day la R2 toi thieu can dat de CV(RMSE) <= 25%.")
    print()
    cv_data = v.std() / v.mean()
    needed_r2 = 1 - (0.25 / cv_data) ** 2
    print(f"CV(data) toan bo merged      = {cv_data:.4f}")
    print(f"R2 can dat de CV(RMSE)<=25%  = {needed_r2:.4f}")
    print()
    print("Doi chieu voi ket qua thuc te trong results/eweld/towt/summary_table.csv:")
    print(f"{'fold':<6}{'test CV':>9}{'R2 req':>9}{'R2 act':>9}{'CV act':>9}")
    reported = {1: (0.33925621865853316, 0.7657583080604287),
                2: (0.4934999541618094, 0.5723182804854754),
                3: (0.40494271394581304, 0.7242918445913135),
                4: (0.3666825280673115, 0.7360224768573682)}
    for i, (a, b, c, d) in enumerate(folds):
        te = merged[(merged.index >= c) & (merged.index < d)]["Value"]
        cv_te = te.std() / te.mean()
        r2_req = 1 - (0.25 / cv_te) ** 2
        cv_act, r2_act = reported[i + 1]
        print(f"{i + 1:<6}{cv_te:>9.4f}{r2_req:>9.4f}{r2_act:>9.4f}{cv_act:>9.4f}")

    rule("DONE")


if __name__ == "__main__":
    main()
