"""Model-based detectors: Isolation Forest on the measured series, LightGBM quantile baseline.

Both are fitted per entity with month cross-fitting: the model that scores month m is trained on
the other 11 months of 2017. The shared set only covers one year, so this stands in for "a year of
history"; it is not strictly causal (unlike the rolling 8-week profile band of the rule detector).
Training rows that the data-quality layer flags (gaps, flatlines) are left out.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .quality import detect_flatlines, detect_gaps

CROSS_FIT_REFERENCE = "month cross-fit (other 11 months of 2017)"
# (q90 - q10) of a normal distribution = 2.563 sigma
Q_SPREAD_TO_SIGMA = 2.563


def dq_mask(series: pd.Series, gap_h: int = 3, flat_h: int = 4) -> np.ndarray:
    """Hours inside a gap or a non-zero flatline, i.e. what the data-quality layer reports."""
    m = np.zeros(len(series), dtype=bool)
    for s, e in [*detect_gaps(series, gap_h), *detect_flatlines(series, flat_h)]:
        m |= (series.index >= s) & (series.index <= e)
    return m


def calendar_features(g: pd.DataFrame) -> pd.DataFrame:
    day_type = g["day_type"].astype(str)
    temp = g["temperature"].interpolate(limit=6, limit_direction="both")
    temp = temp.fillna(temp.median())
    return pd.DataFrame({
        "hour": g.index.hour,
        "dow": g.index.dayofweek,
        "workday": (day_type == "ngay_lam_viec").astype(int),
        "holiday": (day_type == "le").astype(int),
        "temperature": temp,
        "temp_24h": temp.rolling(24, min_periods=1).mean(),
    }, index=g.index)


def _slot_reference(g: pd.DataFrame, key: str, train: np.ndarray) -> pd.Series:
    """Median of the training rows in the same (weekday/weekend) x hour slot, for every row."""
    day_group = np.where(g["day_type"].astype(str) == "ngay_lam_viec", "wd", "we")
    slot = pd.Series(day_group, index=g.index) + "_" + g.index.hour.astype(str)
    med = g.loc[train, key].groupby(slot[train]).median()
    return slot.map(med).astype(float)


def lgbm_baseline(
    g: pd.DataFrame, key: str, seed: int = 0, n_estimators: int = 300,
) -> pd.DataFrame:
    """Quantile LightGBM (q10 / q50 / q90) on calendar + temperature, no lag features.

    Lags are left out on purpose: they would carry the anomaly into the prediction and hide it.
    Returns expected (q50), q_lo, q_hi on the entity's hourly grid.
    """
    X = calendar_features(g)
    y = g[key]
    usable = y.notna().to_numpy() & ~dq_mask(y)
    month = g.index.month.to_numpy()
    out = pd.DataFrame(np.nan, index=g.index, columns=["expected", "q_lo", "q_hi"])
    for m in np.unique(month):
        train = usable & (month != m)
        test = month == m
        for alpha, col in ((0.5, "expected"), (0.1, "q_lo"), (0.9, "q_hi")):
            model = lgb.LGBMRegressor(
                objective="quantile", alpha=alpha, n_estimators=n_estimators, learning_rate=0.05,
                num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1,
                random_state=seed, verbose=-1,
            )
            model.fit(X[train], y[train])
            out.loc[test, col] = model.predict(X[test])
    return out


def lgbm_band(pred: pd.DataFrame, k: float, rel_floor: float = 0.05, abs_floor: float = 0.0) -> pd.DataFrame:
    """O1-style band from the quantile baseline: expected +- k * sigma, sigma from the q10-q90 spread."""
    spread = (pred["q_hi"] - pred["q_lo"]).abs() / Q_SPREAD_TO_SIGMA
    scale = pd.concat([spread, rel_floor * pred["expected"].abs()], axis=1).max(axis=1).clip(lower=abs_floor)
    return pd.DataFrame({
        "expected": pred["expected"],
        "lower": pred["expected"] - k * scale,
        "upper": pred["expected"] + k * scale,
        "scale": scale,
    })


def _if_features(g: pd.DataFrame, key: str) -> pd.DataFrame:
    v = g[key]
    cal = calendar_features(g)
    return pd.DataFrame({
        "value": v,
        "diff_1h": v.diff().fillna(0.0),
        "mean_24h": v.rolling(24, min_periods=12).mean(),
        "hour_sin": np.sin(2 * np.pi * cal["hour"] / 24),
        "hour_cos": np.cos(2 * np.pi * cal["hour"] / 24),
        "workday": cal["workday"],
        "temperature": cal["temperature"],
    }, index=g.index)


def iforest_scores(
    g: pd.DataFrame, key: str, quantiles: list[float], seed: int = 0, n_estimators: int = 200,
    context: bool = True,
) -> pd.DataFrame:
    """Isolation Forest on the measured series.

    context=False: value, 1 h change, 24 h mean, hour, workday, temperature (raw measured data).
    context=True: only deviations from the training median of the same weekday/weekend x hour
    slot (now, 24 h mean) plus the 1 h change. With raw features the forest cannot see
    "day-level load at night" (off_hours_run), since the value itself is common over the year, and
    a moderate deviation on one feature is diluted among the others.

    score = -score_samples (higher = more anomalous). thr_<q> is the q-quantile of the training
    rows' scores of the same fold, i.e. an alarm budget of (1 - q) of normal hours.
    expected = the same slot median (evidence and direction).
    """
    X = _if_features(g, key)
    ok = X.notna().all(axis=1).to_numpy()
    usable = ok & ~dq_mask(g[key])
    month = g.index.month.to_numpy()
    cols = ["score", "expected", *[f"thr_{q}" for q in quantiles]]
    out = pd.DataFrame(np.nan, index=g.index, columns=cols)
    for m in np.unique(month):
        train = usable & (month != m)
        test = ok & (month == m)
        ref = _slot_reference(g, key, train)
        if context:
            vs_slot = X["value"] - ref
            Xf = pd.DataFrame({"vs_slot": vs_slot,
                               "vs_slot_24h": vs_slot.rolling(24, min_periods=12).mean(),
                               "diff_1h": X["diff_1h"]}, index=X.index).fillna(0.0)
        else:
            Xf = X
        model = IsolationForest(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
        model.fit(Xf[train])
        train_score = -model.score_samples(Xf[train])
        out.loc[test, "score"] = -model.score_samples(Xf[test])
        out.loc[test, "expected"] = ref[test]
        for q in quantiles:
            out.loc[month == m, f"thr_{q}"] = float(np.quantile(train_score, q))
    return out
