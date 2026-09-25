"""Point-level anomaly scorers. Each returns a score per timestamp; higher = more anomalous.

- profile_band_scores  rule layer: robust z vs a rolling hour x day-type profile
- efficiency_scores    group B: daily load vs cooling regression on a lagged reference window
- deviation_scores     baseline-deviation branch on the daily baseline CSV structure
- lightgbm_scores      ML baseline: LightGBM regression, robust z of the residual
- fuse_scores          score-level fusion (e.g. LightGBM z with Isolation Forest)
- isolation_forest_scores  ML comparison on the same features as the rule layer
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

MAD_SCALE = 1.4826
WORKDAY = "ngay_lam_viec"


def _robust_z(x: np.ndarray, ref: np.ndarray, floor: float) -> np.ndarray:
    med = np.nanmedian(ref)
    mad = MAD_SCALE * np.nanmedian(np.abs(ref - med))
    return (x - med) / max(mad, floor)


# ---------------------------------------------------------------- rule layer

def rolling_profile(
    values: pd.Series, day_type: pd.Series, n_workday: int = 20, n_offday: int = 8
) -> pd.DataFrame:
    """Rolling median and MAD per (workday/off-day, hour) slot.

    Uses only earlier valid occurrences of the same slot (about four weeks),
    so the profile follows the season and the current point never scores
    itself. Missing hours are skipped rather than counted in the window, so a
    data gap does not leave the profile empty for weeks afterwards. Hours with
    no day type fall back to the calendar (Mon–Fri = workday).
    """
    dt = day_type.to_numpy()
    workday = np.where(pd.isna(dt), values.index.dayofweek < 5, dt == WORKDAY)
    df = pd.DataFrame({
        "v": values.to_numpy(dtype=float),
        "slot": np.where(workday, "w", "o"),
        "hour": values.index.hour,
    }, index=values.index)
    med = pd.Series(np.nan, index=df.index)
    mad = pd.Series(np.nan, index=df.index)
    for (slot, _), g in df.groupby(["slot", "hour"]):
        window = n_workday if slot == "w" else n_offday
        prev = g["v"].dropna().shift(1)
        m = prev.rolling(window, min_periods=window // 2).median()
        dev = (prev - m).abs().rolling(window, min_periods=window // 2).median()
        med[g.index] = m.reindex(g.index).ffill()
        mad[g.index] = (MAD_SCALE * dev).reindex(g.index).ffill()
    return pd.DataFrame({"baseline": med, "mad": mad})


def profile_band_scores(values: pd.Series, day_type: pd.Series, rel_floor: float = 0.05) -> pd.DataFrame:
    """Robust z of each hour vs its rolling profile.

    The MAD is floored at `rel_floor` x baseline so very stable slots (flat
    night load) do not turn small noise into huge z values.
    """
    prof = rolling_profile(values, day_type)
    scale = np.maximum(prof["mad"], rel_floor * prof["baseline"].abs()).replace(0, np.nan)
    z = (values - prof["baseline"]) / scale
    return pd.DataFrame({"score": z, "baseline": prof["baseline"], "scale": scale, "value": values})


# ---------------------------------------------------------------- group B

def efficiency_scores(
    power: pd.Series,
    cooling: pd.Series,
    ref_days: int = 28,
    lag_days: int = 21,
    min_cooling_frac: float = 0.2,
) -> pd.DataFrame:
    """Daily kWh vs cooling kWh regression fitted on a lagged reference window.

    For day t the reference is days [t - lag - ref, t - lag). The output is the
    percent excess of actual over predicted load: same cooling output, more
    input = efficiency loss. Days with low cooling are skipped (ratio unstable).
    """
    daily = pd.DataFrame({"p": power, "c": cooling}).resample("D").sum(min_count=20)
    p, c = daily["p"].to_numpy(), daily["c"].to_numpy()
    active = c > min_cooling_frac * np.nanmedian(c)
    pred = np.full(len(daily), np.nan)
    for t in range(lag_days + ref_days, len(daily)):
        ref = slice(t - lag_days - ref_days, t - lag_days)
        ok = active[ref] & np.isfinite(p[ref]) & np.isfinite(c[ref])
        if ok.sum() < ref_days // 2 or not active[t]:
            continue
        slope, intercept = np.polyfit(c[ref][ok], p[ref][ok], 1)
        pred[t] = intercept + slope * c[t]
    dev = (p - pred) / pred
    return pd.DataFrame(
        {"score": dev, "baseline": pred, "value": p, "cooling": c}, index=daily.index
    )


# ---------------------------------------------------------------- baseline-deviation branch

def cusum(z: np.ndarray, k: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """One-sided upper/lower CUSUM on standardised residuals (NaN counts as 0).

    No restart: the statistic stays above the alarm level while the shift
    lasts, so thresholding it gives one event per sustained deviation.
    """
    s_up = np.zeros(len(z))
    s_dn = np.zeros(len(z))
    for i in range(1, len(z)):
        zi = 0.0 if np.isnan(z[i]) else z[i]
        s_up[i] = max(0.0, s_up[i - 1] + zi - k)
        s_dn[i] = max(0.0, s_dn[i - 1] - zi - k)
    return s_up, s_dn


def deviation_scores(
    df: pd.DataFrame,
    model: str = "dow_median",
    actual_col: str = "actual_kWh",
    ref_days: int = 60,
    k: float = 0.5,
) -> pd.DataFrame:
    """Score a baseline CSV (date, actual, pred_<model>, residual_<model>).

    Returns robust z of the residual (short deviations) and the CUSUM statistic
    (sustained deviations). The first `ref_days` rows set the residual scale.
    Works on any model output with the template columns, so it does not depend
    on which team's baseline produced it.
    """
    pred = df[f"pred_{model}"].to_numpy(dtype=float)
    resid = df[actual_col].to_numpy(dtype=float) - pred
    ref = resid[:ref_days]
    z = _robust_z(resid, ref, floor=1e-9)
    s_up, s_dn = cusum(z, k=k)
    return pd.DataFrame({
        "score": np.abs(z),
        "z": z,
        "cusum": np.maximum(s_up, s_dn),
        "baseline": pred,
        "value": df[actual_col].to_numpy(dtype=float),
    }, index=pd.to_datetime(df["date"]))


# ---------------------------------------------------------------- ML comparison

def hourly_features(values: pd.Series, day_type: pd.Series, band: pd.DataFrame) -> pd.DataFrame:
    """Features relative to the rolling profile, so a model fitted in winter
    still applies in summer (absolute load levels shift with the season)."""
    hour = values.index.hour
    ratio = values / band["baseline"].replace(0, np.nan)
    return pd.DataFrame({
        "ratio": ratio,
        "profile_z": band["score"],
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "workday": (day_type.to_numpy() == WORKDAY).astype(float),
        "ratio_mean_24h": ratio.rolling(24, min_periods=12).mean(),
        "z_std_24h": band["score"].rolling(24, min_periods=12).std(),
        "ratio_diff_1h": ratio.diff(),
    }, index=values.index)


def lightgbm_scores(
    values: pd.Series,
    day_type: pd.Series,
    temperature: pd.Series,
    profile_baseline: pd.Series,
    train_mask: np.ndarray,
    extra_features: pd.DataFrame | None = None,
    rel_floor: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """LightGBM regression baseline; robust z of actual vs prediction.

    The target is load / rolling profile, not raw load: trees cannot
    extrapolate, and a model fitted on winter levels would miss summer ones.
    Prediction = predicted ratio x profile baseline. The objective is L1
    (median) so anomalies in the training window pull the fit less.

    `extra_features` (same index) are appended to the model inputs, e.g. the
    Isolation Forest score from `if_score_features`.
    """
    idx = values.index
    base = profile_baseline.replace(0, np.nan)
    ratio = values / base
    temp = temperature.reindex(idx).interpolate(limit=6)
    X = pd.DataFrame({
        "hour": idx.hour,
        "dow": idx.dayofweek,
        "workday": (day_type.to_numpy() == WORKDAY).astype(int),
        "temp": temp.to_numpy(),
        "temp_dev_7d": (temp - temp.rolling(168, min_periods=24).mean()).to_numpy(),
        "ratio_lag_168h": ratio.shift(168).to_numpy(),
    }, index=idx)
    if extra_features is not None:
        X = X.join(extra_features.reindex(idx))

    fit = train_mask & np.isfinite(ratio.to_numpy())
    model = lgb.LGBMRegressor(
        objective="l1", n_estimators=300, learning_rate=0.05, num_leaves=15,
        min_child_samples=20, random_state=seed, verbose=-1,
    )
    model.fit(X[fit], ratio[fit])
    ratio_hat = pd.Series(model.predict(X), index=idx)

    resid_ratio = (ratio - ratio_hat)[fit]
    mad = MAD_SCALE * np.median(np.abs(resid_ratio - np.median(resid_ratio)))
    pred = ratio_hat * base
    scale = np.maximum(mad * base, rel_floor * pred.abs()).replace(0, np.nan)
    return pd.DataFrame({
        "score": (values - pred) / scale, "baseline": pred, "scale": scale, "value": values,
    })


def if_score_features(if_scores: pd.Series, lagged: bool) -> pd.DataFrame:
    """Isolation Forest score as a LightGBM input.

    lagged=False: score of the same hour. It is computed from that hour's load,
    so the regression can learn to follow anomalies and hide them.
    lagged=True: mean score over the previous 24 hours (no current-hour
    information): tells the model the building was behaving oddly just before.
    """
    if lagged:
        return pd.DataFrame({"if_score_prev_24h": if_scores.shift(1).rolling(24, min_periods=12).mean()})
    return pd.DataFrame({"if_score": if_scores})


def normalize_to_train(score: pd.Series, train_mask: np.ndarray) -> pd.Series:
    """Robust z of a score against its own distribution on the train rows.

    Puts scores with different units (residual z, Isolation Forest score) on
    one scale so they can be compared and combined.
    """
    ref = score.to_numpy(dtype=float)[train_mask]
    return pd.Series(_robust_z(score.to_numpy(dtype=float), ref, floor=1e-9), index=score.index)


def fuse_scores(scores: dict[str, pd.Series], train_mask: np.ndarray, how: str = "max") -> pd.Series:
    """Combine detector scores after normalising each to the train distribution.

    how="max": an hour is as anomalous as its most alarmed detector.
    how="mean": detectors must agree to push the fused score up.
    """
    z = pd.DataFrame({name: normalize_to_train(s, train_mask) for name, s in scores.items()})
    return z.max(axis=1, skipna=True) if how == "max" else z.mean(axis=1, skipna=True)


def isolation_forest_scores(
    features: pd.DataFrame, train_mask: np.ndarray, seed: int = 0
) -> pd.Series:
    """Fit on clean train rows, score every row with complete features."""
    ok = features.notna().all(axis=1).to_numpy()
    model = IsolationForest(n_estimators=200, random_state=seed)
    model.fit(features[ok & train_mask])
    scores = pd.Series(np.nan, index=features.index)
    scores[ok] = -model.score_samples(features[ok])
    return scores
