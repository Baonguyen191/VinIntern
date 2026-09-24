"""Initial detector set. Each function returns O2-style event dicts (problem.md 5.2)."""
import numpy as np
import pandas as pd

from ..core.anomaly_detector import detect_drift_cusum
from .hourly_baseline import NIGHT_HOURS, is_night, is_operating
from .quality import detect_flatlines, detect_gaps, find_runs

EVENT_COLUMNS = [
    "event_id", "group", "module", "entity_id", "key", "type", "severity", "rule_or_model",
    "start", "end", "detected_at", "duration_h", "actual_mean", "expected_mean", "deviation_pct",
    "peak_z", "excess_kwh", "reference_window", "score",
]

ROLLING_REFERENCE = "rolling 8 weeks, same day_type x tod_slot"


def _event(entity, module, key, group, etype, severity, rule, start, end, detected_at,
           actual=np.nan, expected=np.nan, peak_z=np.nan, excess_kwh=np.nan,
           reference=ROLLING_REFERENCE, score=np.nan) -> dict:
    """detected_at = first timestamp at which the rule has enough data to fire (used for delay).

    peak_z is the peak (value - expected) / scale of a baseline detector; score is the peak raw
    model score of a detector without a baseline scale (Isolation Forest).
    """
    dev = (actual - expected) / expected * 100 if expected and not np.isnan(expected) else np.nan
    return {
        "event_id": "", "group": group, "module": module, "entity_id": entity, "key": key,
        "type": etype, "severity": severity, "rule_or_model": rule, "start": start, "end": end,
        "detected_at": detected_at,
        "duration_h": int((end - start) / pd.Timedelta(hours=1)) + 1,
        "actual_mean": round(float(actual), 3), "expected_mean": round(float(expected), 3),
        "deviation_pct": round(float(dev), 2), "peak_z": round(float(peak_z), 2),
        "excess_kwh": round(float(excess_kwh), 1), "reference_window": reference,
        "score": round(float(score), 4),
    }


def run_events(
    g: pd.DataFrame, key: str, flagged: np.ndarray, strength: pd.Series, spike_level,
    expected: pd.Series, entity: str, module: str, rule: str, reference: str,
    min_run_h: int = 3, max_gap_h: int = 1, long_h: int | None = None, strength_is_z: bool = True,
) -> list[dict]:
    """Turn flagged hours into group A events.

    Runs closer than max_gap_h are merged. run < min_run_h is kept only as a spike, and only if
    its peak strength reaches spike_level; runs >= long_h (if set) -> baseload_rise; other runs
    -> off_hours_run / high_consumption by share of operating hours.
    """
    v = g[key]
    spike_level = pd.Series(spike_level, index=g.index) if np.isscalar(spike_level) else spike_level
    op = is_operating(g["day_type"], g.index.hour.to_numpy())

    events = []
    for a, b in find_runs(flagged, max_gap=max_gap_h):
        seg = slice(a, b + 1)
        s = strength.iloc[seg].to_numpy()
        peak = float(np.nanmax(s))
        n_h = b - a + 1
        if n_h < min_run_h:
            hits = s >= spike_level.iloc[seg].to_numpy()
            if not hits.any():
                continue
            etype = "spike"
            detected_at = g.index[a + int(np.argmax(hits))]
        else:
            if long_h is not None and n_h >= long_h:
                etype = "baseload_rise"
            else:
                etype = "high_consumption" if op[seg].mean() > 0.5 else "off_hours_run"
            detected_at = g.index[a + min_run_h - 1]
        excess = (v.iloc[seg] - expected.iloc[seg]).clip(lower=0).sum()
        events.append(_event(
            entity, module, key, "A_consumption", etype, "warning", rule,
            g.index[a], g.index[b], detected_at,
            v.iloc[seg].mean(), expected.iloc[seg].mean(),
            peak_z=peak if strength_is_z else np.nan, excess_kwh=excess, reference=reference,
            score=np.nan if strength_is_z else peak,
        ))
    return events


def band_events(
    g: pd.DataFrame, key: str, band: pd.DataFrame, entity: str, module: str,
    spike_z: float = 5.0, min_run_h: int = 3, max_gap_h: int = 1,
    rule: str = "profile_band", reference: str = ROLLING_REFERENCE,
) -> list[dict]:
    """Group A: runs above a baseline band (expected / upper / scale).

    run < min_run_h and peak z >= spike_z -> spike; longer runs -> off_hours_run / high_consumption
    (by share of operating hours).
    """
    z = (g[key] - band["expected"]) / band["scale"]
    above = (g[key] > band["upper"]).to_numpy()
    return run_events(g, key, above, z, spike_z, band["expected"], entity, module, rule, reference,
                      min_run_h=min_run_h, max_gap_h=max_gap_h)


def baseload_events(
    g: pd.DataFrame, key: str, band: pd.DataFrame, entity: str, module: str,
    k_slack: float = 0.5, h: float = 4.0,
    rule: str = "cusum_night_residual", reference: str = ROLLING_REFERENCE,
) -> list[dict]:
    """Group A base-load rise: CUSUM on the daily mean night residual (01-04h)."""
    night = is_night(g.index.hour.to_numpy())
    resid = (g[key] - band["expected"])[night]
    daily = resid.groupby(resid.index.normalize()).agg(["mean", "count"])
    daily = daily.loc[daily["count"] >= 2, "mean"]
    if len(daily) < 28:
        return []

    s_plus, _, drift_up, _ = detect_drift_cusum(daily.to_numpy(), k_slack=k_slack, h=h)
    windows = []
    for i in np.flatnonzero(drift_up):
        j = i - 1
        while j >= 0 and s_plus[j] > 0:
            j -= 1
        windows.append((j + 1, i))

    merged: list[list[int]] = []
    for a, b in windows:
        if merged and (daily.index[a] - daily.index[merged[-1][1]]).days <= 1:
            merged[-1][1] = b
        else:
            merged.append([a, b])

    events = []
    for a, b in merged:
        start = daily.index[a]
        end = daily.index[b] + pd.Timedelta(hours=23)
        # First trigger is known once that day's night window (01-04h) is complete.
        first_trigger = next(i for x, i in windows if a <= i <= b)
        detected_at = daily.index[first_trigger] + pd.Timedelta(hours=NIGHT_HOURS[1])
        w = (g.index >= start) & (g.index <= end) & night
        events.append(_event(
            entity, module, key, "A_consumption", "baseload_rise", "warning", rule,
            start, end, detected_at, g.loc[w, key].mean(), band.loc[w, "expected"].mean(),
            reference=reference,
        ))
    return events


def quality_events(
    g: pd.DataFrame, key: str, entity: str, module: str, gap_h: int = 3, flat_h: int = 4,
) -> list[dict]:
    """Data-quality layer: gaps and non-zero flatlines."""
    events = []
    for s, e in detect_gaps(g[key], min_hours=gap_h):
        events.append(_event(entity, module, key, "DQ_quality", "data_loss", "warning",
                             f"dq_gap>={gap_h}h", s, e, s + pd.Timedelta(hours=gap_h - 1), reference=""))
    for s, e in detect_flatlines(g[key], min_hours=flat_h):
        events.append(_event(entity, module, key, "DQ_quality", "stuck_sensor", "warning",
                             f"dq_flatline>={flat_h}h", s, e, s + pd.Timedelta(hours=flat_h - 1),
                             actual=g.loc[s, key], reference=""))
    return events
