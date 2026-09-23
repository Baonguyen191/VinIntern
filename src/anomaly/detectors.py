"""Initial detector set. Each function returns O2-style event dicts (problem.md 5.2)."""
import numpy as np
import pandas as pd

from ..core.anomaly_detector import detect_drift_cusum
from .hourly_baseline import NIGHT_HOURS, is_night, is_operating
from .quality import detect_flatlines, detect_gaps, find_runs

EVENT_COLUMNS = [
    "event_id", "group", "module", "entity_id", "key", "type", "severity", "rule_or_model",
    "start", "end", "detected_at", "duration_h", "actual_mean", "expected_mean", "deviation_pct",
    "peak_z", "excess_kwh", "reference_window",
]

ROLLING_REFERENCE = "rolling 8 weeks, same day_type x tod_slot"


def _event(entity, module, key, group, etype, severity, rule, start, end, detected_at,
           actual=np.nan, expected=np.nan, peak_z=np.nan, excess_kwh=np.nan,
           reference=ROLLING_REFERENCE) -> dict:
    """detected_at = first timestamp at which the rule has enough data to fire (used for delay)."""
    dev = (actual - expected) / expected * 100 if expected and not np.isnan(expected) else np.nan
    return {
        "event_id": "", "group": group, "module": module, "entity_id": entity, "key": key,
        "type": etype, "severity": severity, "rule_or_model": rule, "start": start, "end": end,
        "detected_at": detected_at,
        "duration_h": int((end - start) / pd.Timedelta(hours=1)) + 1,
        "actual_mean": round(float(actual), 3), "expected_mean": round(float(expected), 3),
        "deviation_pct": round(float(dev), 2), "peak_z": round(float(peak_z), 2),
        "excess_kwh": round(float(excess_kwh), 1), "reference_window": reference,
    }


def band_events(
    g: pd.DataFrame, key: str, band: pd.DataFrame, entity: str, module: str,
    spike_z: float = 5.0, min_run_h: int = 3, max_gap_h: int = 1,
) -> list[dict]:
    """Group A: runs above the profile band.

    run < min_run_h and peak z >= spike_z -> spike; longer runs -> off_hours_run / high_consumption
    (by share of operating hours), or co2_sustained_high for the CO2 key.
    """
    v = g[key]
    z = (v - band["expected"]) / band["scale"]
    above = (v > band["upper"]).to_numpy()
    op = is_operating(g["day_type"], g.index.hour.to_numpy())
    is_power = key == "power_active_kw"
    group = "A_consumption" if is_power else "A_level"

    events = []
    for a, b in find_runs(above, max_gap=max_gap_h):
        seg = slice(a, b + 1)
        peak = float(z.iloc[seg].max())
        if b - a + 1 < min_run_h:
            if peak < spike_z:
                continue
            etype = "spike"
            detected_at = g.index[a + int(np.argmax(z.iloc[seg].to_numpy() >= spike_z))]
        else:
            etype = "co2_sustained_high" if key == "co2_ppm" else (
                "high_consumption" if op[seg].mean() > 0.5 else "off_hours_run")
            detected_at = g.index[a + min_run_h - 1]
        excess = (v.iloc[seg] - band["expected"].iloc[seg]).clip(lower=0).sum() if is_power else np.nan
        events.append(_event(
            entity, module, key, group, etype, "warning", "profile_band",
            g.index[a], g.index[b], detected_at,
            v.iloc[seg].mean(), band["expected"].iloc[seg].mean(), peak, excess,
        ))
    return events


def leak_events(
    g: pd.DataFrame, key: str, band: pd.DataFrame, entity: str, module: str,
    k_slack: float = 0.5, h: float = 4.0,
) -> list[dict]:
    """Group A leak / base-load rise: CUSUM on the daily mean night residual (01-04h)."""
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
            entity, module, key, "A_consumption", "leak", "warning", "cusum_night_residual",
            start, end, detected_at, g.loc[w, key].mean(), band.loc[w, "expected"].mean(),
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


def efficiency_events(
    g: pd.DataFrame, entity: str, chiller: dict, reference: tuple[pd.Timestamp, pd.Timestamp],
    threshold_pct: float = 8.0, confirm_days: int = 3,
) -> tuple[list[dict], pd.Series]:
    """Group B: kW/kW_cooling vs a quadratic-in-PLR model fitted on a FIXED reference window.

    Flags days where the rolling confirm_days median of the daily deviation >= threshold_pct.
    """
    kpk = g["kw_per_kw_cooling"]
    plr = (g["cooling_kw"] / chiller["q_rated_kw"]).clip(0, 1)
    running = kpk.notna() & (plr >= chiller["min_plr"])
    in_ref = running & (g.index >= reference[0]) & (g.index < reference[1])
    expected = pd.Series(np.nan, index=g.index)
    if in_ref.sum() < 50:
        return [], expected

    coef = np.polyfit(plr[in_ref], kpk[in_ref], 2)
    expected = pd.Series(np.polyval(coef, plr), index=g.index).where(running)
    dev = (kpk - expected) / expected * 100

    daily = dev.groupby(dev.index.normalize()).agg(["median", "count"])
    daily = daily.loc[(daily["count"] >= 4) & (daily.index >= reference[1]), "median"]
    confirmed = daily.rolling(confirm_days, min_periods=confirm_days).median()
    flagged = (confirmed >= threshold_pct).to_numpy()

    ref_label = f"{reference[0].date()}..{reference[1].date()} (fixed, virtual maintenance)"
    events = []
    for a, b in find_runs(flagged, max_gap=2):
        start = daily.index[a]
        end = daily.index[b] + pd.Timedelta(hours=23)
        w = (g.index >= start) & (g.index <= end) & running.to_numpy()
        events.append(_event(
            entity, "M2_chiller", "kw_per_kw_cooling", "B_efficiency", "efficiency_degradation",
            "info", "fixed_reference_regression", start, end, start + pd.Timedelta(hours=23),
            kpk[w].mean(), expected[w].mean(), reference=ref_label,
        ))
    return events, expected
