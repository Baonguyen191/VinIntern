"""Agreed rule detector on one real meter series, for the demo schema (is_anomaly / anomaly_reason).

Same stack as the rule suite of the anomaly-detect pipeline: data-quality layer, profile band
(rolling 8 weeks, k = 3), night CUSUM for base-load rise (electricity only), then sensor-fault
separation and merging of consecutive alerts.
"""
import numpy as np
import pandas as pd

from .detectors import EVENT_COLUMNS, band_events, baseload_events, quality_events
from .hourly_baseline import profile_band
from .postprocess import merge_alerts, separate_sensor_faults

MODULES = {"power_active_kw": "M1_power", "cooling_kw": "M2_cooling"}

TYPE_LABELS = {
    "spike": "Đột biến",
    "off_hours_run": "Chạy ngoài giờ",
    "high_consumption": "Tiêu thụ cao trong giờ làm việc",
    "baseload_rise": "Tăng tải nền",
    "data_loss": "Mất dữ liệu",
    "stuck_sensor": "Cảm biến đơ",
}


def hourly_grid(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Rows (ts, day_type, key) of one meter -> regular hourly grid; missing hours become NaN."""
    df = df.sort_values("ts")
    grid = pd.date_range(df["ts"].min().floor("D"), df["ts"].max().ceil("h"), freq="h", name="ts")
    out = df.set_index("ts")[[key]].reindex(grid)
    known = df.assign(date=df["ts"].dt.normalize()).groupby("date")["day_type"].first().astype(str)
    fallback = pd.Series(np.where(grid.dayofweek < 5, "ngay_lam_viec", "cuoi_tuan"), index=grid)
    out["day_type"] = pd.Series(grid.normalize().map(known), index=grid).fillna(fallback)
    out["tod_slot"] = grid.hour
    return out


def run_rule_detector(df: pd.DataFrame, key: str, entity: str, k: float = 3.0) -> pd.DataFrame:
    """Alerts (O2 columns + n_merged) for one meter. df needs ts, day_type and the key column."""
    module = MODULES[key]
    g = hourly_grid(df, key)
    band = profile_band(g, key, k=k, abs_floor=0.01 * g[key].quantile(0.95))
    events = band_events(g, key, band, entity, module) + quality_events(g, key, entity, module)
    if key == "power_active_kw":
        events += baseload_events(g, key, band, entity, module)
    raw = pd.DataFrame(events, columns=EVENT_COLUMNS)
    if raw.empty:
        return raw.assign(n_merged=pd.Series(dtype=int))
    alerts, _ = separate_sensor_faults(raw)
    return merge_alerts(alerts)


def alert_reason(a) -> str:
    label = TYPE_LABELS.get(a.type, a.type)
    if a.group == "DQ_quality":
        return f"{label} {a.duration_h} h"
    return (f"{label}: {a.actual_mean:.1f} so với kỳ vọng {a.expected_mean:.1f} "
            f"({a.deviation_pct:+.0f}%), {a.duration_h} h")


def flag_hours(timestamps: pd.Series, alerts: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Demo-schema columns: is_anomaly and anomaly_reason for each timestamp."""
    ts = pd.to_datetime(timestamps).reset_index(drop=True)
    is_anomaly = pd.Series(False, index=ts.index)
    reason = pd.Series([None] * len(ts), index=ts.index, dtype=object)
    for a in alerts.sort_values("start").itertuples():
        hit = (ts >= a.start) & (ts <= a.end)
        new = hit & reason.isna()
        is_anomaly |= hit
        reason[new] = alert_reason(a)
    return is_anomaly.to_numpy(), reason.to_numpy()
