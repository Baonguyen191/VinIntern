"""Alert post-processing shared by every detector suite: sensor-fault separation, then merging."""
import numpy as np
import pandas as pd

from .detectors import EVENT_COLUMNS


def _max(a: float, b: float) -> float:
    return b if np.isnan(a) else a if np.isnan(b) else max(a, b)


def separate_sensor_faults(events: pd.DataFrame, min_overlap: float = 0.5) -> tuple[pd.DataFrame, int]:
    """Drop group A events that mostly sit inside a data-quality event of the same meter.

    A frozen or missing sensor is reported once, as DQ (stuck_sensor / data_loss); a consumption
    alarm raised on the same hours would be a second, wrong explanation of the same fault.
    Returns (kept events, number of dropped events).
    """
    dq = events[events["group"] == "DQ_quality"]
    if dq.empty:
        return events, 0
    hour = pd.Timedelta(hours=1)
    drop = []
    for (entity, key), ev in events[events["group"] != "DQ_quality"].groupby(["entity_id", "key"]):
        faults = dq[(dq["entity_id"] == entity) & (dq["key"] == key)]
        for r in ev.itertuples():
            covered = 0.0
            for f in faults.itertuples():
                overlap = (min(r.end, f.end) - max(r.start, f.start)) / hour + 1
                covered += max(overlap, 0.0)
            if covered / r.duration_h >= min_overlap:
                drop.append(r.Index)
    return events.drop(index=drop), len(drop)


def merge_alerts(events: pd.DataFrame, gap_h: int = 2) -> pd.DataFrame:
    """Merge consecutive alerts of the same meter, key, detector and type.

    Two alerts are merged when the second starts at most gap_h hours after the first ends.
    The merged alert keeps the earliest detected_at; means are duration-weighted, peaks are maxed
    and excess energy is summed.
    """
    if events.empty:
        return events
    ev = events.sort_values(["entity_id", "key", "rule_or_model", "type", "start"])
    gap = pd.Timedelta(hours=gap_h + 1)
    rows = []
    for _, grp in ev.groupby(["entity_id", "key", "rule_or_model", "type"], sort=False):
        cur = None
        for r in grp.to_dict("records"):
            if cur is not None and r["start"] <= cur["end"] + gap:
                w0, w1 = cur["duration_h"], r["duration_h"]
                for c in ("actual_mean", "expected_mean"):
                    vals = np.array([cur[c], r[c]], dtype=float)
                    wts = np.array([w0, w1], dtype=float)[~np.isnan(vals)]
                    cur[c] = float(np.dot(vals[~np.isnan(vals)], wts) / wts.sum()) if wts.size else np.nan
                cur["end"] = max(cur["end"], r["end"])
                cur["detected_at"] = min(cur["detected_at"], r["detected_at"])
                cur["duration_h"] = int((cur["end"] - cur["start"]) / pd.Timedelta(hours=1)) + 1
                cur["peak_z"] = _max(cur["peak_z"], r["peak_z"])
                cur["score"] = _max(cur["score"], r["score"])
                cur["excess_kwh"] = np.nansum([cur["excess_kwh"], r["excess_kwh"]])
                cur["n_merged"] += 1
            else:
                if cur is not None:
                    rows.append(cur)
                cur = {**r, "n_merged": 1}
        rows.append(cur)
    out = pd.DataFrame(rows)
    exp = out["expected_mean"]
    out["deviation_pct"] = ((out["actual_mean"] - exp) / exp * 100).where(exp != 0).round(2)
    out[["actual_mean", "expected_mean"]] = out[["actual_mean", "expected_mean"]].round(3)
    out["excess_kwh"] = out["excess_kwh"].round(1)
    return out[[*EVENT_COLUMNS, "n_merged"]].sort_values(["entity_id", "key", "start"]).reset_index(drop=True)
