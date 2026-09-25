"""Turn point scores into alert events: merge, minimum duration, suppression, budget, O2 JSON.

The three alert parameters required by the spec are explicit arguments:
`max_gap` (how adjacent cells merge), `min_duration`, and `budget_per_day`.
"""
import numpy as np
import pandas as pd

from src.anomaly.quality import mask_runs

EVENT_COLUMNS = ["entity_id", "detector", "start", "end", "duration_h", "peak_score", "peak_ts"]


def points_to_events(
    scores: pd.Series,
    threshold: float,
    entity_id: str,
    detector: str,
    max_gap: int = 1,
    min_duration: int = 1,
    step_h: int = 1,
) -> pd.DataFrame:
    """Flag scores >= threshold, merge flags separated by <= `max_gap` cells, drop short events."""
    flags = (scores >= threshold).to_numpy()
    runs = mask_runs(flags)
    merged: list[list[int]] = []
    for a, b in runs:
        if merged and a - merged[-1][1] - 1 <= max_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])

    idx = scores.index
    v = scores.to_numpy(dtype=float)
    rows = []
    for a, b in merged:
        n = b - a + 1
        if n < min_duration:
            continue
        peak = a + int(np.nanargmax(v[a : b + 1]))
        end = idx[b] + pd.Timedelta(hours=step_h - 1)
        rows.append([entity_id, detector, idx[a], end, n * step_h, v[peak], idx[peak]])
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def interval_events(
    score: pd.Series,
    window: int,
    threshold: float,
    entity_id: str,
    detector: str,
    max_gap: int = 1,
) -> pd.DataFrame:
    """Sustained deviations: trailing `window`-hour mean of the score >= threshold.

    Catches stretches where no single hour crosses the point threshold but the
    whole stretch is consistently off. A flag at hour t covers t-window+1..t;
    the event is then trimmed to the first and last hour whose own score
    reaches the threshold, so its bounds are hours that actually deviate.
    """
    rolled = score.rolling(window, min_periods=window).mean()
    ev = points_to_events(rolled, threshold, entity_id, detector, max_gap=max_gap)
    if ev.empty:
        return ev
    starts, ends = [], []
    for s, t in zip(ev["start"] - pd.Timedelta(hours=window - 1), ev["end"]):
        hot = score[s:t]
        hot = hot[hot >= threshold]
        starts.append(hot.index[0] if len(hot) else s)
        ends.append(hot.index[-1] if len(hot) else t)
    ev["start"], ev["end"] = starts, ends
    ev["duration_h"] = ((ev["end"] - ev["start"]) / pd.Timedelta(hours=1)).astype(int) + 1
    return ev


def merge_overlapping(events: pd.DataFrame, max_gap_h: int = 1) -> pd.DataFrame:
    """Merge events of the same entity that overlap or are <= max_gap_h apart."""
    if events.empty:
        return events
    rows = []
    for entity_id, g in events.sort_values("start").groupby("entity_id", sort=False):
        cur = None
        for ev in g.to_dict("records"):
            if cur is not None and ev["start"] <= cur["end"] + pd.Timedelta(hours=max_gap_h + 1):
                cur["end"] = max(cur["end"], ev["end"])
                if ev["peak_score"] > cur["peak_score"]:
                    cur["peak_score"], cur["peak_ts"] = ev["peak_score"], ev["peak_ts"]
                if ev["detector"] not in cur["detector"].split("+"):
                    cur["detector"] += "+" + ev["detector"]
            else:
                if cur is not None:
                    rows.append(cur)
                cur = dict(ev)
        rows.append(cur)
    out = pd.DataFrame(rows)
    out["duration_h"] = ((out["end"] - out["start"]) / pd.Timedelta(hours=1)).astype(int) + 1
    return out.reset_index(drop=True)


def split_point_interval(
    point_events: pd.DataFrame, sustained_events: pd.DataFrame, point_max_h: int = 2
) -> pd.DataFrame:
    """Label each alert by scope.

    scope="point": a point-detector event lasting <= point_max_h hours.
    scope="interval": a longer point-detector event, or a sustained event;
    overlapping ones are merged into one interval. Short point events inside
    an interval are kept, so a spike on top of a raised stretch stays visible.
    """
    is_point = point_events["duration_h"] <= point_max_h
    points = point_events[is_point].assign(scope="point")
    intervals = merge_overlapping(
        pd.concat([point_events[~is_point], sustained_events], ignore_index=True)
    ).assign(scope="interval")
    return pd.concat([points, intervals], ignore_index=True).sort_values(
        ["entity_id", "start"], ignore_index=True
    )


def overlaps(events: pd.DataFrame, intervals: pd.DataFrame) -> np.ndarray:
    """True where an event overlaps any [start, end] interval (same entity)."""
    hit = np.zeros(len(events), dtype=bool)
    for i, (e, s, t) in enumerate(zip(events["entity_id"], events["start"], events["end"])):
        iv = intervals[intervals["entity_id"] == e]
        hit[i] = ((iv["start"] <= t) & (iv["end"] >= s)).any()
    return hit


def suppress_sensor_faults(events: pd.DataFrame, faults: pd.DataFrame) -> pd.DataFrame:
    """Drop operational events that overlap a data-quality fault (alarm on bad input is noise)."""
    return events[~overlaps(events, faults)].reset_index(drop=True)


def apply_budget(events: pd.DataFrame, n_days: float, budget_per_day: float) -> pd.DataFrame:
    """Keep the highest-scoring events per entity: at most budget x days of them."""
    k = int(np.floor(budget_per_day * n_days))
    return (
        events.sort_values("peak_score", ascending=False)
        .groupby("entity_id", group_keys=False)
        .head(k)
        .sort_values(["entity_id", "start"], ignore_index=True)
    )


def classify_hourly(events: pd.DataFrame, day_type: pd.Series) -> pd.Series:
    """after_hours when most of the event falls outside workday 07–19h; otherwise
    spike for point events and sustained_high for interval events."""
    off = (day_type.to_numpy() != "ngay_lam_viec") | (day_type.index.hour < 7) | (day_type.index.hour >= 19)
    off = pd.Series(off, index=day_type.index)
    scopes = events["scope"] if "scope" in events else pd.Series("point", index=events.index)
    kinds = []
    for s, t, scope in zip(events["start"], events["end"], scopes):
        if off[s:t].mean() > 0.5:
            kinds.append("after_hours")
        else:
            kinds.append("sustained_high" if scope == "interval" else "spike")
    return pd.Series(kinds, index=events.index, dtype=object)


def to_o2(
    event: pd.Series,
    event_id: str,
    group: str,
    module: str,
    key: str,
    value: float,
    baseline: float,
    reference_window: str,
    severity: str = "warning",
    suggested_action: str = "",
) -> dict:
    """O2 event record (problem.md §5.2) with the evidence needed to audit it."""
    deviation = (value - baseline) / baseline * 100 if baseline else None
    return {
        "event_id": event_id,
        "group": group,
        "module": module,
        "entity_id": event["entity_id"],
        "type": event["type"],
        "scope": event.get("scope", "point"),
        "severity": severity,
        "rule_or_model": event["detector"],
        "start": event["start"].isoformat(),
        "end": event["end"].isoformat(),
        "evidence": {
            "key": key,
            "value": round(float(value), 3),
            "reference": round(float(baseline), 3),
            "deviation_pct": None if deviation is None else round(float(deviation), 1),
            "peak_ts": event["peak_ts"].isoformat(),
            "peak_score": round(float(event["peak_score"]), 3),
            "reference_window": reference_window,
        },
        "suggested_action": suggested_action,
    }
