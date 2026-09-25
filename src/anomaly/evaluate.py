"""Event-level evaluation against labelled cases (overlap matching)."""
import numpy as np
import pandas as pd

from src.anomaly.events import overlaps


def event_metrics(events: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Event precision, event recall, and median detection delay.

    An event is a true positive when it overlaps any label of the same entity.
    A label is detected when any event overlaps it. Delay = first overlapping
    event start minus label start, clipped at 0.
    """
    n_events, n_labels = len(events), len(labels)
    tp_events = int(overlaps(events, labels).sum()) if n_events else 0

    delays = []
    detected = 0
    for e, s, t in zip(labels["entity_id"], labels["start"], labels["end"]):
        ev = events[(events["entity_id"] == e) & (events["start"] <= t) & (events["end"] >= s)]
        if len(ev):
            detected += 1
            delays.append(max((ev["start"].min() - s).total_seconds() / 3600, 0.0))

    return {
        "n_events": n_events,
        "n_labels": n_labels,
        "event_precision": tp_events / n_events if n_events else np.nan,
        "event_recall": detected / n_labels if n_labels else np.nan,
        "median_delay_h": float(np.median(delays)) if delays else np.nan,
    }


def metrics_by_type(events: pd.DataFrame, labels: pd.DataFrame, detector: str) -> pd.DataFrame:
    """One row for all labels (precision counted against every label) plus recall per type."""
    rows = [{"detector": detector, "type": "ALL", **event_metrics(events, labels)}]
    for case_type, lab in labels.groupby("type"):
        m = event_metrics(events, lab)
        rows.append({
            "detector": detector, "type": case_type,
            "n_events": np.nan, "n_labels": m["n_labels"], "event_precision": np.nan,
            "event_recall": m["event_recall"], "median_delay_h": m["median_delay_h"],
        })
    return pd.DataFrame(rows)
