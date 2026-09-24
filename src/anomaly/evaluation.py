"""Event-level evaluation (problem.md 10: precision is counted per event, not per sample)."""
import numpy as np
import pandas as pd

# CUSUM / confirmation-window detectors fire after the anomaly has built up.
TOLERANCE_AFTER = pd.Timedelta(hours=24)

AMPLITUDE_TYPES = {"spike", "off_hours_run", "baseload_rise"}


def match_events(labels: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per-case table, events with match column).

    An event matches a label on same entity + key and time overlap [start, end + tolerance].
    Events that only hit natural (pre-existing) data-quality windows are marked "natural" and
    left out of precision.
    """
    ev = events.copy()
    ev["match"] = "false_positive"
    ev["matched_case_id"] = ""

    def overlapping(lab):
        return ev[(ev["entity_id"] == lab.entity_id) & (ev["key"] == lab.key)
                  & (ev["start"] <= lab.end + TOLERANCE_AFTER) & (ev["end"] >= lab.start)]

    for lab in labels[labels["source"] == "natural"].itertuples():
        ev.loc[overlapping(lab).index, "match"] = "natural"

    rows = []
    for lab in labels[labels["source"] == "injected"].itertuples():
        hit = overlapping(lab)
        ev.loc[hit.index, "match"] = "true_positive"
        ev.loc[hit.index, "matched_case_id"] = lab.case_id
        types = set(hit["type"])
        delay = (hit["detected_at"].min() - lab.start) / pd.Timedelta(hours=1) if len(hit) else np.nan
        rows.append({
            "case_id": lab.case_id, "entity_id": lab.entity_id, "key": lab.key,
            "group": lab.group, "anomaly_type": lab.anomaly_type,
            "start": lab.start, "end": lab.end, "duration_h": lab.duration_h,
            "magnitude": lab.magnitude, "magnitude_unit": lab.magnitude_unit,
            "detected": len(hit) > 0, "n_events": len(hit),
            "detected_types": ",".join(sorted(types)),
            "type_match": lab.anomaly_type in types,
            "delay_h": max(delay, 0.0) if len(hit) else np.nan,
        })
    return pd.DataFrame(rows), ev


def deviation_check(lab, g: pd.DataFrame, expected: pd.Series) -> dict:
    """Compare the injected excess (value - clean) with what the baseline measures (value - expected).

    recovery ~ 1 means the baseline deviation reflects the injected size.
    """
    key = lab.key
    w = (g.index >= lab.start) & (g.index <= lab.end)
    injected = (g[key] - g[f"{key}_clean"])[w].mean()
    measured = (g[key] - expected)[w].mean()
    recovery = measured / injected if injected else np.nan
    return {"injected_excess": round(float(injected), 3), "measured_excess": round(float(measured), 3),
            "recovery": round(float(recovery), 3)}


def summarize(cases: pd.DataFrame, ev: pd.DataFrame, entity_days: dict[str, float]) -> tuple[pd.DataFrame, dict]:
    per_type = cases.groupby(["key", "anomaly_type"]).agg(
        n_cases=("case_id", "size"),
        detected=("detected", "sum"),
        type_match=("type_match", "sum"),
        median_delay_h=("delay_h", "median"),
    )
    per_type["event_recall"] = (per_type["detected"] / per_type["n_cases"]).round(3)
    per_type["type_match_rate"] = (per_type["type_match"] / per_type["n_cases"]).round(3)

    scored = ev[ev["match"] != "natural"]
    by_det = scored.groupby("type")["match"].agg(
        n_events="size", true_positive=lambda m: (m == "true_positive").sum(),
    )
    by_det["event_precision"] = (by_det["true_positive"] / by_det["n_events"]).round(3)

    fp = scored[scored["match"] == "false_positive"]
    fp_per_key = fp.groupby("key").size()
    overall = {
        "n_injected_cases": int(len(cases)),
        "event_recall": round(float(cases["detected"].mean()), 3),
        "type_match_rate": round(float(cases["type_match"].mean()), 3),
        "n_events_scored": int(len(scored)),
        "n_events_natural": int((ev["match"] == "natural").sum()),
        "event_precision": round(float((scored["match"] == "true_positive").mean()), 3) if len(scored) else None,
        "event_precision_by_key": scored.groupby("key")["match"]
            .apply(lambda m: round(float((m == "true_positive").mean()), 3)).to_dict(),
        "false_alarms_per_device_day": {
            k: round(float(fp_per_key.get(k, 0) / d), 4) for k, d in entity_days.items()
        },
        "precision_by_detector_type": by_det.reset_index().to_dict("records"),
        "note": "Real series may contain real, unlabeled anomalies, so precision is a lower bound.",
    }
    return per_type.reset_index(), overall
