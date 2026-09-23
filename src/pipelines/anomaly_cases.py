"""Sprint 1: labeled anomaly case set + initial detector, on the shared normalized data set."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..anomaly import injection as inj
from ..anomaly.detectors import (
    EVENT_COLUMNS, band_events, efficiency_events, leak_events, quality_events,
)
from ..anomaly.evaluation import AMPLITUDE_TYPES, deviation_check, match_events, summarize
from ..anomaly.hourly_baseline import profile_band
from ..io.normalized_loader import NORMALIZED_DIR, load_equipment_params, load_telemetry

M1_SERIES_COLS = ["day_type", "tod_slot", "temperature", "power_active_kw", "power_active_kw_clean",
                  "co2_ppm", "co2_ppm_clean", "case_id"]
M2_SERIES_COLS = ["day_type", "tod_slot", "temperature", "cooling_kw", "chiller_power_kw",
                  "chiller_power_kw_clean", "kw_per_kw_cooling", "kw_per_kw_cooling_clean", "case_id"]


def _load_grids(module, data_dir, value_cols, select_fn):
    cols = ["entity_id", "ts", "day_type", "primaryspaceusage", *value_cols]
    df = load_telemetry(module, data_dir, columns=cols)
    entities = select_fn(df)
    return {e: inj.to_hourly_grid(df[df["entity_id"] == e], value_cols) for e in entities}


def _stack(series: dict[str, pd.DataFrame], cols: list[str]) -> pd.DataFrame:
    return pd.concat(
        [g[cols].reset_index().assign(entity_id=e) for e, g in series.items()], ignore_index=True,
    )[["entity_id", "ts", *cols]]


def _o1_rows(entity, key, g, band, layer, window) -> pd.DataFrame:
    out = band[["expected", "lower", "upper"]].copy()
    out["entity_id"], out["key"] = entity, key
    out["baseline_layer"], out["window"] = layer, window
    out["day_type"], out["tod_slot"] = g["day_type"], g["tod_slot"]
    return out.dropna(subset=["expected"]).reset_index()


def _to_o2(events: pd.DataFrame) -> list[dict]:
    evidence_cols = ["actual_mean", "expected_mean", "deviation_pct", "peak_z",
                     "excess_kwh", "duration_h", "reference_window"]
    out = []
    for r in events.to_dict("records"):
        evidence = {c: r[c] for c in evidence_cols if not (isinstance(r[c], float) and np.isnan(r[c]))}
        out.append({
            "event_id": r["event_id"], "group": r["group"], "module": r["module"],
            "entity_id": r["entity_id"], "key": r["key"], "type": r["type"],
            "severity": r["severity"], "rule_or_model": r["rule_or_model"],
            "start": r["start"].isoformat(), "end": r["end"].isoformat(),
            "detected_at": r["detected_at"].isoformat(),
            "evidence": evidence, "match": r["match"], "matched_case_id": r["matched_case_id"],
        })
    return out


def _plot_case(lab, g, band, events, out_dir: Path) -> None:
    key = lab.key
    pad = pd.Timedelta(days=14 if lab.anomaly_type == "efficiency_degradation" else 3)
    x0, x1 = lab.start - pad, lab.end + pad
    w = (g.index >= x0) & (g.index <= x1)
    x = g.index[w]
    events = events[(events["end"] >= x0) & (events["start"] <= x1)]

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.set_xlim(x0, x1)
    ax.axvspan(lab.start, lab.end + pd.Timedelta(hours=1), color="orange", alpha=0.2, label="injected case")
    for ev in events.itertuples():
        ax.axvspan(ev.start, ev.end + pd.Timedelta(hours=1), ymin=0, ymax=0.06, color="red", alpha=0.8)
    if band is not None:
        ax.fill_between(x, band["lower"][w], band["upper"][w], color="gray", alpha=0.2, label="baseline band")
        ax.plot(x, band["expected"][w], color="black", lw=1, label="expected")
    ax.plot(x, g[f"{key}_clean"][w], color="steelblue", lw=1, ls="--", alpha=0.7, label="clean (before injection)")
    ax.plot(x, g[key][w], color="crimson", lw=1.1, label="series with case")
    ax.plot([], [], color="red", lw=6, label="detected event")
    ax.set_title(f"{lab.case_id} | {lab.entity_id} | {lab.anomaly_type} | "
                 f"magnitude {lab.magnitude} {lab.magnitude_unit}")
    ax.set_ylabel(key)
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{lab.case_id}_{lab.anomaly_type}.png", dpi=110)
    plt.close(fig)


def _plot_recall(per_type: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(per_type["anomaly_type"], per_type["event_recall"], color="steelblue", label="event recall")
    ax.barh(per_type["anomaly_type"], per_type["type_match_rate"], color="orange", alpha=0.7,
            height=0.4, label="correct type")
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of injected cases")
    ax.set_title("Initial detector on labeled anomaly cases")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "recall_by_type.png", dpi=110)
    plt.close(fig)


def run_anomaly_cases_pipeline(
    data_dir: str | Path = NORMALIZED_DIR,
    output_dir: str | Path = "results/anomaly_cases",
    n_m1: int = 6,
    n_m2: int = 2,
    seed: int = 42,
    k: float = 3.0,
) -> dict:
    out = Path(output_dir)
    plot_dir = out / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    print("[1/5] Building labeled case set")
    m1_grids = _load_grids("M1", data_dir, ["power_active_kw", "temperature"],
                           lambda df: inj.select_m1_entities(df, n_m1, rng, must_include=("Rat_office_Colby",)))
    print(f"  M1 entities: {list(m1_grids)}")
    m1_series, m1_labels = inj.build_m1_cases(m1_grids, rng)

    m2_grids = _load_grids("M2", data_dir, ["cooling_kw", "temperature"],
                           lambda df: inj.select_m2_entities(df, n_m2, rng))
    print(f"  M2 entities: {list(m2_grids)}")
    m2_series, m2_labels, chillers = inj.build_m2_cases(m2_grids, load_equipment_params(data_dir), rng)
    labels = pd.concat([m1_labels, m2_labels], ignore_index=True)

    print("[2/5] Baseline + detection")
    events, o1, bands = [], [], {}
    for e, g in m1_series.items():
        for key, abs_floor, module in (
            ("power_active_kw", 0.01 * g["power_active_kw_clean"].quantile(0.95), "M1_power"),
            ("co2_ppm", 20.0, "M4_air"),
        ):
            band = profile_band(g, key, k=k, abs_floor=abs_floor)
            bands[(e, key)] = band
            o1.append(_o1_rows(e, key, g, band, "history", "rolling 8w"))
            events += band_events(g, key, band, e, module)
        events += leak_events(g, "power_active_kw", bands[(e, "power_active_kw")], e, "M1_power")
        events += quality_events(g, "power_active_kw", e, "M1_power")

    for e, g in m2_series.items():
        ev, expected = efficiency_events(g, e, chillers[e], inj.B_REFERENCE)
        events += ev
        band = pd.DataFrame({"expected": expected, "lower": np.nan, "upper": expected * 1.08})
        bands[(e, "kw_per_kw_cooling")] = band
        o1.append(_o1_rows(e, "kw_per_kw_cooling", g, band, "history",
                           f"fixed {inj.B_REFERENCE[0].date()}..{inj.B_REFERENCE[1].date()}"))

    events_df = pd.DataFrame(events, columns=EVENT_COLUMNS).sort_values(["entity_id", "start"])
    events_df["event_id"] = [f"EV-{i:05d}" for i in range(1, len(events_df) + 1)]
    print(f"  {len(events_df)} events")

    print("[3/5] Event-level evaluation")
    cases, events_df = match_events(labels, events_df)
    all_series = {**m1_series, **m2_series}
    checks = []
    for lab in labels[labels["source"] == "injected"].itertuples():
        if lab.anomaly_type in AMPLITUDE_TYPES or lab.anomaly_type == "efficiency_degradation":
            checks.append({"case_id": lab.case_id,
                           **deviation_check(lab, all_series[lab.entity_id], bands[(lab.entity_id, lab.key)]["expected"])})
    if checks:
        cases = cases.merge(pd.DataFrame(checks), on="case_id", how="left")
    entity_days = {"power_active_kw": len(m1_series) * 365.0, "co2_ppm": len(m1_series) * 365.0,
                   "kw_per_kw_cooling": len(m2_series) * 365.0}
    per_type, overall = summarize(cases, events_df, entity_days)
    overall["seed"], overall["k"] = seed, k

    print("[4/5] Writing outputs")
    labels.to_csv(out / "labels.csv", index=False)
    _stack(m1_series, M1_SERIES_COLS).to_csv(out / "series_M1.csv", index=False)
    _stack(m2_series, M2_SERIES_COLS).to_csv(out / "series_M2.csv", index=False)
    pd.concat(o1, ignore_index=True).round({"expected": 3, "lower": 3, "upper": 3}) \
        .to_csv(out / "baseline_O1.csv", index=False)
    events_df.to_csv(out / "events.csv", index=False)
    with open(out / "events_O2.json", "w", encoding="utf-8") as f:
        json.dump(_to_o2(events_df), f, indent=2, ensure_ascii=False)
    with open(out / "virtual_chillers.json", "w", encoding="utf-8") as f:
        json.dump(chillers, f, indent=2, ensure_ascii=False)
    cases.to_csv(out / "eval_cases.csv", index=False)
    per_type.to_csv(out / "eval_by_type.csv", index=False)
    with open(out / "eval_overall.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, default=int)

    print("[5/5] Plots")
    for lab in labels[labels["source"] == "injected"].itertuples():
        g = all_series[lab.entity_id]
        band = bands[(lab.entity_id, lab.key)]
        ev = events_df[(events_df["entity_id"] == lab.entity_id) & (events_df["key"] == lab.key)]
        _plot_case(lab, g, band if lab.anomaly_type != "efficiency_degradation" else band[["expected"]].assign(
            lower=np.nan, upper=np.nan), ev, plot_dir)
    _plot_recall(per_type, out)

    print("\nPer type:")
    print(per_type.to_string(index=False))
    print("\nOverall:")
    print(json.dumps({k_: v for k_, v in overall.items() if k_ != "precision_by_detector_type"}, indent=2))
    print(f"\nResults saved to {out}")
    return overall
