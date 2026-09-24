"""Sprint 2: anomaly detector — agreed rules vs Isolation Forest vs LightGBM baseline.

All three suites run on the same labeled case set (src/pipelines/anomaly_cases.build_case_set)
and share the same data-quality layer and alert post-processing:

  detections -> separate_sensor_faults (drop A alarms sitting on DQ windows) -> merge_alerts

The detector only needs the normalized telemetry; it does not use any forecast model.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..anomaly import injection as inj
from ..anomaly.detectors import (
    EVENT_COLUMNS, band_events, baseload_events, quality_events, run_events,
)
from ..anomaly.evaluation import match_events, summarize
from ..anomaly.hourly_baseline import profile_band
from ..anomaly.ml_detectors import (
    CROSS_FIT_REFERENCE, dq_mask, iforest_scores, lgbm_band, lgbm_baseline,
)
from ..anomaly.postprocess import merge_alerts, separate_sensor_faults
from ..io.normalized_loader import NORMALIZED_DIR
from .anomaly_cases import _o1_rows, _to_o2, build_case_set

# Threshold sweeps; the first value of each is not special, the operating point is chosen below.
K_SWEEP = [2.5, 3.0, 3.5, 4.0, 5.0]
IF_SWEEP = [0.9, 0.95, 0.98, 0.99, 0.995, 0.998]
# A short IF run is kept as a spike only above a 10x stricter quantile.
IF_SPIKE_FACTOR = 10
FA_BUDGET = 0.2  # false alarms / device / day, problem.md 10
RULES_AGREED_K = 3.0  # the agreed rule threshold (anomaly-cases); rules are not re-tuned
SUITES = {"rules": K_SWEEP, "lightgbm": K_SWEEP, "iforest": IF_SWEEP, "iforest_ctx": IF_SWEEP}
SUITE_PREFIX = {"rules": "RU", "lightgbm": "LG", "iforest": "IF", "iforest_ctx": "IC"}
COLORS = {"rules": "black", "lightgbm": "tab:green", "iforest": "tab:purple", "iforest_ctx": "tab:orange"}


def _with_k(band: pd.DataFrame, k: float) -> pd.DataFrame:
    return band.assign(lower=band["expected"] - k * band["scale"], upper=band["expected"] + k * band["scale"])


def _spike_q(q: float) -> float:
    return round(1 - (1 - q) / IF_SPIKE_FACTOR, 6)


def _prepare(series: dict, seed: int) -> list[dict]:
    """Fit every baseline / model once per meter; thresholds are applied later."""
    qs = sorted({*IF_SWEEP, *(_spike_q(q) for q in IF_SWEEP)})
    units = []
    for source, source_series in series.items():
        module, key = inj.SOURCES[source]
        for e, g in source_series.items():
            print(f"  {source} {e}")
            p95 = g[key].quantile(0.95)
            units.append({
                "entity": e, "key": key, "module": module, "g": g,
                "baseload": "baseload_rise" in inj.CASE_PLANS[source],
                "dq_events": quality_events(g, key, e, module),
                # Rule band keeps the agreed floor (1% of the clean p95), as in anomaly_cases.
                "rule_band": profile_band(g, key, k=1.0, abs_floor=0.01 * g[f"{key}_clean"].quantile(0.95)),
                "lgbm_band": lgbm_band(lgbm_baseline(g, key, seed=seed), k=1.0, abs_floor=0.01 * p95),
                "iforest": iforest_scores(g, key, qs, seed=seed, context=False),
                "iforest_ctx": iforest_scores(g, key, qs, seed=seed, context=True),
                "dq": dq_mask(g[key]),
            })
    return units


def _detect(u: dict, suite: str, param: float) -> list[dict]:
    g, key, e, module = u["g"], u["key"], u["entity"], u["module"]
    events = list(u["dq_events"])
    if suite == "rules":
        band = _with_k(u["rule_band"], param)
        events += band_events(g, key, band, e, module)
        if u["baseload"]:
            events += baseload_events(g, key, band, e, module)
    elif suite == "lightgbm":
        band = _with_k(u["lgbm_band"], param)
        events += band_events(g, key, band, e, module, rule="lgbm_quantile_band",
                              reference=CROSS_FIT_REFERENCE)
        if u["baseload"]:
            events += baseload_events(g, key, band, e, module, rule="lgbm_cusum_night_residual",
                                      reference=CROSS_FIT_REFERENCE)
    elif suite in ("iforest", "iforest_ctx"):
        s = u[suite]
        # Only upward deviations, like the other suites: the case set has no "too low" type.
        flagged = ((s["score"] > s[f"thr_{param}"]) & (g[key] > s["expected"])).to_numpy() & ~u["dq"]
        events += run_events(g, key, flagged, s["score"], s[f"thr_{_spike_q(param)}"], s["expected"],
                             e, module, suite, CROSS_FIT_REFERENCE,
                             long_h=24 if u["baseload"] else None, strength_is_z=False)
    return events


def _run_suite(units, labels, entity_days, suite, param):
    raw = pd.DataFrame([ev for u in units for ev in _detect(u, suite, param)], columns=EVENT_COLUMNS)
    kept, n_suppressed = separate_sensor_faults(raw)
    alerts = merge_alerts(kept)
    alerts["event_id"] = [f"{SUITE_PREFIX[suite]}-{i:05d}" for i in range(1, len(alerts) + 1)]
    cases, alerts = match_events(labels, alerts)
    per_type, overall = summarize(cases, alerts, entity_days)
    group_a = cases[cases["group"] == "A_consumption"]
    row = {
        "suite": suite, "threshold": param,
        "n_raw_events": len(raw), "n_suppressed_as_sensor_fault": n_suppressed, "n_alerts": len(alerts),
        "event_recall": overall["event_recall"],
        "event_recall_group_A": round(float(group_a["detected"].mean()), 3),
        "type_match_rate": overall["type_match_rate"],
        "median_delay_h_group_A": float(group_a["delay_h"].median()),
        "event_precision": overall["event_precision"],
        **{f"precision_{k}": v for k, v in overall["event_precision_by_key"].items()},
        **{f"fa_per_device_day_{k}": v for k, v in overall["false_alarms_per_device_day"].items()},
    }
    return row, cases, alerts, per_type


def _operating_point(sweep: pd.DataFrame, suite: str, fa_cols: list[str]) -> float:
    """Model threshold: highest group-A recall whose false alarms stay within FA_BUDGET on every key,
    ties broken by precision. The rule suite stays at the agreed k (RULES_AGREED_K)."""
    s = sweep[sweep["suite"] == suite]
    ok = s[(s[fa_cols] <= FA_BUDGET).all(axis=1)]
    pick = (ok if len(ok) else s).sort_values(["event_recall_group_A", "event_precision"], ascending=False)
    return float(pick.iloc[0]["threshold"])


def _plot_tradeoff(sweep: pd.DataFrame, keys: list[str], out: Path) -> None:
    fig, axes = plt.subplots(1, len(keys), figsize=(6 * len(keys), 4.5), squeeze=False)
    for ax, key in zip(axes[0], keys):
        for suite, s in sweep.groupby("suite"):
            ax.plot(s[f"fa_per_device_day_{key}"], s["event_recall_group_A"], "o-", color=COLORS[suite], label=suite)
            for r in s.itertuples():
                ax.annotate(str(r.threshold), (getattr(r, f"fa_per_device_day_{key}"), r.event_recall_group_A),
                            fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.axvline(FA_BUDGET, color="red", ls="--", lw=1, label="FA budget")
        ax.set_xlabel(f"false alarms / device / day ({key})")
        ax.set_ylabel("event recall, group A cases (all keys)")
        ax.set_title(key)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "tradeoff_recall_vs_false_alarms.png", dpi=110)
    plt.close(fig)


def _plot_by_type(by_type: pd.DataFrame, out: Path) -> None:
    piv = by_type.assign(name=by_type["key"] + " / " + by_type["anomaly_type"]) \
        .pivot(index="name", columns="suite", values="event_recall")
    ax = piv.plot.barh(figsize=(9, 5), color=COLORS)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("event recall at operating point")
    ax.set_title("Recall by case type")
    plt.tight_layout()
    plt.savefig(out / "recall_by_type_by_suite.png", dpi=110)
    plt.close()


def run_anomaly_detection_pipeline(
    data_dir: str | Path = NORMALIZED_DIR,
    output_dir: str | Path = "results/anomaly_detection",
    seed: int = 42,
) -> pd.DataFrame:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[1/4] Labeled case set (same as anomaly-cases)")
    series, labels = build_case_set(data_dir, seed=seed)
    entity_days = {inj.SOURCES[s][1]: len(ss) * 365.0 for s, ss in series.items()}
    keys = list(entity_days)
    fa_cols = [f"fa_per_device_day_{k}" for k in keys]

    print("[2/4] Fitting baselines / models (month cross-fit)")
    units = _prepare(series, seed)

    print("[3/4] Threshold sweep")
    rows = []
    for suite, params in SUITES.items():
        for p in params:
            rows.append(_run_suite(units, labels, entity_days, suite, p)[0])
    sweep = pd.DataFrame(rows)
    sweep.to_csv(out / "comparison_sweep.csv", index=False)

    print("[4/4] Operating points + alert lists")
    summary, by_type = [], []
    for suite in SUITES:
        p = RULES_AGREED_K if suite == "rules" else _operating_point(sweep, suite, fa_cols)
        row, cases, alerts, per_type = _run_suite(units, labels, entity_days, suite, p)
        summary.append(row)
        by_type.append(per_type.assign(suite=suite, threshold=p))
        cases.to_csv(out / f"eval_cases_{suite}.csv", index=False)
        alerts.to_csv(out / f"alerts_{suite}.csv", index=False)
        with open(out / f"alerts_{suite}_O2.json", "w", encoding="utf-8") as f:
            json.dump(_to_o2(alerts), f, indent=2, ensure_ascii=False, default=str)
    summary = pd.DataFrame(summary)
    by_type = pd.concat(by_type, ignore_index=True)
    summary.to_csv(out / "comparison_summary.csv", index=False)
    by_type.to_csv(out / "comparison_by_type.csv", index=False)

    k_lgbm = float(summary.loc[summary["suite"] == "lightgbm", "threshold"].iloc[0])
    pd.concat([_o1_rows(u["entity"], u["key"], u["g"], _with_k(u["lgbm_band"], k_lgbm),
                        "model_lightgbm", CROSS_FIT_REFERENCE) for u in units], ignore_index=True) \
        .round({"expected": 3, "lower": 3, "upper": 3}).to_csv(out / "baseline_lgbm_O1.csv", index=False)

    _plot_tradeoff(sweep, keys, out)
    _plot_by_type(by_type, out)

    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print("\nSweep:")
        print(sweep.to_string(index=False))
        print("\nOperating points:")
        print(summary.to_string(index=False))
        print("\nRecall by type:")
        print(by_type.pivot_table(index=["key", "anomaly_type"], columns="suite",
                                  values=["event_recall", "type_match_rate"]).to_string())
    print(f"\nResults saved to {out}")
    return summary
