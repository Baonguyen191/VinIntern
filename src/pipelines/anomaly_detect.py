"""Anomaly detection on the labelled case set (sprint task: "Xây bộ phát hiện bất thường").

Layers, all run per entity on hourly data:
1. data quality (gaps, stuck, out of range) -> sensor-fault events, sent to ops
2. rules: rolling profile band (hour x day type)  -> spike / after_hours (group A)
   each alert has a scope: "point" (<= 2 h) or "interval" (longer, or a
   sustained stretch where the 6 h mean z crosses a lower threshold)
3. efficiency: daily load vs cooling regression   -> efficiency_drop (group B)
4. Isolation Forest on the same features           -> compared with 2 at the same budget
5. LightGBM regression baseline (residual z)       -> compared with 2 at the same budget
   plus two variants with the Isolation Forest score as an extra feature
   (same hour, and mean of the previous 24 hours)
6. score fusion: LightGBM z and Isolation Forest score, each normalised to the
   train distribution, combined by max or mean (IF stays out of the baseline)
Operational events overlapping a sensor fault are suppressed, then cut to the
alert budget. The baseline-deviation branch scores the daily baseline CSV.

Outputs in `output_dir`:
- alerts_o2.jsonl        final operational alerts (rules + efficiency), O2 format with evidence
- quality_o2.jsonl       sensor-fault events for the ops team
- events_all.csv         every detector's events after suppression and budget
- comparison.csv         event precision / recall / delay per detector and case type
- budget_sweep.csv       rules vs Isolation Forest vs LightGBM at equal alert budgets
- daily_deviation_events.csv  events from the baseline-deviation branch
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.anomaly.detectors import (
    deviation_scores,
    efficiency_scores,
    hourly_features,
    fuse_scores,
    if_score_features,
    isolation_forest_scores,
    lightgbm_scores,
    profile_band_scores,
)
from src.anomaly.evaluate import metrics_by_type
from src.anomaly.events import (
    apply_budget,
    classify_hourly,
    interval_events,
    overlaps,
    points_to_events,
    split_point_interval,
    suppress_sensor_faults,
    to_o2,
)
from src.anomaly.quality import regularize, run_quality
from src.io.bdg2_telemetry import load_cooling, load_electricity

KEY = "power_active_kw"
OPERATIONAL_GROUPS = ("A", "B")
SWEEP_BUDGETS = (0.02, 0.05, 0.1, 0.2)

ACTIONS = {
    "spike": "Kiểm tra thiết bị khởi động bất thường trong khung giờ này",
    "after_hours": "Kiểm tra lịch BMS và thiết bị để chạy ngoài giờ",
    "sustained_high": "Kiểm tra tải tăng kéo dài: thiết bị không tắt, đổi chế độ vận hành, sự kiện trong tòa nhà",
    "efficiency_drop": "Kiểm tra dàn ngưng, lượng gas và lịch bảo trì chiller",
    "level_shift": "Kiểm tra tải nền tăng: thiết bị mới hoặc thiết bị không tắt",
    "data_missing": "Kiểm tra kết nối gateway / công tơ",
    "stuck_sensor": "Kiểm tra cảm biến / công tơ bị treo giá trị",
    "out_of_range": "Kiểm tra cấu hình đơn vị và dải đo của công tơ",
}


def load_temperature(entities: list[str]) -> dict[str, pd.Series]:
    """Outdoor temperature per entity from BDG2 weather (LightGBM feature)."""
    m1 = load_electricity(usage="Office", buildings=entities)
    return {e: g.set_index("ts")["temperature"].astype(float) for e, g in m1.groupby("entity_id")}


def _load_cases(case_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    series = pd.read_parquet(case_dir / "cases_series.parquet")
    labels = pd.read_csv(case_dir / "cases.csv", parse_dates=["start", "end"])
    natural = pd.read_csv(case_dir / "quality_natural.csv", parse_dates=["start", "end"])
    return series, labels, natural


def _detect_entity(
    entity_id: str,
    g: pd.DataFrame,
    cooling: pd.Series | None,
    temperature: pd.Series,
    train_end: pd.Timestamp,
    k_rule: float,
    k_interval: float,
    interval_window: int,
    if_quantile: float,
    eff_threshold: float,
) -> dict[str, pd.DataFrame]:
    s = g.set_index("ts")["value"].astype(float)
    day_type = g.set_index("ts")["day_type"]
    test = s.index >= train_end

    quality = run_quality(s, entity_id, KEY)

    band = profile_band_scores(s, day_type)
    rules = points_to_events(band["score"].where(test), k_rule, entity_id, "profile_band")
    rules = suppress_sensor_faults(rules, quality)
    sustained = interval_events(
        band["score"].where(test), interval_window, k_interval, entity_id, "profile_band_sustained"
    )
    sustained = suppress_sensor_faults(sustained, quality)
    scoped = split_point_interval(rules, sustained)
    for ev in (rules, scoped):
        ev["type"] = classify_hourly(ev, day_type)
        ev["group"] = "A"

    feats = hourly_features(s, day_type, band)
    if_scores = isolation_forest_scores(feats, ~test)
    threshold = np.nanquantile(if_scores[~test], if_quantile)
    iforest = points_to_events(if_scores.where(test), threshold, entity_id, "isolation_forest")
    iforest = suppress_sensor_faults(iforest, quality)
    iforest["type"] = classify_hourly(iforest, day_type)
    iforest["group"] = "A"

    lgbm_variants, lgbm_plain = {}, None
    for name, extra in (
        ("lightgbm", None),
        ("lightgbm_if", if_score_features(if_scores, lagged=False)),
        ("lightgbm_if_lag", if_score_features(if_scores, lagged=True)),
    ):
        sc = lightgbm_scores(s, day_type, temperature, band["baseline"], ~test, extra_features=extra)
        if extra is None:
            lgbm_plain = sc["score"]
        lgbm_variants[name] = sc["score"]

    for how in ("max", "mean"):
        lgbm_variants[f"fusion_{how}"] = fuse_scores(
            {"lightgbm": lgbm_plain, "iforest": if_scores}, ~test, how=how
        )
    for name, score in list(lgbm_variants.items()):
        ev = points_to_events(score.where(test), k_rule, entity_id, name)
        ev = suppress_sensor_faults(ev, quality)
        ev["type"] = classify_hourly(ev, day_type)
        ev["group"] = "A"
        lgbm_variants[name] = ev

    eff = pd.DataFrame()
    eff_scores = None
    if cooling is not None:
        eff_scores = efficiency_scores(s, regularize(cooling, s.index))
        daily_score = eff_scores["score"].where(eff_scores.index >= train_end)
        eff = points_to_events(
            daily_score, eff_threshold, entity_id, "load_vs_cooling_regression",
            max_gap=2, min_duration=3, step_h=24,
        )
        eff = suppress_sensor_faults(eff, quality)
        eff["type"] = "efficiency_drop"
        eff["group"] = "B"

    return {
        "quality": quality, "rules": rules, "rules_scoped": scoped, "iforest": iforest, **lgbm_variants, "efficiency": eff,
        "band": band, "eff_scores": eff_scores,
    }


def _o2_records(events: pd.DataFrame, bands: dict, effs: dict) -> list[dict]:
    records = []
    for i, ev in events.reset_index(drop=True).iterrows():
        eid = f"A-{ev['start']:%Y%m%d}-{i + 1:04d}"
        if ev["group"] == "B":
            sc = effs[ev["entity_id"]].loc[ev["peak_ts"]]
            records.append(to_o2(
                ev, eid, "B_efficiency", "M2_chiller", "kwh_per_day_vs_cooling",
                sc["value"], sc["baseline"], "hồi quy kWh~cooling, ngày t-49..t-21",
                severity="info", suggested_action=ACTIONS["efficiency_drop"],
            ))
        else:
            sc = bands[ev["entity_id"]].loc[ev["peak_ts"]]
            records.append(to_o2(
                ev, eid, "A_consumption", "M1_electricity", KEY,
                sc["value"], sc["baseline"], "median 4 tuần gần nhất, cùng giờ và loại ngày",
                suggested_action=ACTIONS[ev["type"]],
            ))
    return records


def _quality_records(quality: pd.DataFrame) -> list[dict]:
    return [{
        "event_id": f"DQ-{r.start:%Y%m%d}-{i + 1:04d}",
        "group": "DQ",
        "module": "M1_electricity",
        "entity_id": r.entity_id,
        "type": r.type,
        "severity": "info",
        "rule_or_model": "data_quality",
        "start": r.start.isoformat(),
        "end": r.end.isoformat(),
        "evidence": {"key": r.key, "n_points": int(r.n_points), "detail": r.detail},
        "suggested_action": ACTIONS[r.type],
    } for i, r in enumerate(quality.itertuples())]


def _daily_branch(case_dir: Path, z_threshold: float, cusum_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(case_dir / "baseline_daily_cases.csv")
    labels = pd.read_csv(case_dir / "baseline_daily_labels.csv", parse_dates=["start", "end"])
    entity_id = labels["entity_id"].iloc[0]
    sc = deviation_scores(daily)

    spikes = points_to_events(sc["score"], z_threshold, entity_id, "residual_robust_z", step_h=24)
    shifts = points_to_events(sc["cusum"], cusum_threshold, entity_id, "residual_cusum", step_h=24)
    gaps = run_quality(regularize(sc["value"], freq="D"), entity_id, "actual_kWh", min_flat=10**9)
    gaps = gaps.rename(columns={"n_points": "duration_h"}).assign(
        detector="data_quality", duration_h=lambda d: d["duration_h"] * 24
    )
    events = pd.concat([spikes, shifts], ignore_index=True)
    events = suppress_sensor_faults(events, gaps)
    events = pd.concat([events, gaps[["entity_id", "detector", "start", "end", "duration_h"]]], ignore_index=True)

    comparison = pd.concat([
        metrics_by_type(spikes, labels[labels["type"] == "spike"], "daily_residual_z"),
        metrics_by_type(shifts, labels[labels["type"] == "level_shift"], "daily_residual_cusum"),
        metrics_by_type(gaps, labels[labels["type"] == "data_missing"], "daily_data_quality"),
        metrics_by_type(events, labels, "daily_all"),
    ], ignore_index=True)
    return events, comparison


def run_detection_pipeline(
    case_dir: str = "results/anomaly_cases",
    output_dir: str = "results/anomaly_detection",
    train_end: str = "2017-01-01",
    budget_per_day: float = 0.2,
    k_rule: float = 4.0,
    k_interval: float = 2.5,
    interval_window: int = 6,
    if_quantile: float = 0.99,
    eff_threshold: float = 0.08,
    daily_z: float = 3.0,
    daily_cusum: float = 5.0,
) -> pd.DataFrame:
    case_path, out = Path(case_dir), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_end_ts = pd.Timestamp(train_end)

    series, labels, natural = _load_cases(case_path)
    entities = series["entity_id"].unique()
    m2 = load_cooling(buildings=list(entities))
    cooling = {e: g.set_index("ts")["value"].astype(float) for e, g in m2.groupby("entity_id")}
    temps = load_temperature(list(entities))

    parts: dict[str, list[pd.DataFrame]] = {
        "quality": [], "rules": [], "rules_scoped": [], "iforest": [], "lightgbm": [], "lightgbm_if": [], "lightgbm_if_lag": [],
        "fusion_max": [], "fusion_mean": [], "efficiency": [],
    }
    bands, effs = {}, {}
    for entity_id, g in series.groupby("entity_id"):
        res = _detect_entity(
            entity_id, g, cooling.get(entity_id), temps[entity_id], train_end_ts,
            k_rule, k_interval, interval_window, if_quantile, eff_threshold,
        )
        for name in parts:
            parts[name].append(res[name])
        bands[entity_id] = res["band"]
        if res["eff_scores"] is not None:
            effs[entity_id] = res["eff_scores"]

    found = {name: pd.concat(p, ignore_index=True) for name, p in parts.items()}
    test_days = (series["ts"].max() - train_end_ts).days + 1

    rules = apply_budget(found["rules"], test_days, budget_per_day)
    rules_scoped = apply_budget(found["rules_scoped"], test_days, budget_per_day)
    iforest = apply_budget(found["iforest"], test_days, budget_per_day)
    lgbm = apply_budget(found["lightgbm"], test_days, budget_per_day)
    lgbm_if = apply_budget(found["lightgbm_if"], test_days, budget_per_day)
    lgbm_if_lag = apply_budget(found["lightgbm_if_lag"], test_days, budget_per_day)
    eff = found["efficiency"]
    eff = eff.assign(scope="interval")
    alerts = apply_budget(pd.concat([rules_scoped, eff], ignore_index=True), test_days, budget_per_day)

    op_labels = labels[labels["group"].isin(OPERATIONAL_GROUPS)]
    a_labels = op_labels[op_labels["group"] == "A"]
    dq_labels = labels[labels["group"] == "DQ"]
    quality = found["quality"].rename(columns={"n_points": "duration_h"})
    quality["detector"] = "data_quality"
    # Faults already in the clean series are real faults, so they count as correct detections.
    dq_truth = pd.concat([dq_labels, natural.assign(type="natural_" + natural["type"])], ignore_index=True)

    comparison = pd.concat([
        metrics_by_type(rules, a_labels, "rules_profile_band"),
        metrics_by_type(rules_scoped, a_labels, "rules_point_and_interval"),
        metrics_by_type(rules_scoped[rules_scoped["scope"] == "point"], a_labels, "rules_scope_point"),
        metrics_by_type(rules_scoped[rules_scoped["scope"] == "interval"], a_labels, "rules_scope_interval"),
        metrics_by_type(iforest, a_labels, "isolation_forest"),
        metrics_by_type(lgbm, a_labels, "lightgbm_residual"),
        metrics_by_type(lgbm_if, a_labels, "lightgbm_if_score"),
        metrics_by_type(lgbm_if_lag, a_labels, "lightgbm_if_score_lag24h"),
        metrics_by_type(apply_budget(found["fusion_max"], test_days, budget_per_day), a_labels, "fusion_max_lgbm_if"),
        metrics_by_type(apply_budget(found["fusion_mean"], test_days, budget_per_day), a_labels, "fusion_mean_lgbm_if"),
        metrics_by_type(eff, op_labels[op_labels["type"] == "efficiency_drop"], "efficiency_regression"),
        metrics_by_type(alerts, op_labels, "final_alerts"),
        metrics_by_type(quality, dq_truth, "data_quality"),
    ], ignore_index=True)

    # Group A detectors at equal alert budgets (ranking by peak score).
    candidates = (
        ("rules_profile_band", found["rules"]),
        ("isolation_forest", found["iforest"]),
        ("lightgbm_residual", found["lightgbm"]),
        ("lightgbm_if_score", found["lightgbm_if"]),
        ("lightgbm_if_score_lag24h", found["lightgbm_if_lag"]),
        ("fusion_max_lgbm_if", found["fusion_max"]),
        ("fusion_mean_lgbm_if", found["fusion_mean"]),
    )
    sweep = []
    for b in SWEEP_BUDGETS:
        for name, det in candidates:
            m = metrics_by_type(apply_budget(det, test_days, b), a_labels, name).iloc[0]
            sweep.append({"budget_per_day": b, **m.drop("type").to_dict()})
    sweep = pd.DataFrame(sweep)

    daily_events, daily_cmp = _daily_branch(case_path, daily_z, daily_cusum)
    comparison = pd.concat([comparison, daily_cmp], ignore_index=True)

    with open(out / "alerts_o2.jsonl", "w", encoding="utf-8") as f:
        for rec in _o2_records(alerts, bands, effs):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(out / "quality_o2.jsonl", "w", encoding="utf-8") as f:
        for rec in _quality_records(found["quality"]):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    all_events = pd.concat([rules_scoped, iforest, lgbm, lgbm_if, lgbm_if_lag, eff], ignore_index=True)
    all_events["matches_label"] = overlaps(all_events, op_labels)
    all_events.to_csv(out / "events_all.csv", index=False)
    daily_events.to_csv(out / "daily_deviation_events.csv", index=False)
    comparison.to_csv(out / "comparison.csv", index=False)
    sweep.to_csv(out / "budget_sweep.csv", index=False)

    n_dev_days = len(entities) * test_days
    print(f"Entities: {len(entities)}, test days: {test_days}, budget: {budget_per_day}/entity/day")
    print(f"Final alerts: {len(alerts)} ({len(alerts) / n_dev_days:.3f}/entity/day), "
          f"sensor faults: {len(found['quality'])}")
    with pd.option_context("display.width", 140, "display.float_format", "{:.3f}".format):
        print(comparison.to_string(index=False))
        print("\nGroup A detectors at equal budget:")
        print(sweep.to_string(index=False))
    print(f"Saved to {out}")
    return comparison
