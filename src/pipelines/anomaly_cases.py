"""Build the labelled anomaly case set (sprint task: "Chuẩn bị bộ ca bất thường").

Outputs in `output_dir`:
Source: BDG2 cleaned electricity (Office buildings, 2016–2017) via src.io.bdg2_telemetry.
2016 is the clean training year; cases are injected into 2017 only.

- cases.csv                  labels for hourly cases injected into building electricity
- cases_series.parquet       entity_id, ts, day_type, value_clean, value (injected; NaN = missing)
- quality_natural.csv        data-quality faults already present in the clean series
- baseline_daily_cases.csv   baseline CSV with injected cases (same structure as the template)
- baseline_daily_labels.csv  labels for the daily cases
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.anomaly.cases import build_hourly_cases, cases_frame, inject_daily_baseline
from src.anomaly.quality import fault_mask, regularize, run_quality
from src.io.bdg2_telemetry import load_cooling, load_electricity

KEY = "power_active_kw"

HOURLY_COUNTS = {"spike": 2, "after_hours": 2, "efficiency_drop": 1, "stuck_sensor": 1, "data_missing": 1}
DAILY_COUNTS = {"spike": 4, "level_shift": 3, "data_missing": 2}


def _select_entities(m1: pd.DataFrame, m2_ids: set[str], min_completeness: float, priority: str | None = None) -> list[str]:
    """Office buildings with enough rows; buildings that also have M2 cooling data first."""
    stats = m1.groupby("entity_id", observed=True)[KEY].agg(["size", "median"])
    n_hours = m1["ts"].nunique()
    ok = stats[(stats["size"] >= min_completeness * n_hours) & (stats["median"] > 1.0)].index
    return sorted(ok, key=lambda e: (e != priority, e not in m2_ids, e))


def _day_type_grid(g: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """day_type on the full grid; hours missing from the data take their date's value.

    Dates with no data at all fall back to the calendar (weekend or workday).
    """
    by_date = g.assign(date=g["ts"].dt.normalize()).groupby("date")["day_type"].first().astype(str)
    day_type = pd.Series(index.normalize().map(by_date).to_numpy(), index=index)
    calendar = np.where(index.dayofweek >= 5, "cuoi_tuan", "ngay_lam_viec")
    return day_type.fillna(pd.Series(calendar, index=index))


def run_case_pipeline(
    output_dir: str = "results/anomaly_cases",
    n_entities: int = 20,
    seed: int = 42,
    train_end: str = "2017-01-01",
    min_completeness: float = 0.95,
    max_fault_frac: float = 0.05,
    baseline_csv: str | None = "results/bdg2/statistical/Rat_office_Colby/our_cleaned_predictions.csv",
    baseline_entity: str = "Rat_office_Colby",
) -> pd.DataFrame:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    train_end_ts = pd.Timestamp(train_end)

    m1 = load_electricity(usage="Office")[["entity_id", "ts", "value", "day_type"]].rename(columns={"value": KEY})
    m2_ids = set(load_cooling()["entity_id"].unique())
    grid = pd.date_range(m1["ts"].min(), m1["ts"].max(), freq="h")
    groups = dict(tuple(m1.groupby("entity_id", observed=True)))

    series_parts, natural_parts, all_cases = [], [], []
    used = 0
    for entity_id in _select_entities(m1, m2_ids, min_completeness, priority=baseline_entity):
        if used == n_entities:
            break
        g = groups[entity_id]
        s = regularize(g.set_index("ts")[KEY].astype(float), grid)
        natural = run_quality(s, entity_id, KEY)
        faults = fault_mask(s, natural)
        if faults.mean() > max_fault_frac:
            continue

        counts = dict(HOURLY_COUNTS)
        if entity_id not in m2_ids:
            counts["efficiency_drop"] = 0  # detector needs chilled-water data for kW/kW_cooling
        day_type = _day_type_grid(g, grid)
        injected, cases = build_hourly_cases(
            entity_id, s, day_type, ~faults, train_end_ts, rng, counts, key=KEY
        )
        series_parts.append(pd.DataFrame({
            "entity_id": entity_id, "ts": grid, "day_type": day_type.to_numpy(),
            "value_clean": s.to_numpy(), "value": injected.to_numpy(),
        }))
        natural_parts.append(natural)
        all_cases.extend(cases)
        used += 1

    cases = cases_frame(all_cases, prefix="H")
    series = pd.concat(series_parts, ignore_index=True)
    natural = pd.concat(natural_parts, ignore_index=True)
    cases.to_csv(out / "cases.csv", index=False)
    series.to_parquet(out / "cases_series.parquet", index=False)
    natural.to_csv(out / "quality_natural.csv", index=False)

    print(f"Hourly case set: {used} entities, {len(cases)} cases, train < {train_end}")
    print(cases.groupby(["group", "type"]).size().to_string())
    print(f"Natural quality faults in clean series: {len(natural)}")

    if baseline_csv:
        daily = pd.read_csv(baseline_csv)
        daily_out, daily_cases = inject_daily_baseline(daily, baseline_entity, rng, DAILY_COUNTS)
        daily_labels = cases_frame(daily_cases, prefix="D")
        daily_out.to_csv(out / "baseline_daily_cases.csv", index=False)
        daily_labels.to_csv(out / "baseline_daily_labels.csv", index=False)
        print(f"Daily baseline cases ({baseline_entity}): {len(daily_labels)}")
        print(daily_labels.groupby("type").size().to_string())

    print(f"Saved to {out}")
    return cases
