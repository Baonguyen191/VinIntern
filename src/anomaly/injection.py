"""Labeled anomaly case set built on the shared normalized telemetry (data_normalized).

Cases are only injected into keys that exist as real measurements in the shared set:
M1 `power_active_kw` and M2 `cooling_kw`. Keys the set does not have (CO2, water meters, chiller
power) get no cases, and meters whose series are mostly missing, zero or frozen are never selected.

Every injected case is marked is_synthetic=True / source="injected". Data-quality problems already
present in the real series (gaps, flatlines) are labeled source="natural" so evaluation does not
count them as false alarms.
"""
import numpy as np
import pandas as pd

from .hourly_baseline import is_night, is_operating
from .quality import detect_flatlines, detect_gaps

YEAR_START = pd.Timestamp("2017-01-01 00:00")
YEAR_END = pd.Timestamp("2017-12-31 23:00")
# Jan..mid-Mar stays clean so the 8-week rolling baseline warms up before the first case.
REFERENCE_END = pd.Timestamp("2017-03-15")
SLOT_DAYS = 21

# Entity quality gates (share of the 8760 hourly samples of 2017).
MIN_COVERAGE = 0.98
MAX_FLAT_SHARE = 0.02
MAX_ZERO_SHARE_M2 = 0.20

# (module, key) the plan is injected into.
SOURCES = {
    "M1": ("M1_power", "power_active_kw"),
    "M2": ("M2_cooling", "cooling_kw"),
}

# Chilled-water load is seasonal (near 0 in winter), so the M2 plan puts the cases that need
# a running chiller (stuck_sensor, off_hours_run) in the late-spring / summer slots.
CASE_PLANS = {
    "M1": ["spike", "off_hours_run", "baseload_rise", "data_loss",
           "stuck_sensor", "spike", "off_hours_run", "baseload_rise"],
    "M2": ["spike", "data_loss", "off_hours_run", "stuck_sensor", "spike", "off_hours_run"],
}

CASE_GROUP = {
    "spike": "A_consumption",
    "off_hours_run": "A_consumption",
    "baseload_rise": "A_consumption",
    "data_loss": "DQ_quality",
    "stuck_sensor": "DQ_quality",
}

LABEL_COLUMNS = [
    "case_id", "entity_id", "module", "key", "group", "anomaly_type", "start", "end",
    "duration_h", "magnitude", "magnitude_unit", "is_synthetic", "source", "note",
]


def to_hourly_grid(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """One entity -> full 2017 hourly grid. Missing rows become NaN (source drops them)."""
    grid = pd.date_range(YEAR_START, YEAR_END, freq="h", name="ts")
    out = df.set_index("ts")[value_cols].reindex(grid)
    known = df.assign(date=df["ts"].dt.normalize()).groupby("date")["day_type"].first().astype(str)
    day_type = pd.Series(grid.normalize().map(known), index=grid)
    fallback = pd.Series(np.where(grid.dayofweek < 5, "ngay_lam_viec", "cuoi_tuan"), index=grid)
    out["day_type"] = day_type.fillna(fallback)
    out["tod_slot"] = grid.hour
    return out


def entity_quality(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Per-entity coverage, zero share and share of repeated non-zero values (df sorted by entity, ts)."""
    v = df[key]
    same_entity = df["entity_id"].eq(df["entity_id"].shift())
    repeated = v.eq(v.shift()) & same_entity & v.ne(0)
    by = df["entity_id"]
    return pd.DataFrame({
        "coverage": by.value_counts() / 8760,
        "zero_share": v.eq(0).groupby(by).mean(),
        "flat_share": repeated.groupby(by).mean(),
        "p95": v.groupby(by).quantile(0.95),
    })


def _pick(candidates: list[str], n: int, rng: np.random.Generator, must_include: tuple[str, ...]) -> list[str]:
    chosen = [e for e in must_include if e in candidates]
    for e in must_include:
        if e not in candidates:
            print(f"  [skip] {e}: not eligible (coverage / zero / flatline / day-night ratio)")
    pool = [e for e in candidates if e not in chosen]
    chosen += [str(e) for e in rng.choice(pool, size=min(n - len(chosen), len(pool)), replace=False)]
    return sorted(chosen)


def select_m1_entities(
    m1: pd.DataFrame, n: int, rng: np.random.Generator, must_include: tuple[str, ...] = (),
) -> list[str]:
    """Office meters with near-full coverage, no frozen stretches and a clear day/night pattern."""
    office = m1[m1["primaryspaceusage"] == "Office"]
    hour = office["ts"].dt.hour
    stats = entity_quality(office, "power_active_kw")
    stats["day"] = office[(office["day_type"] == "ngay_lam_viec") & hour.between(9, 16)] \
        .groupby("entity_id")["power_active_kw"].median()
    stats["night"] = office[is_night(hour.to_numpy())].groupby("entity_id")["power_active_kw"].median()
    ok = stats[(stats["coverage"] >= MIN_COVERAGE) & (stats["flat_share"] < MAX_FLAT_SHARE)
               & (stats["night"] > 1) & (stats["day"] / stats["night"] >= 1.5)]
    return _pick(sorted(ok.index), n, rng, must_include)


def select_m2_entities(m2: pd.DataFrame, n: int, rng: np.random.Generator) -> list[str]:
    """Office chilled-water meters with near-full coverage, rarely off or frozen, and a summer
    day/night pattern (a plant that runs flat 24/7 has no off-hours regime for off_hours_run)."""
    office = m2[m2["primaryspaceusage"] == "Office"]
    hour = office["ts"].dt.hour
    summer = office["ts"].dt.month.between(5, 8)
    stats = entity_quality(office, "cooling_kw")
    stats["day"] = office[(office["day_type"] == "ngay_lam_viec") & hour.between(9, 16) & summer] \
        .groupby("entity_id")["cooling_kw"].median()
    stats["night"] = office[is_night(hour.to_numpy()) & summer].groupby("entity_id")["cooling_kw"].median()
    ok = stats[(stats["coverage"] >= MIN_COVERAGE) & (stats["flat_share"] < MAX_FLAT_SHARE)
               & (stats["zero_share"] < MAX_ZERO_SHARE_M2) & (stats["p95"] >= 50)
               & (stats["day"] >= 1.5 * stats["night"]) & (stats["day"] > 0)]
    return _pick(sorted(ok.index), n, rng, ())


def _label(case_id, entity, module, key, ctype, start, end, magnitude, unit, note,
           source="injected") -> dict:
    return {
        "case_id": case_id, "entity_id": entity, "module": module, "key": key,
        "group": CASE_GROUP[ctype], "anomaly_type": ctype, "start": start, "end": end,
        "duration_h": int((end - start) / pd.Timedelta(hours=1)) + 1,
        "magnitude": magnitude, "magnitude_unit": unit,
        "is_synthetic": source == "injected", "source": source, "note": note,
    }


def _draw_window(rng, index, valid, slot_start, dur_h, start_hour=None, tries=50):
    slot_h = SLOT_DAYS * 24
    for _ in range(tries):
        if start_hour is None:
            start = slot_start + pd.Timedelta(hours=int(rng.integers(0, slot_h - dur_h)))
        else:
            max_day = max(1, SLOT_DAYS - int(np.ceil((start_hour + dur_h) / 24)))
            start = slot_start + pd.Timedelta(days=int(rng.integers(0, max_day)), hours=start_hour)
        end = start + pd.Timedelta(hours=dur_h - 1)
        w = (index >= start) & (index <= end)
        if w.sum() == dur_h and valid[w].mean() >= 0.95:
            return start, end, w
    return None


def _inject(g, key, ctype, w, rng, ctx):
    """Apply one case in place; returns (magnitude, unit, note)."""
    if ctype == "spike":
        add = float(rng.uniform(1.0, 2.0) * ctx["p95"])
        g.loc[w, key] += add
        return round(add, 2), "kW_added", "added 1.0-2.0 x p95 of the meter"
    if ctype == "off_hours_run":
        level = float(rng.uniform(0.7, 1.0) * ctx["day_median"])
        # 3% noise: a running load is never perfectly constant (else it looks like a stuck sensor).
        running = level * (1 + rng.normal(0, 0.03, int(w.sum())))
        g.loc[w, key] = np.maximum(g.loc[w, key], running)
        return round(level, 2), "kW_level", "load held at 70-100% of working-hour median from 20:00"
    if ctype == "baseload_rise":
        add = float(rng.uniform(0.3, 0.6) * ctx["night_median"])
        g.loc[w, key] += add
        return round(add, 2), "kW_added", "constant 24/7 offset of 30-60% of the night base load"
    if ctype == "data_loss":
        g.loc[w, key] = np.nan
        return 0.0, "", "samples removed"
    if ctype == "stuck_sensor":
        frozen = float(g.loc[w, key].iloc[0])
        g.loc[w, key] = frozen
        return round(frozen, 2), "kW_frozen", "value frozen at first sample of the window"
    raise ValueError(ctype)


_DRAW = {
    "spike": lambda rng: (int(rng.integers(1, 3)), None),
    "off_hours_run": lambda rng: (int(rng.integers(8, 11)), 20),
    "baseload_rise": lambda rng: (int(rng.integers(72, 169)), None),
    "data_loss": lambda rng: (int(rng.integers(6, 49)), None),
    "stuck_sensor": lambda rng: (int(rng.integers(8, 25)), None),
}


def build_cases(
    grids: dict[str, pd.DataFrame], source: str, rng: np.random.Generator,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Inject CASE_PLANS[source] into each entity's real series; one case per 21-day slot."""
    module, key = SOURCES[source]
    series, labels = {}, []
    for ei, (entity, g) in enumerate(grids.items(), start=1):
        g = g.copy()
        clean = g[key].astype(float)
        g[f"{key}_clean"] = clean
        g[key] = clean.copy()
        g["case_id"] = ""

        natural = np.zeros(len(g), dtype=bool)
        nat_windows = [("data_loss", w) for w in detect_gaps(clean)] + \
                      [("stuck_sensor", w) for w in detect_flatlines(clean)]
        for k, (ctype, (s, e)) in enumerate(sorted(nat_windows, key=lambda x: x[1][0]), start=1):
            natural |= (g.index >= s) & (g.index <= e)
            labels.append(_label(f"NAT-{source}-{ei:02d}-{k:03d}", entity, module, key, ctype, s, e,
                                 np.nan, "", "present in the real series before injection",
                                 source="natural"))

        hour = g.index.hour.to_numpy()
        operating = is_operating(g["day_type"], hour) & (clean > 0).to_numpy()
        ctx = {
            "p95": float(clean.quantile(0.95)),
            "day_median": float(clean[operating].median()),
            "night_median": float(clean[is_night(hour)].median()),
        }
        # off_hours_run only raises samples below the injected level; if the real load is already
        # high the case would change nothing, so require a visible mean effect or redraw.
        min_off_hours_effect = 0.3 * (ctx["day_median"] - ctx["night_median"])

        for slot, ctype in enumerate(CASE_PLANS[source]):
            slot_start = REFERENCE_END + pd.Timedelta(days=slot * SLOT_DAYS)
            valid = clean.notna().to_numpy() & ~natural & (g["case_id"] == "").to_numpy()
            if ctype == "stuck_sensor":
                # A frozen 0 is indistinguishable from a switched-off meter.
                valid &= (clean != 0).to_numpy()
            injected = None
            for _ in range(10):
                dur_h, start_hour = _DRAW[ctype](rng)
                win = _draw_window(rng, g.index, valid, slot_start, dur_h, start_hour)
                if win is None:
                    break
                start, end, w = win
                before = g.loc[w, key].copy()
                result = _inject(g, key, ctype, w, rng, ctx)
                if ctype != "off_hours_run" or (g.loc[w, key] - before).mean() >= min_off_hours_effect:
                    injected = (start, end, w, *result)
                    break
                g.loc[w, key] = before
            if injected is None:
                print(f"  [skip] {entity} {ctype}: no usable window in slot {slot}")
                continue
            start, end, w, magnitude, unit, note = injected
            case_id = f"SYN-{source}-{ei:02d}-{slot + 1:02d}"
            g.loc[w, "case_id"] = case_id
            labels.append(_label(case_id, entity, module, key, ctype, start, end, magnitude, unit, note))

        series[entity] = g
    return series, pd.DataFrame(labels, columns=LABEL_COLUMNS)
