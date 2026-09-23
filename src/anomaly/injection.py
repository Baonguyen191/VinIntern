"""Labeled anomaly case set built on the shared normalized telemetry (data/normalized).

Every injected case is marked is_synthetic=True / source="injected". Data-quality problems already
present in the real series (gaps, flatlines) are labeled source="natural" so evaluation does not
count them as false alarms.
"""
import numpy as np
import pandas as pd

from .hourly_baseline import NIGHT_HOURS, is_night, is_operating
from .quality import detect_flatlines, detect_gaps

YEAR_START = pd.Timestamp("2017-01-01 00:00")
YEAR_END = pd.Timestamp("2017-12-31 23:00")
# Jan..mid-Mar stays clean so the 8-week rolling baseline warms up before the first case.
REFERENCE_END = pd.Timestamp("2017-03-15")
SLOT_DAYS = 21

# Group B needs a FIXED reference window (problem.md 3.2): virtual maintenance on 2017-04-01, 8 weeks.
B_REFERENCE = (pd.Timestamp("2017-04-01"), pd.Timestamp("2017-05-27"))
B_DEGRADE_FROM = pd.Timestamp("2017-06-01")
B_RAMP_DAYS = 7
B_TOTAL_DAYS = 28

M1_CASE_PLAN = [
    "spike", "off_hours_run", "leak", "co2_sustained_high",
    "data_loss", "stuck_sensor", "spike", "off_hours_run",
]

CASE_META = {
    "spike": ("A_consumption", "M1_power", "power_active_kw"),
    "off_hours_run": ("A_consumption", "M1_power", "power_active_kw"),
    "leak": ("A_consumption", "M1_power", "power_active_kw"),
    "co2_sustained_high": ("A_level", "M4_air", "co2_ppm"),
    "data_loss": ("DQ_quality", "M1_power", "power_active_kw"),
    "stuck_sensor": ("DQ_quality", "M1_power", "power_active_kw"),
    "efficiency_degradation": ("B_efficiency", "M2_chiller", "kw_per_kw_cooling"),
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


def select_m1_entities(
    m1: pd.DataFrame, n: int, rng: np.random.Generator, must_include: tuple[str, ...] = (),
) -> list[str]:
    """Office meters with near-full coverage and a clear day/night pattern."""
    office = m1[m1["primaryspaceusage"] == "Office"]
    hour = office["ts"].dt.hour
    rows = office.groupby("entity_id").size()
    day = office[(office["day_type"] == "ngay_lam_viec") & hour.between(9, 16)] \
        .groupby("entity_id")["power_active_kw"].median()
    night = office[hour.between(*NIGHT_HOURS)].groupby("entity_id")["power_active_kw"].median()
    p = office["power_active_kw"]
    repeated = p.eq(p.shift()) & office["entity_id"].eq(office["entity_id"].shift()) & p.ne(0)
    stats = pd.DataFrame({
        "rows": rows, "day": day, "night": night,
        "flat_share": repeated.groupby(office["entity_id"]).mean(),
    })
    stats["ratio"] = stats["day"] / stats["night"]
    ok = stats[(stats["rows"] >= 0.98 * 8760) & (stats["night"] > 1) & (stats["ratio"] >= 1.5)
               & (stats["flat_share"] < 0.02)]
    candidates = sorted(ok.index)

    chosen = [e for e in must_include if e in candidates]
    for e in must_include:
        if e not in candidates:
            print(f"  [skip] {e}: not eligible (coverage / day-night ratio)")
    pool = [e for e in candidates if e not in chosen]
    chosen += [str(e) for e in rng.choice(pool, size=min(n - len(chosen), len(pool)), replace=False)]
    return sorted(chosen)


def select_m2_entities(m2: pd.DataFrame, n: int, rng: np.random.Generator) -> list[str]:
    """Office chilled-water meters that run in both the B reference window and summer."""
    office = m2[m2["primaryspaceusage"] == "Office"]
    p95 = office.groupby("entity_id")["cooling_kw"].quantile(0.95)
    running = office["cooling_kw"] > 0.15 * office["entity_id"].map(p95)
    ts = office["ts"]
    in_ref = (ts >= B_REFERENCE[0]) & (ts < B_REFERENCE[1])
    in_summer = (ts >= B_DEGRADE_FROM) & (ts < B_DEGRADE_FROM + pd.Timedelta(days=60))
    stats = pd.DataFrame({
        "rows": office.groupby("entity_id").size(),
        "p95": p95,
        "run_ref": (running & in_ref).groupby(office["entity_id"]).sum(),
        "run_summer": (running & in_summer).groupby(office["entity_id"]).sum(),
    })
    ok = stats[(stats["rows"] >= 0.95 * 8760) & (stats["p95"] >= 50)
               & (stats["run_ref"] >= 300) & (stats["run_summer"] >= 500)]
    pool = sorted(ok.index)
    return sorted(str(e) for e in rng.choice(pool, size=min(n, len(pool)), replace=False))


def synth_co2(power: pd.Series, rng: np.random.Generator) -> pd.Series:
    """Fully synthetic CO2 sensor: 420 ppm outdoor + occupancy proxy from the power profile.

    The shared data set has no CO2 key (M4 is out of MVP scope); this channel only exists so the
    detector for a sustained CO2 rise can be exercised.
    """
    p = power.interpolate(limit_direction="both")
    lo, hi = p.quantile(0.05), p.quantile(0.95)
    occ = ((p - lo) / (hi - lo)).clip(0, 1).rolling(2, min_periods=1).mean()
    return (420 + 550 * occ + rng.normal(0, 12, len(p))).round(1)


def build_virtual_chiller(cooling: pd.Series, equipment: pd.DataFrame) -> dict:
    q_rated = float(cooling.quantile(0.99) * 1.1)
    row = equipment.iloc[int((equipment["q_rated_kw"] - q_rated).abs().argmin())]
    return {
        "template_chiller_id": row["chiller_id"],
        "model_type": row["model_type"],
        "q_rated_kw": round(q_rated, 1),
        "cop_rated": float(row["cop_rated"]),
        "a": float(row["eir_fplr_a"]),
        "b": float(row["eir_fplr_b"]),
        "c": float(row["eir_fplr_c"]),
        "min_plr": float(row["min_plr"]),
    }


def synth_chiller_power(
    cooling: pd.Series, chiller: dict, rng: np.random.Generator, noise: float = 0.02,
) -> pd.Series:
    """P = Q_rated/COP * (a + b*PLR + c*PLR^2) from equipment_params.csv, 0 below min_plr."""
    plr = (cooling / chiller["q_rated_kw"]).clip(0, 1)
    eir = chiller["a"] + chiller["b"] * plr + chiller["c"] * plr ** 2
    p = chiller["q_rated_kw"] / chiller["cop_rated"] * eir * (1 + rng.normal(0, noise, len(plr)))
    return p.where((plr >= chiller["min_plr"]) | cooling.isna(), 0.0)


def kw_per_kw(power: pd.Series, cooling: pd.Series) -> pd.Series:
    return (power / cooling).where(power > 0)


def _label(case_id, entity, ctype, start, end, magnitude, unit, note, source="injected") -> dict:
    group, module, key = CASE_META[ctype]
    return {
        "case_id": case_id, "entity_id": entity, "module": module, "key": key,
        "group": group, "anomaly_type": ctype, "start": start, "end": end,
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


def _inject_m1(g, ctype, w, rng, ctx):
    """Apply one case in place; returns (magnitude, unit, note)."""
    if ctype == "spike":
        add = float(rng.uniform(1.0, 2.0) * ctx["p95"])
        g.loc[w, "power_active_kw"] += add
        return round(add, 2), "kW_added", "added 1.0-2.0 x p95 of the meter"
    if ctype == "off_hours_run":
        level = float(rng.uniform(0.7, 1.0) * ctx["day_median"])
        # 3% noise: a running load is never perfectly constant (else it looks like a stuck sensor).
        running = level * (1 + rng.normal(0, 0.03, int(w.sum())))
        g.loc[w, "power_active_kw"] = np.maximum(g.loc[w, "power_active_kw"], running)
        return round(level, 2), "kW_level", "load held at 70-100% of working-hour median from 20:00"
    if ctype == "leak":
        add = float(rng.uniform(0.3, 0.6) * ctx["night_median"])
        g.loc[w, "power_active_kw"] += add
        return round(add, 2), "kW_added", (
            "constant 24/7 offset (30-60% of night base). Surrogate for M3 water leak: "
            "the shared set has no water meter"
        )
    if ctype == "co2_sustained_high":
        add = float(rng.uniform(250, 450))
        g.loc[w, "co2_ppm"] += add
        return round(add, 1), "ppm_added", "synthetic CO2 channel, ventilation-failure style step"
    if ctype == "data_loss":
        g.loc[w, "power_active_kw"] = np.nan
        return 0.0, "", "samples removed"
    if ctype == "stuck_sensor":
        vals = g.loc[w, "power_active_kw"]
        frozen = float(vals[vals != 0].iloc[0])
        g.loc[w, "power_active_kw"] = frozen
        return round(frozen, 2), "kW_frozen", "value frozen at first sample of the window"
    raise ValueError(ctype)


_M1_DRAW = {
    "spike": lambda rng: (int(rng.integers(1, 3)), None),
    "off_hours_run": lambda rng: (int(rng.integers(8, 11)), 20),
    "leak": lambda rng: (int(rng.integers(72, 169)), None),
    "co2_sustained_high": lambda rng: (int(rng.integers(48, 73)), 8),
    "data_loss": lambda rng: (int(rng.integers(6, 49)), None),
    "stuck_sensor": lambda rng: (int(rng.integers(8, 25)), None),
}


def build_m1_cases(
    grids: dict[str, pd.DataFrame], rng: np.random.Generator,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    series, labels = {}, []
    for ei, (entity, g) in enumerate(grids.items(), start=1):
        g = g.copy()
        clean = g["power_active_kw"].astype(float)
        g["power_active_kw_clean"] = clean
        g["co2_ppm_clean"] = synth_co2(clean, rng)
        g["power_active_kw"] = clean.copy()
        g["co2_ppm"] = g["co2_ppm_clean"].copy()
        g["case_id"] = ""

        natural = np.zeros(len(g), dtype=bool)
        nat_windows = [("data_loss", w) for w in detect_gaps(clean)] + \
                      [("stuck_sensor", w) for w in detect_flatlines(clean)]
        for k, (ctype, (s, e)) in enumerate(sorted(nat_windows, key=lambda x: x[1][0]), start=1):
            natural |= (g.index >= s) & (g.index <= e)
            labels.append(_label(f"NAT-M1-{ei:02d}-{k:03d}", entity, ctype, s, e, np.nan, "",
                                 "present in the real series before injection", source="natural"))

        ref = g[g.index < REFERENCE_END]
        hour = ref.index.hour.to_numpy()
        ctx = {
            "p95": float(clean.quantile(0.95)),
            "day_median": float(ref.loc[is_operating(ref["day_type"], hour), "power_active_kw_clean"].median()),
            "night_median": float(ref.loc[is_night(hour), "power_active_kw_clean"].median()),
        }

        # off_hours_run only raises samples below the injected level; if the real load is already
        # high the case would change nothing, so require a visible mean effect or redraw.
        min_off_hours_effect = 0.3 * (ctx["day_median"] - ctx["night_median"])

        for slot, ctype in enumerate(M1_CASE_PLAN):
            slot_start = REFERENCE_END + pd.Timedelta(days=slot * SLOT_DAYS)
            key = CASE_META[ctype][2]
            valid = clean.notna().to_numpy() & ~natural & (g["case_id"] == "").to_numpy()
            injected = None
            for _ in range(10):
                dur_h, start_hour = _M1_DRAW[ctype](rng)
                win = _draw_window(rng, g.index, valid, slot_start, dur_h, start_hour)
                if win is None:
                    break
                start, end, w = win
                before = g.loc[w, key].copy()
                result = _inject_m1(g, ctype, w, rng, ctx)
                if ctype != "off_hours_run" or (g.loc[w, key] - before).mean() >= min_off_hours_effect:
                    injected = (start, end, w, *result)
                    break
                g.loc[w, key] = before
            if injected is None:
                print(f"  [skip] {entity} {ctype}: no usable window in slot {slot}")
                continue
            start, end, w, magnitude, unit, note = injected
            case_id = f"SYN-M1-{ei:02d}-{slot + 1:02d}"
            g.loc[w, "case_id"] = case_id
            labels.append(_label(case_id, entity, ctype, start, end, magnitude, unit, note))

        series[entity] = g
    return series, pd.DataFrame(labels, columns=LABEL_COLUMNS)


def build_m2_cases(
    grids: dict[str, pd.DataFrame], equipment: pd.DataFrame, rng: np.random.Generator,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, dict]]:
    series, labels, chillers = {}, [], {}
    for ei, (entity, g) in enumerate(grids.items(), start=1):
        g = g.copy()
        cooling = g["cooling_kw"].astype(float)
        chiller = build_virtual_chiller(cooling.dropna(), equipment)
        chillers[entity] = chiller

        clean = synth_chiller_power(cooling, chiller, rng)
        g["chiller_power_kw_clean"] = clean
        g["kw_per_kw_cooling_clean"] = kw_per_kw(clean, cooling)
        g["case_id"] = ""

        start = B_DEGRADE_FROM + pd.Timedelta(days=int(rng.integers(0, 21)))
        end = start + pd.Timedelta(days=B_TOTAL_DAYS) - pd.Timedelta(hours=1)
        w = (g.index >= start) & (g.index <= end)
        d = float(rng.uniform(0.12, 0.20))
        ramp = np.clip(((g.index[w] - start) / pd.Timedelta(days=B_RAMP_DAYS)).to_numpy(), 0, 1)
        power = clean.copy()
        power[w] = clean[w] * (1 + d * ramp)
        g["chiller_power_kw"] = power
        g["kw_per_kw_cooling"] = kw_per_kw(power, cooling)

        case_id = f"SYN-M2-{ei:02d}-01"
        g.loc[w, "case_id"] = case_id
        labels.append(_label(
            case_id, entity, "efficiency_degradation", start, end, round(d * 100, 2),
            "pct_kw_per_kw_increase",
            f"linear ramp {B_RAMP_DAYS}d then hold; chiller power is SYNTHETIC from equipment_params "
            f"{chiller['template_chiller_id']} curve; reference window "
            f"{B_REFERENCE[0].date()}..{B_REFERENCE[1].date()}",
        ))
        series[entity] = g
    return series, pd.DataFrame(labels, columns=LABEL_COLUMNS), chillers
