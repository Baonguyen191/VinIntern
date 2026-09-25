"""Labelled anomaly case set: inject known faults into clean series.

Injectors are pure functions on numpy arrays. `build_hourly_cases` places
non-overlapping cases per entity and returns the injected series with labels.
Every generated case has `is_synthetic=True`.
"""
import numpy as np
import pandas as pd

CASE_COLUMNS = [
    "case_id", "entity_id", "key", "type", "group",
    "start", "end", "duration_h", "magnitude", "detail", "is_synthetic",
]

CASE_GROUPS = {
    "spike": "A",
    "after_hours": "A",
    "level_shift": "A",
    "efficiency_drop": "B",
    "data_missing": "DQ",
    "stuck_sensor": "DQ",
}

WORKDAY = "ngay_lam_viec"
WEEKEND = "cuoi_tuan"
OCCUPIED_HOURS = range(9, 18)
NIGHT_HOURS = range(0, 6)


# ---------------------------------------------------------------- injectors

def inject_spike(values: np.ndarray, start: int, length: int, factor: float, level: float) -> np.ndarray:
    """Multiply by `factor`, with a floor of +0.5*level so low readings still spike."""
    out = values.copy()
    w = slice(start, start + length)
    out[w] = np.maximum(out[w] * factor, out[w] + 0.5 * level)
    return out


def inject_after_hours(
    values: np.ndarray, start: int, length: int, level: float, rng: np.random.Generator
) -> np.ndarray:
    """Raise off-hours load to the occupied level (equipment left running)."""
    out = values.copy()
    w = slice(start, start + length)
    out[w] = np.maximum(out[w], level * rng.uniform(0.9, 1.1, length))
    return out


def inject_efficiency_drop(
    values: np.ndarray, start: int, ramp: int, hold: int, pct: float
) -> np.ndarray:
    """Same output, more input: ramp load up by `pct` then hold."""
    out = values.copy()
    mult = np.concatenate([np.linspace(1.0, 1.0 + pct, ramp), np.full(hold, 1.0 + pct)])
    out[start : start + ramp + hold] *= mult
    return out


def inject_level_shift(values: np.ndarray, start: int, length: int, pct: float) -> np.ndarray:
    out = values.copy()
    out[start : start + length] *= 1.0 + pct
    return out


def inject_stuck(values: np.ndarray, start: int, length: int) -> np.ndarray:
    """Hold the last good reading."""
    out = values.copy()
    out[start : start + length] = values[start - 1]
    return out


def inject_missing(values: np.ndarray, start: int, length: int) -> np.ndarray:
    out = values.copy()
    out[start : start + length] = np.nan
    return out


# ---------------------------------------------------------------- placement

class SlotPicker:
    """Pick random non-overlapping windows inside the free positions."""

    def __init__(self, free: np.ndarray):
        self.free = np.asarray(free, dtype=bool).copy()

    def pick(
        self,
        rng: np.random.Generator,
        length: int,
        guard: int,
        allowed_starts: np.ndarray | None = None,
    ) -> int | None:
        n = len(self.free)
        blocked = np.concatenate([[0], np.cumsum(~self.free)])
        starts = np.arange(guard, n - length - guard + 1)
        if allowed_starts is not None:
            starts = starts[allowed_starts[starts]]
        if len(starts) == 0:
            return None
        ok = blocked[starts + length + guard] - blocked[starts - guard] == 0
        starts = starts[ok]
        if len(starts) == 0:
            return None
        s = int(rng.choice(starts))
        self.free[s - guard : s + length + guard] = False
        return s


# ---------------------------------------------------------------- hourly cases

def load_levels(values: np.ndarray, hours: np.ndarray, day_type: np.ndarray, train: np.ndarray) -> tuple[float, float]:
    """Median occupied (workday 9–17h) and night (workday 0–5h) load in the train window."""
    workday = train & (day_type == WORKDAY)
    occupied = np.nanmedian(values[workday & np.isin(hours, OCCUPIED_HOURS)])
    night = np.nanmedian(values[workday & np.isin(hours, NIGHT_HOURS)])
    return float(occupied), float(night)


def build_hourly_cases(
    entity_id: str,
    series: pd.Series,
    day_type: pd.Series,
    free: np.ndarray,
    train_end: pd.Timestamp,
    rng: np.random.Generator,
    counts: dict[str, int],
    key: str = "power_active_kw",
    min_after_hours_ratio: float = 1.3,
) -> tuple[pd.Series, list[dict]]:
    """Inject `counts[type]` cases into one regular hourly series.

    `free` marks positions safe to modify (no natural faults). Positions before
    `train_end` are never modified so detectors can learn a clean reference.
    """
    idx = series.index
    values = series.to_numpy(dtype=float)
    hours = idx.hour.to_numpy()
    dtype_arr = day_type.to_numpy()
    train = idx < train_end
    occupied, night = load_levels(values, hours, dtype_arr, train)

    picker = SlotPicker(free & ~train)
    cases = []

    def add(case_type: str, start: int, length: int, magnitude: float, detail: str = ""):
        cases.append({
            "entity_id": entity_id, "key": key, "type": case_type,
            "group": CASE_GROUPS[case_type],
            "start": idx[start], "end": idx[start + length - 1], "duration_h": length,
            "magnitude": magnitude, "detail": detail, "is_synthetic": True,
        })

    for _ in range(counts.get("efficiency_drop", 0)):
        ramp, hold = 24 * int(rng.integers(7, 15)), 24 * 14
        s = picker.pick(rng, ramp + hold, guard=48)
        if s is None:
            break
        pct = float(rng.uniform(0.05, 0.20))
        values = inject_efficiency_drop(values, s, ramp, hold, pct)
        add("efficiency_drop", s, ramp + hold, pct, f"ramp_days={ramp // 24}")

    if occupied >= min_after_hours_ratio * night:
        night_starts = (hours == 20) & (dtype_arr == WORKDAY)
        weekend_starts = (hours == 8) & (dtype_arr == WEEKEND)
        for _ in range(counts.get("after_hours", 0)):
            variant = "night" if rng.random() < 0.5 else "weekend"
            allowed, length = (night_starts, 10) if variant == "night" else (weekend_starts, 11)
            s = picker.pick(rng, length, guard=24, allowed_starts=allowed)
            if s is None:
                break
            before = np.nanmean(values[s : s + length])
            values = inject_after_hours(values, s, length, occupied, rng)
            add("after_hours", s, length, float(np.nanmean(values[s : s + length]) / before),
                f"variant={variant}")

    for _ in range(counts.get("spike", 0)):
        length = int(rng.integers(1, 4))
        s = picker.pick(rng, length, guard=24)
        if s is None:
            break
        factor = float(rng.uniform(1.5, 3.0))
        values = inject_spike(values, s, length, factor, occupied)
        add("spike", s, length, factor)

    for _ in range(counts.get("stuck_sensor", 0)):
        length = int(rng.integers(3, 25))
        s = picker.pick(rng, length, guard=24)
        if s is None:
            break
        values = inject_stuck(values, s, length)
        add("stuck_sensor", s, length, np.nan, f"value={values[s]:.4g}")

    for _ in range(counts.get("data_missing", 0)):
        length = int(rng.integers(2, 49))
        s = picker.pick(rng, length, guard=24)
        if s is None:
            break
        values = inject_missing(values, s, length)
        add("data_missing", s, length, np.nan)

    return pd.Series(values, index=idx, name=series.name), cases


# ---------------------------------------------------------------- daily baseline cases

def inject_daily_baseline(
    df: pd.DataFrame,
    entity_id: str,
    rng: np.random.Generator,
    counts: dict[str, int],
    actual_col: str = "actual_kWh",
) -> tuple[pd.DataFrame, list[dict]]:
    """Inject cases into a baseline CSV (date, actual, pred_*, residual_*).

    Residual columns are recomputed as actual - pred. The baseline's own
    `anomaly_*` flags are dropped because they no longer match the data.
    """
    df = df.sort_values("date", ignore_index=True)
    dates = pd.to_datetime(df["date"])
    values = df[actual_col].to_numpy(dtype=float)
    picker = SlotPicker(np.ones(len(df), dtype=bool))
    cases = []

    def add(case_type: str, start: int, length: int, magnitude: float):
        cases.append({
            "entity_id": entity_id, "key": actual_col, "type": case_type,
            "group": CASE_GROUPS[case_type],
            "start": dates[start], "end": dates[start + length - 1], "duration_h": 24 * length,
            "magnitude": magnitude, "detail": "daily", "is_synthetic": True,
        })

    for _ in range(counts.get("level_shift", 0)):
        length = int(rng.integers(7, 15))
        s = picker.pick(rng, length, guard=3)
        if s is None:
            break
        pct = float(rng.uniform(0.10, 0.20))
        values = inject_level_shift(values, s, length, pct)
        add("level_shift", s, length, pct)

    for _ in range(counts.get("spike", 0)):
        s = picker.pick(rng, 1, guard=3)
        if s is None:
            break
        factor = float(rng.uniform(1.3, 1.6))
        values[s] *= factor
        add("spike", s, 1, factor)

    for _ in range(counts.get("data_missing", 0)):
        length = int(rng.integers(1, 4))
        s = picker.pick(rng, length, guard=3)
        if s is None:
            break
        values = inject_missing(values, s, length)
        add("data_missing", s, length, np.nan)

    out = df.drop(columns=[c for c in df.columns if c.startswith("anomaly_")])
    out[actual_col] = values
    for pred_col in [c for c in out.columns if c.startswith("pred_")]:
        out["residual_" + pred_col.removeprefix("pred_")] = out[actual_col] - out[pred_col]
    return out, cases


def cases_frame(cases: list[dict], prefix: str = "C") -> pd.DataFrame:
    df = pd.DataFrame(cases, columns=[c for c in CASE_COLUMNS if c != "case_id"])
    df = df.sort_values(["entity_id", "start"], ignore_index=True)
    df.insert(0, "case_id", [f"{prefix}{i:04d}" for i in range(1, len(df) + 1)])
    return df
