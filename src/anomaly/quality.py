"""Data-quality layer: gaps, flatlines, out-of-range values, counter reversals.

Every check returns fault events with the same columns, so sensor faults can be
reported separately from operational anomalies and used to suppress them.
"""
import numpy as np
import pandas as pd

FREQ = "h"
QUALITY_COLUMNS = ["entity_id", "key", "type", "start", "end", "n_points", "detail"]


def regularize(s: pd.Series, index: pd.DatetimeIndex | None = None, freq: str = FREQ) -> pd.Series:
    """Reindex to a full regular grid; absent rows become NaN."""
    if index is None:
        index = pd.date_range(s.index.min(), s.index.max(), freq=freq)
    return s.reindex(index)


def mask_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return (first, last) positions of each run of True values, inclusive."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return []
    edges = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def find_gaps(s: pd.Series, min_len: int = 1) -> list[tuple[int, int]]:
    return [(a, b) for a, b in mask_runs(s.isna().to_numpy()) if b - a + 1 >= min_len]


def find_flatline(s: pd.Series, min_len: int = 3, tol: float = 1e-6) -> list[tuple[int, int]]:
    """Runs of at least `min_len` consecutive points with the same value."""
    v = s.to_numpy(dtype=float)
    same = np.zeros(len(v), dtype=bool)
    same[1:] = np.abs(np.diff(v)) <= tol  # NaN compares False, so gaps break runs
    # A run of k "same" flags starting at i covers points i-1 .. i+k-1.
    return [(a - 1, b) for a, b in mask_runs(same) if b - a + 2 >= min_len]


def find_out_of_range(
    s: pd.Series, lo: float | None = 0.0, hi: float | None = None
) -> list[tuple[int, int]]:
    v = s.to_numpy(dtype=float)
    mask = np.zeros(len(v), dtype=bool)
    if lo is not None:
        mask |= v < lo
    if hi is not None:
        mask |= v > hi
    return mask_runs(mask)


def find_counter_reverse(total: pd.Series) -> list[tuple[int, int]]:
    """Cumulative counter going backwards (reset or meter swap)."""
    d = total.diff().to_numpy(dtype=float)
    return mask_runs(np.nan_to_num(d, nan=0.0) < 0)


def run_quality(
    s: pd.Series,
    entity_id: str,
    key: str,
    min_gap: int = 1,
    min_flat: int = 3,
    lo: float | None = 0.0,
    hi: float | None = None,
    counter: pd.Series | None = None,
) -> pd.DataFrame:
    """Run every check on one regular series and return fault events."""
    idx = s.index
    checks = {
        "data_missing": find_gaps(s, min_gap),
        "stuck_sensor": find_flatline(s, min_flat),
        "out_of_range": find_out_of_range(s, lo, hi),
    }
    if counter is not None:
        checks["counter_reverse"] = find_counter_reverse(counter)

    rows = []
    for fault, runs in checks.items():
        for a, b in runs:
            detail = f"value={s.iloc[a]:.4g}" if fault == "stuck_sensor" else ""
            rows.append([entity_id, key, fault, idx[a], idx[b], b - a + 1, detail])
    out = pd.DataFrame(rows, columns=QUALITY_COLUMNS)
    return out.sort_values("start", ignore_index=True)


def fault_mask(s: pd.Series, events: pd.DataFrame) -> np.ndarray:
    """Boolean mask over `s` positions covered by any fault event."""
    mask = np.zeros(len(s), dtype=bool)
    pos = pd.Series(np.arange(len(s)), index=s.index)
    for start, end in zip(events["start"], events["end"]):
        mask[pos[start] : pos[end] + 1] = True
    return mask
