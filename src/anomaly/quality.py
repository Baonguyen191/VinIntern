import numpy as np
import pandas as pd


def find_runs(mask: np.ndarray, max_gap: int = 0) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of True runs; runs separated by <= max_gap False are merged."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > max_gap + 1)
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    ends = np.concatenate([idx[breaks], [idx[-1]]])
    return list(zip(starts.tolist(), ends.tolist()))


def detect_gaps(series: pd.Series, min_hours: int = 3) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Series must be on a regular hourly DatetimeIndex; NaN = missing sample."""
    runs = find_runs(series.isna().to_numpy())
    return [(series.index[a], series.index[b]) for a, b in runs if b - a + 1 >= min_hours]


def detect_flatlines(series: pd.Series, min_hours: int = 4) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Runs of identical non-zero values; zero is excluded because a real meter can sit at 0 when off."""
    v = series.to_numpy(dtype=float)
    same = np.zeros(len(v), dtype=bool)
    same[1:] = (v[1:] == v[:-1]) & (v[1:] != 0) & ~np.isnan(v[1:])
    out = []
    for a, b in find_runs(same):
        start = a - 1
        if b - start + 1 >= min_hours:
            out.append((series.index[start], series.index[b]))
    return out
