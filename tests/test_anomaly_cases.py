import numpy as np
import pandas as pd

from src.anomaly.cases import (
    SlotPicker,
    build_hourly_cases,
    inject_daily_baseline,
    inject_efficiency_drop,
    inject_missing,
    inject_spike,
    inject_stuck,
)
from src.anomaly.quality import find_flatline, find_gaps, find_out_of_range, mask_runs, run_quality


def _office_series(days: int = 120, seed: int = 0) -> tuple[pd.Series, pd.Series]:
    idx = pd.date_range("2017-01-01", periods=24 * days, freq="h")
    workday = idx.dayofweek < 5
    occupied = workday & (idx.hour >= 8) & (idx.hour < 19)
    rng = np.random.default_rng(seed)
    values = np.where(occupied, 100.0, 40.0) + rng.normal(0, 2, len(idx))
    day_type = pd.Series(np.where(workday, "ngay_lam_viec", "cuoi_tuan"), index=idx)
    return pd.Series(values, index=idx), day_type


def test_mask_runs():
    assert mask_runs(np.array([0, 1, 1, 0, 1], dtype=bool)) == [(1, 2), (4, 4)]
    assert mask_runs(np.zeros(3, dtype=bool)) == []


def test_quality_checks_find_injected_faults():
    s, _ = _office_series(days=10)
    v = inject_missing(s.to_numpy(), 50, 5)
    v = inject_stuck(v, 100, 6)
    v[150] = -1.0
    s = pd.Series(v, index=s.index)

    assert find_gaps(s) == [(50, 54)]
    assert find_flatline(s, min_len=3) == [(99, 105)]
    assert find_out_of_range(s) == [(150, 150)]
    types = set(run_quality(s, "b1", "power_active_kw")["type"])
    assert types == {"data_missing", "stuck_sensor", "out_of_range"}


def test_injectors():
    v = np.full(10, 10.0)
    assert np.allclose(inject_spike(v, 2, 2, 2.0, level=10.0)[2:4], 20.0)
    drop = inject_efficiency_drop(v, 0, ramp=4, hold=2, pct=0.1)
    assert np.isclose(drop[3], 11.0) and np.isclose(drop[5], 11.0) and drop[6] == 10.0
    assert v[2] == 10.0  # injectors do not mutate input


def test_slot_picker_no_overlap():
    rng = np.random.default_rng(0)
    picker = SlotPicker(np.ones(100, dtype=bool))
    starts = [picker.pick(rng, 5, guard=2) for _ in range(8)]
    windows = sorted((s, s + 5) for s in starts if s is not None)
    assert all(a[1] + 2 <= b[0] - 2 for a, b in zip(windows, windows[1:]))


def test_build_hourly_cases_labels_match_changes():
    s, day_type = _office_series()
    train_end = pd.Timestamp("2017-02-01")
    counts = {"spike": 2, "after_hours": 2, "efficiency_drop": 1, "stuck_sensor": 1, "data_missing": 1}
    injected, cases = build_hourly_cases(
        "b1", s, day_type, np.ones(len(s), dtype=bool), train_end, np.random.default_rng(1), counts
    )
    assert sorted(c["type"] for c in cases) == sorted(t for t, n in counts.items() for _ in range(n))

    changed = ~np.isclose(injected.to_numpy(), s.to_numpy(), equal_nan=False)
    labelled = np.zeros(len(s), dtype=bool)
    for c in cases:
        labelled[(s.index >= c["start"]) & (s.index <= c["end"])] = True
    assert not (changed & ~labelled).any()  # nothing modified outside a label
    assert not changed[s.index < train_end].any()  # train window untouched


def test_inject_daily_baseline_recomputes_residuals():
    dates = pd.date_range("2017-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "actual_kWh": 100.0,
        "pred_dow_median": 98.0,
        "residual_dow_median": 2.0,
        "anomaly_dow_median": False,
    })
    out, cases = inject_daily_baseline(
        df, "b1", np.random.default_rng(0), {"spike": 2, "level_shift": 1, "data_missing": 1}
    )
    assert "anomaly_dow_median" not in out.columns
    assert np.allclose(
        out["residual_dow_median"], out["actual_kWh"] - out["pred_dow_median"], equal_nan=True
    )
    assert len(cases) == 4
