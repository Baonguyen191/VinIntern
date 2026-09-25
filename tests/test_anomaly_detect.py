import numpy as np
import pandas as pd

from src.anomaly.detectors import cusum, deviation_scores, efficiency_scores, profile_band_scores
from src.anomaly.evaluate import event_metrics
from src.anomaly.events import apply_budget, classify_hourly, points_to_events, suppress_sensor_faults


def _office(days: int = 60):
    idx = pd.date_range("2017-01-02", periods=24 * days, freq="h")
    workday = idx.dayofweek < 5
    occupied = workday & (idx.hour >= 8) & (idx.hour < 19)
    rng = np.random.default_rng(0)
    v = pd.Series(np.where(occupied, 100.0, 40.0) + rng.normal(0, 2, len(idx)), index=idx)
    day_type = pd.Series(np.where(workday, "ngay_lam_viec", "cuoi_tuan"), index=idx)
    return v, day_type


def test_points_to_events_merges_and_filters():
    idx = pd.date_range("2017-01-01", periods=10, freq="h")
    s = pd.Series([0, 5, 0, 5, 0, 0, 0, 5, 0, 0], index=idx, dtype=float)
    ev = points_to_events(s, 1.0, "b1", "t", max_gap=1)
    assert list(ev["duration_h"]) == [3, 1]
    assert len(points_to_events(s, 1.0, "b1", "t", max_gap=1, min_duration=2)) == 1


def test_profile_band_flags_spike_and_after_hours():
    v, day_type = _office()
    spike_ts = pd.Timestamp("2017-02-21 11:00")  # Tuesday
    night = (v.index >= "2017-02-22 20:00") & (v.index <= "2017-02-23 05:00")
    v[spike_ts] *= 2.5
    v[night] = 100.0
    band = profile_band_scores(v, day_type)
    ev = points_to_events(band["score"], 4.0, "b1", "profile_band")
    ev["type"] = classify_hourly(ev, day_type)
    assert ((ev["start"] <= spike_ts) & (ev["end"] >= spike_ts) & (ev["type"] == "spike")).any()
    assert ((ev["start"] >= pd.Timestamp("2017-02-22 20:00")) & (ev["type"] == "after_hours")).any()


def test_efficiency_scores_detect_load_increase():
    idx = pd.date_range("2017-01-01", periods=24 * 120, freq="h")
    rng = np.random.default_rng(1)
    cooling = pd.Series(50 + 30 * np.sin(np.arange(len(idx)) / 500) + rng.normal(0, 1, len(idx)), index=idx)
    power = 20 + 0.8 * cooling
    power[idx >= "2017-04-01"] *= 1.15
    sc = efficiency_scores(power, cooling)
    assert sc.loc["2017-04-05":"2017-04-20", "score"].median() > 0.1
    assert abs(sc.loc["2017-03-01":"2017-03-25", "score"].median()) < 0.03


def test_deviation_branch_on_template_columns():
    dates = pd.date_range("2017-01-01", periods=120, freq="D")
    rng = np.random.default_rng(2)
    actual = 100 + rng.normal(0, 2, 120)
    actual[90:100] += 15
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "actual_kWh": actual, "pred_dow_median": 100.0})
    sc = deviation_scores(df)
    assert sc["cusum"].iloc[95] > 5 and sc["cusum"].iloc[:80].max() < 5


def test_cusum_accumulates_shift():
    up, dn = cusum(np.r_[np.zeros(10), np.full(10, 2.0)])
    assert up[-1] == 15.0 and dn.max() == 0.0


def test_suppress_budget_and_metrics():
    t = pd.Timestamp("2017-01-01")
    h = pd.Timedelta(hours=1)
    ev = pd.DataFrame({
        "entity_id": ["b1"] * 3, "detector": "x",
        "start": [t, t + 10 * h, t + 20 * h], "end": [t, t + 10 * h, t + 21 * h],
        "duration_h": [1, 1, 2], "peak_score": [9.0, 5.0, 7.0], "peak_ts": [t, t + 10 * h, t + 20 * h],
    })
    faults = pd.DataFrame({"entity_id": ["b1"], "start": [t + 10 * h], "end": [t + 12 * h]})
    kept = suppress_sensor_faults(ev, faults)
    assert len(kept) == 2
    assert list(apply_budget(kept, n_days=5, budget_per_day=0.2)["peak_score"]) == [9.0]

    labels = pd.DataFrame({"entity_id": ["b1"], "start": [t + 21 * h], "end": [t + 23 * h]})
    m = event_metrics(kept, labels)
    assert m["event_precision"] == 0.5 and m["event_recall"] == 1.0 and m["median_delay_h"] == 0.0


def test_if_score_lag_feature_has_no_current_hour():
    from src.anomaly.detectors import if_score_features

    idx = pd.date_range("2017-01-01", periods=48, freq="h")
    scores = pd.Series(0.0, index=idx)
    scores.iloc[30] = 100.0
    lagged = if_score_features(scores, lagged=True)["if_score_prev_24h"]
    assert lagged.iloc[30] == 0.0  # the spike hour does not see its own score
    assert lagged.iloc[31] > 0.0


def test_fuse_scores_max_takes_most_alarmed_detector():
    from src.anomaly.detectors import fuse_scores

    rng = np.random.default_rng(3)
    idx = pd.date_range("2017-01-01", periods=200, freq="h")
    a = pd.Series(rng.normal(0, 1, 200), index=idx)
    b = pd.Series(rng.normal(10, 0.1, 200), index=idx)  # different scale
    a.iloc[150], b.iloc[160] = 8.0, 11.0  # each detector alarms on a different hour
    train = np.arange(200) < 100
    fused = fuse_scores({"a": a, "b": b}, train, how="max")
    top2 = set(fused.nlargest(2).index)
    assert top2 == {idx[150], idx[160]}


def test_profile_recovers_right_after_data_gap():
    v, day_type = _office(days=90)
    gap = (v.index >= "2017-02-01") & (v.index < "2017-02-15")
    v[gap] = np.nan
    day_type[gap] = np.nan  # dates with no rows have no day type either
    band = profile_band_scores(v, day_type)
    after = band.loc["2017-02-15":"2017-02-19"]  # Wed–Sun: workday and weekend slots
    assert after["baseline"].notna().all() and after["score"].notna().all()


def test_split_point_interval():
    from src.anomaly.events import interval_events, split_point_interval

    idx = pd.date_range("2017-01-01", periods=48, freq="h")
    z = pd.Series(0.0, index=idx)
    z.iloc[5] = 9.0  # isolated spike -> point
    z.iloc[20:32] = 3.0  # 12 h moderately raised -> no single hour >= 4, sustained mean >= 2.5
    point = points_to_events(z, 4.0, "b1", "p")
    sustained = interval_events(z, 6, 2.5, "b1", "s")
    out = split_point_interval(point, sustained)

    assert list(out["scope"]) == ["point", "interval"]
    iv = out[out["scope"] == "interval"].iloc[0]
    assert iv["start"] == idx[20] and iv["end"] == idx[31]
