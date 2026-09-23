import numpy as np
import pandas as pd

OP_HOURS = (7, 19)
NIGHT_HOURS = (1, 4)
SLOTS_PER_WEEK = {"ngay_lam_viec": 5, "cuoi_tuan": 2}


def is_operating(day_type: pd.Series, hour: np.ndarray) -> np.ndarray:
    return (
        (day_type.astype(str).to_numpy() == "ngay_lam_viec")
        & (hour >= OP_HOURS[0]) & (hour < OP_HOURS[1])
    )


def is_night(hour: np.ndarray) -> np.ndarray:
    return (hour >= NIGHT_HOURS[0]) & (hour <= NIGHT_HOURS[1])


def profile_band(
    g: pd.DataFrame,
    key: str,
    n_weeks: int = 8,
    k: float = 3.0,
    rel_floor: float = 0.05,
    abs_floor: float = 0.0,
) -> pd.DataFrame:
    """Causal profile band per (day_type x tod_slot) on an hourly grid.

    expected = median of the same slot over the previous n_weeks (current sample excluded);
    scale = IQR/1.349 of the same history, floored so a flat history does not give a zero-width band.
    Holidays ("le") share the weekend profile.
    """
    day_group = g["day_type"].astype(str).replace({"le": "cuoi_tuan"})
    expected = pd.Series(np.nan, index=g.index)
    spread = pd.Series(np.nan, index=g.index)
    for (dg, _), idx in g.groupby([day_group, g["tod_slot"]]).groups.items():
        n = SLOTS_PER_WEEK[dg] * n_weeks
        roll = g.loc[idx, key].shift(1).rolling(n, min_periods=max(3, n // 4))
        expected.loc[idx] = roll.median()
        spread.loc[idx] = (roll.quantile(0.75) - roll.quantile(0.25)) / 1.349

    scale = pd.concat([spread, rel_floor * expected.abs()], axis=1).max(axis=1).clip(lower=abs_floor)
    return pd.DataFrame({
        "expected": expected,
        "lower": expected - k * scale,
        "upper": expected + k * scale,
        "scale": scale,
    })
