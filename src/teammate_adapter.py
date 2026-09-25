import os
import pandas as pd
import numpy as np

# Real outputs: data/teammates/{baseline,anomaly}_output.csv
# Mock outputs: tests/fixtures/mock_{baseline,anomaly}.csv
from paths import REAL_BASELINE_PATH, REAL_ANOMALY_PATH, MOCK_BASELINE_PATH, MOCK_ANOMALY_PATH

# ==============================================================================
# CONFIGURATION SWITCH: SET TO FALSE ONCE TEAMMATES 1 & 2 DELIVER REAL OUTPUTS
# ==============================================================================
USE_MOCK = True  # <-- ĐỔI THÀNH False KHI CÓ FILE THẬT TỪ 2 BẠN

# In-memory caches to ensure high dashboard performance
_baseline_cache = None
_anomaly_cache = None

def _load_baseline_df():
    global _baseline_cache
    if _baseline_cache is not None:
        return _baseline_cache
    path = MOCK_BASELINE_PATH if USE_MOCK else REAL_BASELINE_PATH
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        _baseline_cache = df
    else:
        _baseline_cache = pd.DataFrame(columns=['timestamp', 'building_id', 'metric', 'baseline_expected'])
    return _baseline_cache

def _load_anomaly_df():
    global _anomaly_cache
    if _anomaly_cache is not None:
        return _anomaly_cache
    path = MOCK_ANOMALY_PATH if USE_MOCK else REAL_ANOMALY_PATH
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        _anomaly_cache = df
    else:
        _anomaly_cache = pd.DataFrame(columns=['timestamp', 'building_id', 'metric', 'is_anomaly', 'anomaly_reason'])
    return _anomaly_cache

def get_baseline_expected(building_id: str, metric: str, timestamp):
    """
    Returns baseline_expected value(s) for a specific building, metric, and timestamp(s).
    Signature conforms strictly to demo_schema.md.
    
    Parameters:
      - building_id: str
      - metric: str
      - timestamp: single timestamp (str / pd.Timestamp) OR iterable / pd.Series / DatetimeIndex
      
    Returns:
      - float if single timestamp provided
      - pd.Series if collection of timestamps provided
    """
    df_base = _load_baseline_df()
    is_scalar = not hasattr(timestamp, '__iter__') or isinstance(timestamp, (str, bytes))
    ts_series = pd.Series([pd.to_datetime(timestamp)]) if is_scalar else pd.Series(pd.to_datetime(timestamp))
    
    sub = df_base[(df_base['building_id'] == building_id) & (df_base['metric'] == metric)]
    
    if not sub.empty:
        merged = pd.merge(pd.DataFrame({'timestamp': ts_series}), sub[['timestamp', 'baseline_expected']], on='timestamp', how='left')
        res = merged['baseline_expected']
        if res.isna().any():
            # Interpolate or fill if timestamps slightly misaligned
            res = res.ffill().bfill()
    else:
        # Fallback if building is not pre-cached in mock fixture
        res = pd.Series(np.nan, index=range(len(ts_series)))
        
    return res.iloc[0] if is_scalar else res

def get_anomaly_flags(building_id: str, metric: str, timestamp_range):
    """
    Returns anomaly flags and reason for a specific building, metric, and timestamp collection.
    Signature conforms strictly to demo_schema.md.
    
    Parameters:
      - building_id: str
      - metric: str
      - timestamp_range: iterable / pd.Series / DatetimeIndex of timestamps
      
    Returns:
      pd.DataFrame with columns ['timestamp', 'is_anomaly', 'anomaly_reason']
    """
    df_anom = _load_anomaly_df()
    ts_list = pd.to_datetime(pd.Series(timestamp_range)).values
    df_req = pd.DataFrame({'timestamp': ts_list})
    
    sub = df_anom[(df_anom['building_id'] == building_id) & (df_anom['metric'] == metric)]
    
    if not sub.empty:
        merged = pd.merge(df_req, sub[['timestamp', 'is_anomaly', 'anomaly_reason']], on='timestamp', how='left')
        merged['is_anomaly'] = merged['is_anomaly'].fillna(False).astype(bool)
    else:
        merged = df_req.copy()
        merged['is_anomaly'] = False
        merged['anomaly_reason'] = None
        
    return merged[['timestamp', 'is_anomaly', 'anomaly_reason']]

def get_adapter_status():
    """
    Returns description string of current adapter status for display on dashboard.
    """
    if USE_MOCK:
        return "MOCK (Placeholder đang chờ Bạn 1 & Bạn 2 — Đổi USE_MOCK=False khi có file thật)"
    else:
        return "REAL (Dữ liệu THẬT đã tích hợp từ Bạn 1 & Bạn 2)"

if __name__ == '__main__':
    print(f"[ADAPTER STATUS] USE_MOCK={USE_MOCK}")
    test_ts = ['2017-10-20 00:00:00', '2017-10-20 01:00:00']
    base = get_baseline_expected('Bull_education_Luke', 'cooling_kw', test_ts)
    print("Test get_baseline_expected:\n", base)
    anom = get_anomaly_flags('Bull_education_Luke', 'cooling_kw', test_ts)
    print("Test get_anomaly_flags:\n", anom)
