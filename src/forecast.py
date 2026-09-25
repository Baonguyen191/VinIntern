import os
import numpy as np
import pandas as pd

from paths import FORECAST_REPORT_PATH as REPORT_PATH, REGRESSION_COMP_PATH, FORECAST_PARQUET_PATH

def build_forecast_final_report():
    """
    Selects the winning model for each (building_id, metric) from Sprint 2:
      - If Ridge CV(RMSE) < Seasonal-naive CV(RMSE) (delta < 0): choose ridge_regression.
      - Otherwise, keep seasonal_naive_168h as final model.
    Computes Median Absolute Error (MedAE) on Sprint 2 test set.
    Exports forecast_final_report.csv.
    """
    print("[FORECAST_FINAL] Building forecast_final_report.csv...", flush=True)
    if not os.path.exists(REGRESSION_COMP_PATH):
        raise FileNotFoundError(f"Missing {REGRESSION_COMP_PATH}. Please run `python scripts/train_ridge.py` first.")
        
    df_comp = pd.read_csv(REGRESSION_COMP_PATH)
    df_fc = pd.read_parquet(FORECAST_PARQUET_PATH)
    
    # Compute MedAE for Ridge on test set
    df_fc['abs_err'] = np.abs(df_fc['value_actual'] - df_fc['forecast_next_24h'])
    med_ae_map = df_fc.groupby(['building_id', 'metric'])['abs_err'].median().to_dict()
    
    report_rows = []
    for (b_id, m_name), group in df_comp.groupby(['building_id', 'metric']):
        sn_sub = group[group['model'] == 'seasonal_naive_168h']
        ridge_sub = group[group['model'] == 'ridge_regression']
        
        if sn_sub.empty or ridge_sub.empty:
            continue
            
        sn_row = sn_sub.iloc[0]
        ridge_row = ridge_sub.iloc[0]
        
        # Selection: Ridge wins if CV(RMSE) is strictly lower than Seasonal-naive
        if ridge_row['CV(RMSE)'] < sn_row['CV(RMSE)']:
            model_sel = 'ridge_regression'
            final_mae = ridge_row['MAE']
            final_cvrmse = ridge_row['CV(RMSE)']
        else:
            model_sel = 'seasonal_naive_168h'
            final_mae = sn_row['MAE']
            final_cvrmse = sn_row['CV(RMSE)']
            
        med_ae = med_ae_map.get((b_id, m_name), final_mae * 0.85)
        if pd.isna(med_ae) or med_ae <= 0:
            med_ae = final_mae * 0.85
            
        report_rows.append({
            'building_id': b_id,
            'metric': m_name,
            'model_selected': model_sel,
            'MAE': round(final_mae, 4),
            'CV(RMSE)': round(final_cvrmse, 2),
            'cvrmse_seasonal_naive': round(sn_row['CV(RMSE)'], 2),
            'delta_cvrmse_pct': round(ridge_row['delta_cvrmse_vs_seasonal_naive_pct'], 2),
            'med_ae': round(float(med_ae), 4)
        })
        
    df_final_report = pd.DataFrame(report_rows)
    df_final_report.to_csv(REPORT_PATH, index=False, encoding='utf-8-sig')
    print(f"[FORECAST_FINAL] Successfully generated {REPORT_PATH} ({len(df_final_report)} rows).", flush=True)
    
    # Update forecast_test_results.parquet with +/- 1.5 * MedAE confidence bands
    med_lookup = df_final_report.set_index(['building_id', 'metric'])['med_ae'].to_dict()
    df_fc['med_ae'] = df_fc.apply(lambda r: med_lookup.get((r['building_id'], r['metric']), 1.0), axis=1)
    df_fc['forecast_lower'] = np.maximum(0.0, df_fc['forecast_next_24h'] - 1.5 * df_fc['med_ae'])
    df_fc['forecast_upper'] = df_fc['forecast_next_24h'] + 1.5 * df_fc['med_ae']
    df_fc.drop(columns=['abs_err', 'med_ae'], inplace=True, errors='ignore')
    df_fc.to_parquet(FORECAST_PARQUET_PATH, index=False)
    print(f"[FORECAST_FINAL] Updated {FORECAST_PARQUET_PATH} with 1.5x MedAE confidence interval.", flush=True)
    
    return df_final_report

def get_final_report():
    if not os.path.exists(REPORT_PATH):
        return build_forecast_final_report()
    return pd.read_csv(REPORT_PATH)

def predict_next_24h(building_id: str, metric: str, as_of_time=None) -> pd.DataFrame:
    """
    Returns the next 24-hour forecast for (building_id, metric) according to the
    selected winning model, accompanied by a +/- 1.5 * MedAE confidence interval.
    
    Parameters:
      - building_id: str (e.g. 'Panther_office_space_Bear' or 'Bear_education_Austin')
      - metric: str ('power_active_kw' or 'cooling_kw')
      - as_of_time: str or pd.Timestamp (issue time, defaults to latest available test window)
      
    Returns:
      pd.DataFrame with 24 rows and columns:
      ['timestamp', 'building_id', 'metric', 'value_actual', 'forecast_next_24h', 
       'forecast_lower', 'forecast_upper', 'model_used']
    """
    df_report = get_final_report()
    match = df_report[(df_report['building_id'] == building_id) & (df_report['metric'] == metric)]
    
    if match.empty:
        raise ValueError(f"No forecast metadata found for ({building_id}, {metric})")
        
    model_sel = match['model_selected'].iloc[0]
    med_ae = match['med_ae'].iloc[0]
    
    df_fc = pd.read_parquet(FORECAST_PARQUET_PATH)
    sub = df_fc[(df_fc['building_id'] == building_id) & (df_fc['metric'] == metric)].sort_values('timestamp').copy()
    
    if sub.empty:
        raise ValueError(f"No forecast time-series found for ({building_id}, {metric})")
        
    sub['timestamp'] = pd.to_datetime(sub['timestamp'])
    
    if as_of_time is not None:
        t0 = pd.to_datetime(as_of_time)
        mask = (sub['timestamp'] > t0) & (sub['timestamp'] <= t0 + pd.Timedelta(hours=24))
        df_24h = sub[mask].copy()
    else:
        # Default: take the last 24 hours of available test period
        t_max = sub['timestamp'].max()
        t_start = t_max - pd.Timedelta(hours=23)
        df_24h = sub[sub['timestamp'] >= t_start].copy()
        
    if len(df_24h) < 24:
        # If less than 24 rows, take the tail 24 rows
        df_24h = sub.tail(24).copy()
        
    df_24h = df_24h.head(24).copy()
    
    # Ensure confidence interval is exactly +/- 1.5 * MedAE
    df_24h['forecast_lower'] = np.maximum(0.0, df_24h['forecast_next_24h'] - 1.5 * med_ae)
    df_24h['forecast_upper'] = df_24h['forecast_next_24h'] + 1.5 * med_ae
    df_24h['model_used'] = model_sel
    
    cols = ['timestamp', 'building_id', 'metric', 'value_actual', 
            'forecast_next_24h', 'forecast_lower', 'forecast_upper', 'model_used']
    return df_24h[cols].reset_index(drop=True)

if __name__ == '__main__':
    rep = build_forecast_final_report()
    print("\n--- Summary of Model Selection ---")
    vc = rep['model_selected'].value_counts()
    print(vc)
    print(f"Seasonal-naive retained: {vc.get('seasonal_naive_168h', 0)} / {len(rep)} ({vc.get('seasonal_naive_168h', 0)/len(rep)*100:.2f}%)")
    print(f"Ridge regression selected: {vc.get('ridge_regression', 0)} / {len(rep)} ({vc.get('ridge_regression', 0)/len(rep)*100:.2f}%)")
    
    # Test predict_next_24h on a sample building
    sample_bld = rep[rep['metric'] == 'cooling_kw']['building_id'].iloc[0]
    sample_fc = predict_next_24h(sample_bld, 'cooling_kw')
    print(f"\nTest predict_next_24h on {sample_bld} (cooling_kw):")
    print(sample_fc[['timestamp', 'forecast_next_24h', 'forecast_lower', 'forecast_upper', 'model_used']].head(5))
