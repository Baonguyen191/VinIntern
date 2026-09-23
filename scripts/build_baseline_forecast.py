import os
import numpy as np
import pandas as pd

def process_dataframe(df, entity_col, ts_col, value_col, metric_name):
    print(f"[{metric_name}] Vectorized processing {df[entity_col].nunique()} buildings...")
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.sort_values([entity_col, ts_col]).reset_index(drop=True)
    
    # Hour of week (0..167)
    df['hour_of_week'] = df[ts_col].dt.dayofweek * 24 + df[ts_col].dt.hour
    
    # Index & split 80% train, 20% test per building
    df['obs_idx'] = df.groupby(entity_col).cumcount()
    df['total_obs'] = df.groupby(entity_col)[ts_col].transform('count')
    df['split_idx'] = (df['total_obs'] * 0.8).astype(int)
    df['is_test'] = df['obs_idx'] >= df['split_idx']

    # --- MODEL 1: Hour of Week Profile ---
    train_mask = ~df['is_test']
    hw_stats = df[train_mask].groupby([entity_col, 'hour_of_week'])[value_col].agg(['mean', 'std']).reset_index()
    hw_stats.rename(columns={'mean': 'hw_pred', 'std': 'hw_std'}, inplace=True)
    hw_stats['hw_std'] = hw_stats['hw_std'].fillna(0.0)

    df = pd.merge(df, hw_stats, on=[entity_col, 'hour_of_week'], how='left')
    df['hw_lower'] = np.maximum(0.0, df['hw_pred'] - 1.28 * df['hw_std'])
    df['hw_upper'] = df['hw_pred'] + 1.28 * df['hw_std']

    # --- MODEL 2: Seasonal Naive (168h lag) ---
    # Merge on (entity_id, ts - 168h) to get exact 168h lag
    df_lag = df[[entity_col, ts_col, value_col]].copy()
    df_lag['ts_future'] = df_lag[ts_col] + pd.Timedelta(hours=168)
    df_lag.rename(columns={value_col: 'snaive_pred'}, inplace=True)
    
    df = pd.merge(df, df_lag[[entity_col, 'ts_future', 'snaive_pred']], 
                  left_on=[entity_col, ts_col], right_on=[entity_col, 'ts_future'], 
                  how='left').drop(columns=['ts_future'])

    # Compute residual std on train set per building for Seasonal Naive
    df['sn_train_res'] = np.where(~df['is_test'], df[value_col] - df['snaive_pred'], np.nan)
    sn_std = df.groupby(entity_col)['sn_train_res'].std().reset_index()
    sn_std.rename(columns={'sn_train_res': 'sn_std'}, inplace=True)
    sn_std['sn_std'] = sn_std['sn_std'].fillna(0.0)

    df = pd.merge(df, sn_std, on=entity_col, how='left')
    df['sn_lower'] = np.maximum(0.0, df['snaive_pred'] - 1.28 * df['sn_std'])
    df['sn_upper'] = df['snaive_pred'] + 1.28 * df['sn_std']

    # --- EVALUATION ON TEST SET ---
    test_df = df[df['is_test']].copy()
    report_rows = []

    for entity, group in test_df.groupby(entity_col):
        y_true = group[value_col].values

        # 1. Hour of Week Profile
        p_hw = group['hw_pred'].values
        valid_hw = ~np.isnan(p_hw) & ~np.isnan(y_true)
        if valid_hw.sum() > 0:
            yt = y_true[valid_hw]
            yp = p_hw[valid_hw]
            mae_hw = float(np.mean(np.abs(yt - yp)))
            nz = np.abs(yt) > 0.1
            mape_hw = float(np.mean(np.abs(yt[nz] - yp[nz]) / np.abs(yt[nz])) * 100.0) if nz.sum() > 0 else 0.0
            report_rows.append({
                'building_id': entity,
                'metric': metric_name,
                'model': 'hour_of_week_profile',
                'MAE': round(mae_hw, 4),
                'MAPE': round(mape_hw, 4),
                'n_obs': int(valid_hw.sum())
            })

        # 2. Seasonal Naive
        p_sn = group['snaive_pred'].values
        valid_sn = ~np.isnan(p_sn) & ~np.isnan(y_true)
        if valid_sn.sum() > 0:
            yt = y_true[valid_sn]
            yp = p_sn[valid_sn]
            mae_sn = float(np.mean(np.abs(yt - yp)))
            nz = np.abs(yt) > 0.1
            mape_sn = float(np.mean(np.abs(yt[nz] - yp[nz]) / np.abs(yt[nz])) * 100.0) if nz.sum() > 0 else 0.0
            report_rows.append({
                'building_id': entity,
                'metric': metric_name,
                'model': 'seasonal_naive_168h',
                'MAE': round(mae_sn, 4),
                'MAPE': round(mape_sn, 4),
                'n_obs': int(valid_sn.sum())
            })

    # Output forecast records (Test set)
    forecast_df = test_df[[
        ts_col, entity_col, value_col, 
        'hw_pred', 'hw_lower', 'hw_upper'
    ]].copy()
    forecast_df.columns = [
        'timestamp', 'building_id', 'value_actual', 
        'forecast_next_24h', 'forecast_lower', 'forecast_upper'
    ]
    forecast_df['metric'] = metric_name

    return pd.DataFrame(report_rows), forecast_df

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    m1_path = os.path.join(base_dir, 'data_normalized', 'telemetry_M1.parquet')
    m2_path = os.path.join(base_dir, 'data_normalized', 'telemetry_M2.parquet')
    report_out = os.path.join(base_dir, 'baseline_forecast_report.csv')
    forecast_out = os.path.join(base_dir, 'data_normalized', 'forecast_test_results.parquet')

    all_reports = []
    all_forecasts = []

    if os.path.exists(m1_path):
        print(f"Reading {m1_path}...")
        df_m1 = pd.read_parquet(m1_path, columns=['entity_id', 'ts', 'power_active_kw'])
        rep_m1, fc_m1 = process_dataframe(df_m1, 'entity_id', 'ts', 'power_active_kw', 'power_active_kw')
        all_reports.append(rep_m1)
        all_forecasts.append(fc_m1)

    if os.path.exists(m2_path):
        print(f"Reading {m2_path}...")
        df_m2 = pd.read_parquet(m2_path, columns=['entity_id', 'ts', 'cooling_kw'])
        rep_m2, fc_m2 = process_dataframe(df_m2, 'entity_id', 'ts', 'cooling_kw', 'cooling_kw')
        all_reports.append(rep_m2)
        all_forecasts.append(fc_m2)

    df_report_all = pd.concat(all_reports, ignore_index=True)
    df_report_all.to_csv(report_out, index=False, encoding='utf-8-sig')
    print(f"\n[SUCCESS] Exported baseline_forecast_report.csv: {len(df_report_all)} rows.")

    df_fc_all = pd.concat(all_forecasts, ignore_index=True)
    df_fc_all.to_parquet(forecast_out, index=False)
    print(f"[SUCCESS] Exported forecast_test_results.parquet: {len(df_fc_all)} rows.")

    summary = df_report_all.groupby(['metric', 'model'])[['MAE', 'MAPE', 'n_obs']].agg({
        'MAE': 'mean',
        'MAPE': 'mean',
        'n_obs': 'sum'
    }).reset_index()
    print("\n=== AGGREGATE BASELINE REPORT (AVERAGE OVER BUILDINGS) ===")
    print(summary.to_string(index=False))

if __name__ == '__main__':
    main()
