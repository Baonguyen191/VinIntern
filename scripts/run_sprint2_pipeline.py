import os
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

def prepare_features_and_split(df_raw, entity_col, ts_col, value_col, metric_name):
    print(f"\n[{metric_name}] Starting feature preparation for {df_raw[entity_col].nunique()} buildings...", flush=True)
    t0 = time.time()
    
    df = df_raw.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.sort_values([entity_col, ts_col]).reset_index(drop=True)
    
    # Hour of week (0..167), hour of day (0..23), day of week (0..6)
    df['hour_of_week'] = df[ts_col].dt.dayofweek * 24 + df[ts_col].dt.hour
    df['hour_of_day'] = df[ts_col].dt.hour
    df['day_of_week'] = df[ts_col].dt.dayofweek
    
    # 80% train / 20% test strictly chronological split per building (exact same as Sprint 1)
    df['obs_idx'] = df.groupby(entity_col).cumcount()
    df['total_obs'] = df.groupby(entity_col)[ts_col].transform('count')
    df['split_idx'] = (df['total_obs'] * 0.8).astype(int)
    df['is_test'] = df['obs_idx'] >= df['split_idx']
    
    # Feature: hour_of_week_mean strictly computed on Train set
    hw_mean = df[~df['is_test']].groupby([entity_col, 'hour_of_week'])[value_col].mean().reset_index()
    hw_mean.rename(columns={value_col: 'hour_of_week_mean'}, inplace=True)
    df = pd.merge(df, hw_mean, on=[entity_col, 'hour_of_week'], how='left')
    
    # Features: lag_24h, lag_48h, lag_168h via timestamp offset
    for lag_h in [24, 48, 168]:
        df_lag = df[[entity_col, ts_col, value_col]].copy()
        df_lag['ts_future'] = df_lag[ts_col] + pd.Timedelta(hours=lag_h)
        col_name = f'lag_{lag_h}h'
        df_lag.rename(columns={value_col: col_name}, inplace=True)
        df = pd.merge(df, df_lag[[entity_col, 'ts_future', col_name]], 
                      left_on=[entity_col, ts_col], right_on=[entity_col, 'ts_future'], 
                      how='left').drop(columns=['ts_future'])
    
    # Feature: outdoor_temp_target (temperature at target hour t)
    # Forward and backward fill per building if any missing
    if 'temperature' in df.columns:
        df['outdoor_temp_target'] = df.groupby(entity_col)['temperature'].ffill().bfill().fillna(25.0)
    else:
        df['outdoor_temp_target'] = 25.0
        
    # Feature: day_type (ngay_lam_viec=0, cuoi_tuan=1, le=2)
    day_type_map = {'ngay_lam_viec': 0, 'cuoi_tuan': 1, 'le': 2}
    if 'day_type' in df.columns:
        df['day_type_code'] = df['day_type'].map(day_type_map).fillna(0)
    else:
        df['day_type_code'] = np.where(df['day_of_week'] >= 5, 1, 0)
        
    print(f"[{metric_name}] Feature prep completed in {time.time() - t0:.2f}s.", flush=True)
    return df

def train_and_evaluate(df, entity_col, ts_col, value_col, metric_name):
    features = [
        'lag_24h', 'lag_48h', 'lag_168h', 
        'hour_of_week_mean', 'hour_of_day', 'day_of_week', 
        'day_type_code', 'outdoor_temp_target'
    ]
    
    print(f"[{metric_name}] Training Ridge Regression across buildings and evaluating on test set...", flush=True)
    t0 = time.time()
    
    report_rows = []
    forecast_rows = []
    
    wins = 0
    total_evaluated = 0
    
    # Direct fast groupby iteration
    for b, b_df in df.groupby(entity_col):
        # Train data: points in train set with all features and target valid
        train_data = b_df[~b_df['is_test']].dropna(subset=features + [value_col])
        
        # Test data: points in test set with lag_168h and target and features valid
        # This guarantees EXACT SAME test sample alignment with Seasonal-naive (t-168h)
        test_data = b_df[b_df['is_test']].dropna(subset=['lag_168h', value_col] + features).copy()
        
        if len(train_data) < 24 or len(test_data) < 24:
            continue
            
        y_train = train_data[value_col].values
        X_train = train_data[features].values
        
        y_test = test_data[value_col].values
        X_test = test_data[features].values
        y_sn = test_data['lag_168h'].values
        
        mean_y = float(np.mean(y_test))
        if mean_y < 0.1:
            # Meter is essentially inactive / zero load throughout test set (denominator near zero)
            continue
            
        total_evaluated += 1
        
        # 1. Seasonal Naive Metrics
        mae_sn = float(np.mean(np.abs(y_test - y_sn)))
        rmse_sn = float(np.sqrt(np.mean((y_test - y_sn)**2)))
        cvrmse_sn = float((rmse_sn / mean_y) * 100.0)
        
        # 2. Ridge Regression
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        
        y_pred_test = np.maximum(0.0, model.predict(X_test))
        
        mae_ridge = float(np.mean(np.abs(y_test - y_pred_test)))
        rmse_ridge = float(np.sqrt(np.mean((y_test - y_pred_test)**2)))
        cvrmse_ridge = float((rmse_ridge / mean_y) * 100.0)
        
        # Delta CV(RMSE) vs Seasonal Naive (% change)
        if cvrmse_sn > 1e-4:
            delta_cvrmse = float(((cvrmse_ridge - cvrmse_sn) / cvrmse_sn) * 100.0)
        else:
            delta_cvrmse = 0.0
        
        if cvrmse_ridge < cvrmse_sn:
            wins += 1
            
        # Compute residual std on train set for 80% confidence interval (z=1.28)
        train_preds = np.maximum(0.0, model.predict(X_train))
        train_res_std = float(np.std(y_train - train_preds))
        
        test_data['forecast_next_24h'] = y_pred_test
        test_data['forecast_lower'] = np.maximum(0.0, y_pred_test - 1.28 * train_res_std)
        test_data['forecast_upper'] = y_pred_test + 1.28 * train_res_std
        test_data['metric'] = metric_name
        
        forecast_rows.append(test_data[[
            ts_col, entity_col, value_col, 
            'forecast_next_24h', 'forecast_lower', 'forecast_upper', 'metric'
        ]])
        
        # Append comparison records
        report_rows.append({
            'building_id': b,
            'metric': metric_name,
            'model': 'seasonal_naive_168h',
            'MAE': round(mae_sn, 4),
            'CV(RMSE)': round(cvrmse_sn, 2),
            'delta_cvrmse_vs_seasonal_naive_pct': 0.0
        })
        
        report_rows.append({
            'building_id': b,
            'metric': metric_name,
            'model': 'ridge_regression',
            'MAE': round(mae_ridge, 4),
            'CV(RMSE)': round(cvrmse_ridge, 2),
            'delta_cvrmse_vs_seasonal_naive_pct': round(delta_cvrmse, 2)
        })
        
    df_report = pd.DataFrame(report_rows)
    df_forecast = pd.concat(forecast_rows, ignore_index=True) if forecast_rows else pd.DataFrame()
    
    if not df_forecast.empty:
        df_forecast.rename(columns={
            ts_col: 'timestamp',
            entity_col: 'building_id',
            value_col: 'value_actual'
        }, inplace=True)
        
    win_rate = (wins / total_evaluated * 100.0) if total_evaluated > 0 else 0.0
    print(f"[{metric_name}] Ridge vs Seasonal-naive Evaluation Finished in {time.time() - t0:.2f}s:", flush=True)
    print(f"  - Total buildings evaluated: {total_evaluated}", flush=True)
    print(f"  - Ridge WON: {wins}/{total_evaluated} ({win_rate:.1f}%)", flush=True)
    print(f"  - Mean Seasonal-Naive CV(RMSE): {df_report[df_report['model']=='seasonal_naive_168h']['CV(RMSE)'].mean():.2f}%", flush=True)
    print(f"  - Mean Ridge CV(RMSE): {df_report[df_report['model']=='ridge_regression']['CV(RMSE)'].mean():.2f}%", flush=True)
    
    return df_report, df_forecast, wins, total_evaluated

def compute_optimization_scenarios(df_forecast):
    """
    Computes 3 preliminary optimization scenarios on predicted electrical load.
    Tariff based on EVN Commercial (Decision 1279/QD-BCT and 963/QD-BCT, >= 22kV):
      - Off-peak (thap_diem, 00:00 - 06:00, all days): 1,609 VND/kWh
      - Peak (cao_diem, 18:00 - 23:00, Mon-Sat): 5,025 VND/kWh (hours 18, 19, 20, 21, 22)
      - Standard (binh_thuong, remaining hours): 2,887 VND/kWh
    """
    print("\n[OPTIMIZATION] Computing 3 preliminary optimization scenarios on forecast...", flush=True)
    
    all_building_scenarios = []
    
    for metric_name, group in df_forecast.groupby('metric'):
        df = group.copy()
        df['ts'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['ts'].dt.hour
        df['dayofweek'] = df['ts'].dt.dayofweek
        
        # Determine tariff period
        is_offpeak = df['hour'].isin([0, 1, 2, 3, 4, 5])
        is_peak = (~df['dayofweek'].isin([6])) & (df['hour'].isin([18, 19, 20, 21, 22]))
        
        price_offpeak = 1609.0
        price_peak = 5025.0
        price_standard = 2887.0
        
        df['tariff_rate'] = np.where(is_offpeak, price_offpeak, np.where(is_peak, price_peak, price_standard))
        df['is_peak'] = is_peak
        df['is_offpeak'] = is_offpeak
        
        # Conversion for M2: cooling_kw to electric power kW using nominal COP = 5.4
        cop = 5.4 if metric_name == 'cooling_kw' else 1.0
        df['load_kw'] = df['forecast_next_24h'] / cop
        
        # Compute for each individual building
        for b_id, b_sub in df.groupby('building_id'):
            sub_copy = b_sub.copy()
            sub_copy['date'] = sub_copy['ts'].dt.date
            
            # Scenario A: Hiện trạng
            kwh_a = float(sub_copy['load_kw'].sum())
            cost_a = float((sub_copy['load_kw'] * sub_copy['tariff_rate']).sum())
            
            # Scenario B: Dịch tải thủ công (15% peak shifted to off-peak)
            peak_load = np.where(sub_copy['is_peak'], sub_copy['load_kw'] * 0.15, 0.0)
            sub_copy['shifted_out'] = peak_load
            
            daily_shifted = sub_copy.groupby('date')['shifted_out'].transform('sum')
            sub_copy['load_kw_b'] = np.where(
                sub_copy['is_peak'],
                sub_copy['load_kw'] * 0.85,
                np.where(sub_copy['is_offpeak'], sub_copy['load_kw'] + daily_shifted / 6.0, sub_copy['load_kw'])
            )
            
            kwh_b = float(sub_copy['load_kw_b'].sum())
            cost_b = float((sub_copy['load_kw_b'] * sub_copy['tariff_rate']).sum())
            savings_vnd_b = cost_a - cost_b
            savings_pct_b = (savings_vnd_b / cost_a * 100.0) if cost_a > 0 else 0.0
            
            # Scenario C: Giảm đỉnh đơn giản (10% peak load curtailment)
            sub_copy['load_kw_c'] = np.where(
                sub_copy['is_peak'],
                sub_copy['load_kw'] * 0.90,
                sub_copy['load_kw']
            )
            kwh_c = float(sub_copy['load_kw_c'].sum())
            cost_c = float((sub_copy['load_kw_c'] * sub_copy['tariff_rate']).sum())
            savings_vnd_c = cost_a - cost_c
            savings_pct_c = (savings_vnd_c / cost_a * 100.0) if cost_a > 0 else 0.0
            
            all_building_scenarios.extend([
                {
                    'building_id': b_id,
                    'metric': metric_name,
                    'scenario': 'hien_trang',
                    'total_kwh': round(kwh_a, 2),
                    'cost_vnd': round(cost_a, 0),
                    'savings_vnd': 0.0,
                    'savings_pct': 0.0,
                    'assumptions': 'Hien trang van hanh theo du bao phu tai, khong can thiep dich chuyen hay tiet giam'
                },
                {
                    'building_id': b_id,
                    'metric': metric_name,
                    'scenario': 'dich_tai_thu_cong',
                    'total_kwh': round(kwh_b, 2),
                    'cost_vnd': round(cost_b, 0),
                    'savings_vnd': round(savings_vnd_b, 0),
                    'savings_pct': round(savings_pct_b, 2),
                    'assumptions': 'Dich 15% phu tai gio cao diem (18h-22h T2-T7) sang gio thap diem (0h-6h). Tong kWh khong doi'
                },
                {
                    'building_id': b_id,
                    'metric': metric_name,
                    'scenario': 'giam_dinh_don_gian',
                    'total_kwh': round(kwh_c, 2),
                    'cost_vnd': round(cost_c, 0),
                    'savings_vnd': round(savings_vnd_c, 0),
                    'savings_pct': round(savings_pct_c, 2),
                    'assumptions': 'Cat giam 10% cong suat gio cao diem (18h-22h T2-T7) nho tang setpoint/tiet giam den'
                }
            ])

    df_bldgs = pd.DataFrame(all_building_scenarios)
    
    # Compute clean PORTFOLIO_TOTAL by summing building results
    portfolio_rows = []
    for metric_name, m_group in df_bldgs.groupby('metric'):
        base_cost = m_group[m_group['scenario'] == 'hien_trang']['cost_vnd'].sum()
        for sc_name in ['hien_trang', 'dich_tai_thu_cong', 'giam_dinh_don_gian']:
            sc_sub = m_group[m_group['scenario'] == sc_name]
            tot_kwh = sc_sub['total_kwh'].sum()
            tot_cost = sc_sub['cost_vnd'].sum()
            tot_sav = sc_sub['savings_vnd'].sum()
            sav_pct = (tot_sav / base_cost * 100.0) if base_cost > 0 else 0.0
            
            assump = sc_sub['assumptions'].iloc[0]
            portfolio_rows.append({
                'building_id': 'PORTFOLIO_TOTAL',
                'metric': metric_name,
                'scenario': sc_name,
                'total_kwh': round(tot_kwh, 2),
                'cost_vnd': round(tot_cost, 0),
                'savings_vnd': round(tot_sav, 0),
                'savings_pct': round(sav_pct, 2),
                'assumptions': assump
            })
            
    df_portfolio = pd.DataFrame(portfolio_rows)
    df_all_scenarios = pd.concat([df_portfolio, df_bldgs], ignore_index=True)
    return df_all_scenarios

def export_optimization_csv(df_scenarios, out_path):
    header_comment = """# ==============================================================================
# optimization_scenarios_preliminary.csv
# GHI CHU QUAN TRONG:
# Kich ban so bo, dung luat don gian (if-then), CHUA phai ket qua tu bo giai toi uu (MILP)
# — khong dung so nay de khang dinh muc tiet kiem chac chan, Sprint 3 se tinh chinh.
#
# CAN CU VA GIA DINH MINH HOA:
# 1. Bieu gia tam: Dien kinh doanh EVN (QD 1279/QD-BCT & 963/QD-BCT), cap dien ap >= 22kV.
#    - Thap diem (00:00 - 06:00 T2-CN): 1.609 VND/kWh
#    - Binh thuong (06:00 - 18:00 & 23:00 - 24:00 T2-T7; 06:00 - 24:00 CN): 2.887 VND/kWh
#    - Cao diem (18:00 - 23:00 T2-T7, khong co CN): 5.025 VND/kWh
# 2. Kich ban 'hien_trang': Chi phi theo bieu gia TOU nhan voi tai du bao cua mo hinh hoi quy.
# 3. Kich ban 'dich_tai_thu_cong': Dich 15% tai gio cao diem sang gio thap diem gan nhat (0h-6h),
#    tong kWh tieu thu bao toan 100%.
# 4. Kich ban 'giam_dinh_don_gian': Cat giam 10% cong suat trong gio cao diem nho dieu chinh
#    nhiet do setpoint / giam chieu sang phu tro.
# 5. Doi voi Chiller (cooling_kw): Cong suat dien quy doi = cooling_kw / COP_rated (COP dinh muc = 5.4).
# ==============================================================================
"""
    with open(out_path, 'w', encoding='utf-8-sig') as f:
        f.write(header_comment)
        df_scenarios.to_csv(f, index=False, encoding='utf-8-sig')
    print(f"[OPTIMIZATION] Successfully exported {out_path} ({len(df_scenarios)} rows).", flush=True)

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    m1_path = os.path.join(base_dir, 'data_normalized', 'telemetry_M1.parquet')
    m2_path = os.path.join(base_dir, 'data_normalized', 'telemetry_M2.parquet')
    
    out_comparison = os.path.join(base_dir, 'regression_vs_baseline.csv')
    out_forecast = os.path.join(base_dir, 'data_normalized', 'forecast_test_results.parquet')
    out_scenarios = os.path.join(base_dir, 'optimization_scenarios_preliminary.csv')
    
    all_reports = []
    all_forecasts = []
    
    total_wins = 0
    total_buildings = 0
    
    # 1. Process M1 (power_active_kw)
    if os.path.exists(m1_path):
        df_m1_raw = pd.read_parquet(m1_path, columns=['entity_id', 'ts', 'power_active_kw', 'temperature', 'day_type'])
        df_m1_prep = prepare_features_and_split(df_m1_raw, 'entity_id', 'ts', 'power_active_kw', 'power_active_kw')
        rep_m1, fc_m1, wins_m1, tot_m1 = train_and_evaluate(df_m1_prep, 'entity_id', 'ts', 'power_active_kw', 'power_active_kw')
        all_reports.append(rep_m1)
        all_forecasts.append(fc_m1)
        total_wins += wins_m1
        total_buildings += tot_m1
        
    # 2. Process M2 (cooling_kw)
    if os.path.exists(m2_path):
        df_m2_raw = pd.read_parquet(m2_path, columns=['entity_id', 'ts', 'cooling_kw', 'temperature', 'day_type'])
        df_m2_prep = prepare_features_and_split(df_m2_raw, 'entity_id', 'ts', 'cooling_kw', 'cooling_kw')
        rep_m2, fc_m2, wins_m2, tot_m2 = train_and_evaluate(df_m2_prep, 'entity_id', 'ts', 'cooling_kw', 'cooling_kw')
        all_reports.append(rep_m2)
        all_forecasts.append(fc_m2)
        total_wins += wins_m2
        total_buildings += tot_m2
        
    # Combine reports & forecasts
    df_report_all = pd.concat(all_reports, ignore_index=True)
    df_forecast_all = pd.concat(all_forecasts, ignore_index=True)
    
    # Export regression_vs_baseline.csv
    df_report_all = df_report_all[['building_id', 'metric', 'model', 'MAE', 'CV(RMSE)', 'delta_cvrmse_vs_seasonal_naive_pct']]
    df_report_all.to_csv(out_comparison, index=False, encoding='utf-8-sig')
    print(f"\n[SUCCESS] Exported regression_vs_baseline.csv: {len(df_report_all)} rows.", flush=True)
    
    # Export updated forecast_test_results.parquet
    df_forecast_all.to_parquet(out_forecast, index=False)
    print(f"[SUCCESS] Exported forecast_test_results.parquet: {len(df_forecast_all)} rows.", flush=True)
    
    # Compute Optimization Scenarios
    df_scenarios = compute_optimization_scenarios(df_forecast_all)
    export_optimization_csv(df_scenarios, out_scenarios)
    
    # Overall summary prints
    overall_win_rate = (total_wins / total_buildings * 100.0) if total_buildings > 0 else 0.0
    print("\n" + "="*80, flush=True)
    print("[SUMMARY] SPRINT 2 PIPELINE EXECUTION SUMMARY", flush=True)
    print("="*80, flush=True)
    print(f"Total evaluated buildings across M1 & M2: {total_buildings}", flush=True)
    print(f"Ridge Regression won against Seasonal-naive (t-168h): {total_wins}/{total_buildings} ({overall_win_rate:.2f}%)", flush=True)
    
    print("\n--- Summary by Metric & Model ---", flush=True)
    summary = df_report_all.groupby(['metric', 'model'])[['MAE', 'CV(RMSE)', 'delta_cvrmse_vs_seasonal_naive_pct']].mean().reset_index()
    print(summary.to_string(index=False), flush=True)
    
    print("\n--- Preliminary Optimization Scenarios (Portfolio Total) ---", flush=True)
    port_scenarios = df_scenarios[df_scenarios['building_id'] == 'PORTFOLIO_TOTAL']
    print(port_scenarios[['metric', 'scenario', 'total_kwh', 'cost_vnd', 'savings_vnd', 'savings_pct']].to_string(index=False), flush=True)
    print("="*80, flush=True)

if __name__ == '__main__':
    main()
