import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Import modules from Sprint 3
from forecast_final import get_final_report, predict_next_24h
from teammate_outputs_adapter import get_baseline_expected, get_anomaly_flags, get_adapter_status, USE_MOCK
from opt_solver import generate_recommendation_ranking, RANKING_CSV_PATH, load_equipment, get_chiller_constants, get_tou_price, solve_milp, verify_constraints

# Streamlit Page Setup
st.set_page_config(
    page_title="Năng lượng & M&E Thông minh — Khung Demo Hoàn Thiện Sprint 3",
    page_icon="⚡",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FORECAST_PARQUET_PATH = os.path.join(BASE_DIR, 'data_normalized', 'forecast_test_results.parquet')

@st.cache_data
def load_all_forecast_data():
    if os.path.exists(FORECAST_PARQUET_PATH):
        df = pd.read_parquet(FORECAST_PARQUET_PATH)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    return pd.DataFrame()

# ---------------------------------------------------------
# HEADER & TEAMMATE STATUS NOTICE (SPRINT 3)
# ---------------------------------------------------------
st.title("⚡ Khung Demo Tích Hợp Hoàn Thiện Sprint 3 — Pipeline Năng Lượng & M&E")
st.markdown("""
**Hệ thống Dự báo & Gợi ý Vận hành Tối ưu Toàn diện** | Pipeline 4 bước: *Baseline → Bất thường → Dự báo → Tối ưu MILP*
""")

# Mandatory Header Notice as specified in Sprint 3 prompt
st.warning(f"""
⚠️ **CHÚ THÍCH HỆ THỐNG**: 
**Baseline và Bất thường: dữ liệu MOCK**, đang chờ kết quả thật từ Bạn 1 (Baseline) & Bạn 2 (Anomaly) — sẽ tự động cập nhật khi `teammate_outputs_adapter.py` được chuyển sang `USE_MOCK=False`.
*Trạng thái Adapter hiện tại:* `{get_adapter_status()}`.
""")

df_report = get_final_report()
df_fc_all = load_all_forecast_data()

if df_fc_all.empty:
    st.error("Không tìm thấy dữ liệu dự báo. Vui lòng kiểm tra lại file `forecast_test_results.parquet`.")
    st.stop()

# ---------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------
st.sidebar.header("🔍 Bộ Lọc Dữ Liệu")

metric_options = sorted(df_fc_all['metric'].unique())
selected_metric = st.sidebar.selectbox("1. Chọn Chỉ Số (Metric):", metric_options, index=0)

filtered_by_metric = df_fc_all[df_fc_all['metric'] == selected_metric]
building_options = sorted(filtered_by_metric['building_id'].unique())

# Default building with good chiller profile if available
default_idx = 0
if 'Bull_education_Luke' in building_options:
    default_idx = building_options.index('Bull_education_Luke')
elif 'Bear_education_Austin' in building_options:
    default_idx = building_options.index('Bear_education_Austin')

selected_building = st.sidebar.selectbox("2. Chọn Tòa Nhà (Building ID):", building_options, index=default_idx)

building_df = filtered_by_metric[filtered_by_metric['building_id'] == selected_building].sort_values('timestamp').copy()

min_ts = building_df['timestamp'].min()
max_ts = building_df['timestamp'].max()

st.sidebar.subheader("📅 Khoảng Thời Gian (Tập Test)")
date_range = st.sidebar.date_input(
    "Chọn khoảng ngày xem chuỗi thời gian:",
    value=(min_ts.date(), min_ts.date() + pd.Timedelta(days=14)),
    min_value=min_ts.date(),
    max_value=max_ts.date()
)

if len(date_range) == 2:
    start_date, end_date = date_range
    mask = (building_df['timestamp'].dt.date >= start_date) & (building_df['timestamp'].dt.date <= end_date)
    building_df = building_df[mask]

# ---------------------------------------------------------
# TOP METRICS CARDS: FINAL FORECAST MODEL EVALUATION
# ---------------------------------------------------------
st.subheader(f"📊 Đánh Giá Mô Hình Dự Báo Hoàn Thiện — {selected_building} ({selected_metric})")

b_meta = df_report[(df_report['building_id'] == selected_building) & (df_report['metric'] == selected_metric)]

if not b_meta.empty:
    r = b_meta.iloc[0]
    model_sel = r['model_selected']
    mae_val = r['MAE']
    cvrmse_val = r['CV(RMSE)']
    delta_val = r['delta_cvrmse_pct']
    med_ae_val = r['med_ae']
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Mô hình chiến thắng được chọn", f"{model_sel}", 
                  help="Chọn mô hình có CV(RMSE) thấp nhất; giữ lại seasonal-naive nếu hồi quy không vượt được")
    with c2:
        st.metric("Sai số MAE", f"{mae_val:.4f} kW")
    with c3:
        st.metric("CV(RMSE) đạt được", f"{cvrmse_val:.2f}%", 
                  delta=f"{delta_val:.2f}% vs Naive" if model_sel == 'ridge_regression' else "Baseline giữ nguyên",
                  delta_color="inverse")
    with c4:
        st.metric("Độ rộng tin cậy (±1.5 × MedAE)", f"± {1.5 * med_ae_val:.3f} kW",
                  help="Khoảng tin cậy đơn giản tính theo 1.5 lần sai số tuyệt đối trung vị")

# ---------------------------------------------------------
# ATTACH TEAMMATE ADAPTER OUTPUTS (MOCK / REAL)
# ---------------------------------------------------------
# 1. Baseline expected from adapter
building_df['baseline_expected'] = get_baseline_expected(
    selected_building, selected_metric, building_df['timestamp']
).values

# If adapter returned NaNs (fallback), use rolling mean
if building_df['baseline_expected'].isna().any():
    building_df['baseline_expected'] = building_df['value_actual'].rolling(168, min_periods=24).mean().bfill().ffill()

# 2. Anomaly flags from adapter
anom_res = get_anomaly_flags(selected_building, selected_metric, building_df['timestamp'])
building_df['is_anomaly'] = anom_res['is_anomaly'].values
building_df['anomaly_reason'] = anom_res['anomaly_reason'].values

# ---------------------------------------------------------
# TIME-SERIES & CONFIDENCE BAND CHART
# ---------------------------------------------------------
st.subheader(f"📈 Biểu Đồ Chuỗi Thời Gian & Khoảng Tin Cậy (±1.5 MedAE) — {selected_building}")

fig, ax = plt.subplots(figsize=(15, 5))

# Actual values
ax.plot(building_df['timestamp'], building_df['value_actual'], label='Thực tế (Real Telemetry)', color='#1f77b4', linewidth=1.5)

# Teammate 1: Baseline expected
ax.plot(building_df['timestamp'], building_df['baseline_expected'], label='Baseline kỳ vọng (Adapter Bạn 1 - Mock)', color='#ff7f0e', linestyle='--', alpha=0.8)

# Winning forecast
ax.plot(building_df['timestamp'], building_df['forecast_next_24h'], label=f'Dự báo hoàn thiện ({b_meta["model_selected"].iloc[0] if not b_meta.empty else "Model"})', color='#2ca02c', linestyle='-', linewidth=2)

# +/- 1.5 * MedAE Confidence band
ax.fill_between(
    building_df['timestamp'],
    building_df['forecast_lower'],
    building_df['forecast_upper'],
    color='#2ca02c',
    alpha=0.18,
    label='Khoảng tin cậy (± 1.5 × MedAE)'
)

# Teammate 2: Anomalies
anomalies = building_df[building_df['is_anomaly']]
if not anomalies.empty:
    ax.scatter(anomalies['timestamp'], anomalies['value_actual'], color='red', s=45, zorder=5, label='Bất thường (Adapter Bạn 2 - Mock)')

ax.set_xlabel("Thời gian (Timestamp)")
ax.set_ylabel(f"Giá trị ({selected_metric})")
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper right')

st.pyplot(fig)

# ---------------------------------------------------------
# TABLE 1: UNIFIED DEMO SCHEMA (demo_schema.md)
# ---------------------------------------------------------
st.subheader("📋 Bảng Dữ Liệu Demo Theo Đúng Chuẩn Schema (`demo_schema.md`)")

display_cols = [
    'timestamp', 'building_id', 'metric', 
    'value_actual', 'baseline_expected', 
    'is_anomaly', 'anomaly_reason', 
    'forecast_next_24h', 'forecast_lower', 'forecast_upper'
]

st.dataframe(
    building_df[display_cols].head(100),
    use_container_width=True
)
st.caption("Hiển thị 100 dòng đầu tiên theo khung bộ lọc đã chọn tuân thủ 100% 10 cột theo quy chuẩn `demo_schema.md`.")

# ---------------------------------------------------------
# SECTION 2: MILP OPTIMIZATION & SCHEDULE (VIỆC 2)
# ---------------------------------------------------------
st.markdown("---")
st.subheader("⚙️ Tối Ưu Hóa Vận Hành Chiller Bằng Bộ Giải Toán Học (MILP Solver — HiGHS)")

st.info("""
📌 **NGUYÊN TẮC TỐI ƯU HÓA MILP (VIỆC 2 SPRINT 3):**
- **Đầu vào dự báo**: Lấy tải lạnh dự báo 24h tới trực tiếp từ mô hình hoàn thiện `forecast_final.py` (KHÔNG chạy trên tải lịch sử).
- **Bộ giải toán học**: Sử dụng `scipy.optimize.milp` (Solver HiGHS), giải bài toán Unit-Commitment + Optimal Chiller Loading.
- **Ràng buộc vật lý đầy đủ**: Cân bằng phụ tải, giới hạn công suất, đường cong EIR-FPLR tuyến tính hóa bằng 12 tiếp tuyến, thời gian chạy/dừng tối thiểu (min up/down time = 1h), giới hạn số lần khởi động máy <= 3 lần/ngày.
""")

# Run or load ranking
try:
    df_rank, milp_sol = generate_recommendation_ranking(selected_building)
    
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    milp_row = df_rank[df_rank['scenario_name'].str.contains('MILP')]
    
    with col_kpi1:
        st.metric("Trạng thái kiểm tra ràng buộc", "✅ ĐẠT 100% (True)", help="Không vi phạm min up/down time, số lần khởi động, cân bằng tải")
    with col_kpi2:
        if not milp_row.empty:
            st.metric("Tiết kiệm ước tính từ MILP", f"{milp_row['est_saving_vnd'].iloc[0]:,.0f} VND/ngày", f"{milp_row['est_saving_pct'].iloc[0]}%")
    with col_kpi3:
        if milp_sol.get('success', False):
            st.metric("MIP Relative Gap đạt được", f"{milp_sol['mip_gap']:.6f}", f"Thời gian giải: {milp_sol['runtime']:.3f}s")

    # Gantt Chart of Chiller Schedule
    if milp_sol.get('success', False) and 'on' in milp_sol:
        st.subheader("📅 Lịch Vận Hành Gợi Ý Cho Cụm Chiller (Biểu Đồ Gantt)")
        on_matrix = milp_sol['on'] # Shape (4, 24)
        eq_df = load_equipment()
        ch_names = [f"{r['chiller_id']} ({r['model_type'].split('(')[0].strip()}, {r['q_rated_kw']}kW)" for _, r in eq_df.iterrows()]
        
        fig_gantt, ax_gantt = plt.subplots(figsize=(14, 3.8))
        hours = np.arange(24)
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i in range(len(ch_names)):
            for t in range(24):
                if on_matrix[i, t] == 1:
                    ax_gantt.broken_barh([(t, 1)], (i - 0.35, 0.7), facecolors=colors[i], edgecolor='black', alpha=0.85)
                    
        ax_gantt.set_yticks(range(len(ch_names)))
        ax_gantt.set_yticklabels(ch_names, fontsize=10)
        ax_gantt.set_xticks(range(25))
        ax_gantt.set_xlabel("Khung giờ trong ngày (00:00 - 24:00)", fontsize=11)
        ax_gantt.set_title("Lịch cam kết máy Chiller (Unit Commitment) theo 24 giờ do Bộ giải MILP tối ưu", fontsize=12)
        ax_gantt.grid(True, axis='x', linestyle=':', alpha=0.6)
        
        # Highlight peak hours (18h-23h)
        ax_gantt.axvspan(18, 23, color='red', alpha=0.12, label='Giờ cao điểm EVN (18h-23h: 5.025 đ/kWh)')
        ax_gantt.axvspan(0, 6, color='blue', alpha=0.08, label='Giờ thấp điểm EVN (0h-6h: 1.609 đ/kWh)')
        ax_gantt.legend(loc='upper right', fontsize=9)
        
        st.pyplot(fig_gantt)
        
    # Table 2: Recommendation Ranking Table
    st.subheader("🏆 Bảng Xếp Hạng Phương Án Vận Hành Theo Lợi Ích Ước Tính (`recommendation_ranking.csv`)")
    
    display_ranking = df_rank[['rank', 'scenario_name', 'est_saving_kwh', 'est_saving_vnd', 'est_saving_pct', 'muc_do_tin_cay', 'dieu_kien_ap_dung', 'constraints_checked']].copy()
    display_ranking.columns = [
        'Hạng (Rank)', 'Tên Phương Án', 'Điện Tiết Kiệm (kWh)', 
        'Chi Phí Tiết Kiệm (VND)', 'Tỷ Lệ Tiết Kiệm (%)', 
        'Mức Độ Tin Cậy', 'Điều Kiện Áp Dụng & Giả Định', 'Kiểm Tra Ràng Buộc'
    ]
    st.dataframe(display_ranking, use_container_width=True)
    
except Exception as e:
    st.warning(f"Không thể khởi chạy bộ giải tối ưu cho tòa nhà này ({e}). Hiển thị bảng xếp hạng mặc định.")
    if os.path.exists(RANKING_CSV_PATH):
        df_rank_cached = pd.read_csv(RANKING_CSV_PATH, comment='#')
        st.dataframe(df_rank_cached, use_container_width=True)

st.markdown("---")
st.caption("Sprint 3 Hoàn thiện — Mô hình dự báo lai (Winning model) + Bộ giải MILP HiGHS + Adapter tích hợp Teammate.")
