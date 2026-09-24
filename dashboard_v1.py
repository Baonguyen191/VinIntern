import os
import pandas as pd
import numpy as np
import streamlit as st

from src.anomaly.rules import flag_hours, run_rule_detector
from src.io.normalized_loader import load_telemetry

# Setup page layout
st.set_page_config(
    page_title="Năng lượng & M&E Thông minh — Khung Demo Sprint 1",
    page_icon="⚡",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_PATH = os.path.join(BASE_DIR, 'baseline_forecast_report.csv')
FORECAST_PATH = os.path.join(BASE_DIR, 'data_normalized', 'forecast_test_results.parquet')

@st.cache_data
def load_baseline_report():
    if os.path.exists(REPORT_PATH):
        return pd.read_csv(REPORT_PATH)
    return pd.DataFrame()

@st.cache_data
def detect_anomalies(building_id, metric):
    source = "M1" if metric == "power_active_kw" else "M2"
    df = load_telemetry(source, columns=["entity_id", "ts", "day_type", metric], entities=[building_id])
    return run_rule_detector(df, metric, building_id)

@st.cache_data
def load_forecast_data():
    if os.path.exists(FORECAST_PATH):
        df = pd.read_parquet(FORECAST_PATH)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    return pd.DataFrame()

# ---------------------------------------------------------
# HEADER & VERIFICATION STATUS
# ---------------------------------------------------------
st.title("⚡ Khung Demo Tích Hợp Sprint 1 — Pipeline Năng Lượng & M&E")
st.markdown("""
**Hệ thống Dự báo & Gợi ý Tối ưu** | Pipeline 4 bước: *Baseline → Bất thường → Dự báo → Gợi ý tối ưu*
""")

# Bảng phân định RÕ RÀNG dữ liệu THẬT vs MOCK
st.info("""
📌 **PHÂN ĐỊNH TRẠNG THÁI DỮ LIỆU TRÊN DASHBOARD (SPRINT 1):**
- 🟢 **DỮ LIỆU THẬT (Dự báo của tôi)**: `value_actual`, `forecast_next_24h`, `forecast_lower`, `forecast_upper` (được tính từ mô hình Hour-of-Week Profile / Seasonal Naive trên 20% tập test cuối chuỗi).
- 🟢 **DỮ LIỆU THẬT (Bạn 2 — Bất thường)**: `is_anomaly`, `anomaly_reason` từ bộ phát hiện theo luật đã thống nhất: dải profile trượt 8 tuần (k = 3), CUSUM tải nền đêm (điện), lớp chất lượng dữ liệu (mất mẫu, cảm biến đơ); cảnh báo liên tiếp được gộp.
- 🟡 **DỮ LIỆU MOCK / PLACEHOLDER (Chờ Bạn 1)**:
  - `baseline_expected`: Giả lập bằng *Trung bình trượt 7 ngày (Rolling 7-day mean)*.
""")

# Load Data
df_report = load_baseline_report()
df_fc = load_forecast_data()

if df_fc.empty:
    st.error("Không tìm thấy file `forecast_test_results.parquet`. Vui lòng chạy `python scripts/build_baseline_forecast.py` trước.")
    st.stop()

# ---------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------
# Calendar popup: on short windows BaseWeb flips it above the input and its month/week navigation
# ends up off-screen. Pin it to the top of the viewport and let it scroll inside instead.
st.markdown("""
<style>
div[data-baseweb="popover"]:has(div[data-baseweb="calendar"]) {
    inset: 8px auto auto 16px !important;
    transform: none !important;
    max-height: calc(100vh - 16px);
    overflow-y: auto;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.header("🔍 Bộ Lọc Dữ Liệu")
date_box = st.sidebar.container()  # filled below, once the building's date range is known

# Select Metric
metric_options = sorted(df_fc['metric'].unique())
selected_metric = st.sidebar.selectbox("Chọn Chỉ Số (Metric):", metric_options, index=0)

# Select Building
filtered_by_metric = df_fc[df_fc['metric'] == selected_metric]
building_options = sorted(filtered_by_metric['building_id'].unique())
selected_building = st.sidebar.selectbox("Chọn Tòa Nhà (Building ID):", building_options, index=0)

# Filter dataset for selected building & metric
building_df = filtered_by_metric[filtered_by_metric['building_id'] == selected_building].sort_values('timestamp').copy()

# Date range slider
min_ts = building_df['timestamp'].min()
max_ts = building_df['timestamp'].max()

# The date picker is drawn in date_box, reserved at the top of the sidebar: lower down, the
# calendar popup opens upward and its month/week navigation gets clipped on short windows.
date_box.subheader("📅 Khoảng Thời Gian (Tập Test)")
date_range = date_box.date_input(
    "Chọn khoảng ngày:",
    value=(min_ts.date(), max_ts.date()),
    min_value=min_ts.date(),
    max_value=max_ts.date(),
    format="DD/MM/YYYY",
)

if len(date_range) == 2:
    start_date, end_date = date_range
    mask = (building_df['timestamp'].dt.date >= start_date) & (building_df['timestamp'].dt.date <= end_date)
    building_df = building_df[mask]

# ---------------------------------------------------------
# MOCK PLACEHOLDER GENERATION (FOR TEAMMATE 1 & 2)
# ---------------------------------------------------------
# [PLACEHOLDER TEAMMATE 1 - BASELINE]
# Sprint 1 mock logic: Rolling 7-day mean
building_df['baseline_expected'] = building_df['value_actual'].rolling(window=168, min_periods=24).mean()

# [TEAMMATE 2 - ANOMALY] Bộ phát hiện thật: luật đã thống nhất (src/anomaly/rules.py)
# Chạy trên cả chuỗi năm của tòa nhà (baseline trượt 8 tuần cần lịch sử), rồi gắn cờ cho tập test.
alerts = detect_anomalies(selected_building, selected_metric)
building_df['is_anomaly'], building_df['anomaly_reason'] = flag_hours(building_df['timestamp'], alerts)

# ---------------------------------------------------------
# TOP METRICS CARD (REAL BASELINE REPORT EVALUATION)
# ---------------------------------------------------------
st.subheader(f"📊 Đánh Giá Hiệu Năng Mô Hình Naive — {selected_building} ({selected_metric})")

if not df_report.empty:
    b_report = df_report[(df_report['building_id'] == selected_building) & (df_report['metric'] == selected_metric)]
    col1, col2, col3, col4 = st.columns(4)
    
    sn_row = b_report[b_report['model'] == 'seasonal_naive_168h']
    hw_row = b_report[b_report['model'] == 'hour_of_week_profile']
    
    with col1:
        sn_mae = sn_row['MAE'].values[0] if not sn_row.empty else "N/A"
        st.metric("Seasonal Naive (168h) MAE", f"{sn_mae} kW" if sn_mae != "N/A" else "N/A")
    with col2:
        sn_mape = sn_row['MAPE'].values[0] if not sn_row.empty else "N/A"
        st.metric("Seasonal Naive (168h) MAPE", f"{sn_mape}%" if sn_mape != "N/A" else "N/A")
    with col3:
        hw_mae = hw_row['MAE'].values[0] if not hw_row.empty else "N/A"
        st.metric("Hour-of-Week Profile MAE", f"{hw_mae} kW" if hw_mae != "N/A" else "N/A")
    with col4:
        hw_mape = hw_row['MAPE'].values[0] if not hw_row.empty else "N/A"
        st.metric("Hour-of-Week Profile MAPE", f"{hw_mape}%" if hw_mape != "N/A" else "N/A")

# ---------------------------------------------------------
# MAIN TIME-SERIES CHART
# ---------------------------------------------------------
st.subheader(f"📈 Biểu Đồ Chuỗi Thời Gian & Dự Báo 24h — {selected_building}")

# Interactive Altair chart: hover shows the time and every value; dragging on the chart selects
# a time range that filters the table below.
import altair as alt

SERIES_LABELS = {
    'value_actual': 'Thực tế (Real Actual)',
    'baseline_expected': 'Baseline kỳ vọng (Mock Bạn 1)',
    'forecast_next_24h': 'Dự báo (Real Forecast - Me)',
}
plot_df = building_df.copy()
# Naive timestamps are sent as UTC ("Z") and drawn on a UTC scale, so the axis shows the data's
# own clock time and the selected range comes back without a timezone shift.
plot_df['ts_utc'] = plot_df['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
plot_df['thoi_gian'] = plot_df['timestamp'].dt.strftime('%d/%m/%Y %H:%M')

x = alt.X('ts_utc:T', title='Thời gian (Timestamp)', scale=alt.Scale(type='utc'),
          axis=alt.Axis(format='%d/%m %H:%M', labelAngle=0))
hover = alt.selection_point(fields=['ts_utc'], nearest=True, on='pointerover', empty=False)
brush = alt.selection_interval(encodings=['x'], name='khoang_thoi_gian')

band = alt.Chart(plot_df).mark_area(opacity=0.15, color='#2ca02c').encode(
    x=x, y=alt.Y('forecast_lower:Q', title=f'Giá trị ({selected_metric})'), y2='forecast_upper:Q',
).add_params(brush)
lines_df = plot_df.melt(id_vars=['ts_utc'], value_vars=list(SERIES_LABELS), var_name='chuoi', value_name='gia_tri')
lines_df['chuoi'] = lines_df['chuoi'].map(SERIES_LABELS)
lines = alt.Chart(lines_df).mark_line(strokeWidth=1.6).encode(
    x=x, y='gia_tri:Q',
    color=alt.Color('chuoi:N', title=None, legend=alt.Legend(orient='top', labelLimit=300),
                    scale=alt.Scale(domain=list(SERIES_LABELS.values()), range=['#1f77b4', '#ff7f0e', '#2ca02c'])),
    strokeDash=alt.StrokeDash('chuoi:N', legend=None,
                              scale=alt.Scale(domain=list(SERIES_LABELS.values()), range=[[1, 0], [6, 4], [2, 3]])),
)
anomaly_points = alt.Chart(plot_df[plot_df['is_anomaly']]).mark_circle(color='red', size=45, opacity=0.9).encode(
    x=x, y='value_actual:Q',
)
hover_rule = alt.Chart(plot_df).mark_rule(color='#888888').encode(
    x=x,
    opacity=alt.condition(hover, alt.value(0.8), alt.value(0)),
    tooltip=[
        alt.Tooltip('thoi_gian:N', title='Thời gian'),
        alt.Tooltip('value_actual:Q', title='Thực tế', format='.3f'),
        alt.Tooltip('baseline_expected:Q', title='Baseline (mock)', format='.3f'),
        alt.Tooltip('forecast_next_24h:Q', title='Dự báo', format='.3f'),
        alt.Tooltip('forecast_lower:Q', title='P10', format='.3f'),
        alt.Tooltip('forecast_upper:Q', title='P90', format='.3f'),
        alt.Tooltip('is_anomaly:N', title='Bất thường'),
        alt.Tooltip('anomaly_reason:N', title='Lý do'),
    ],
).add_params(hover)

chart = alt.layer(band, lines, anomaly_points, hover_rule).properties(height=420)
chart_event = st.altair_chart(chart, width='stretch', on_select='rerun',
                              selection_mode=['khoang_thoi_gian'], key='main_chart')
st.caption("🔴 Chấm đỏ: giờ bất thường (Bạn 2). Rê chuột để xem thời gian và giá trị; "
           "kéo chuột trên biểu đồ để chọn khoảng thời gian cho bảng bên dưới (nhấp đúp để bỏ chọn).")

# ---------------------------------------------------------
# UNIFIED DEMO SCHEMA TABLE
# ---------------------------------------------------------
st.subheader("📋 Bảng Dữ Liệu Khung Theo Schema Thống Nhất (`demo_schema.md`)")

display_cols = [
    'timestamp', 'building_id', 'metric', 
    'value_actual', 'baseline_expected', 
    'is_anomaly', 'anomaly_reason', 
    'forecast_next_24h', 'forecast_lower', 'forecast_upper'
]

def _selected_range(event):
    """Time range dragged on the chart, or None. Vega-Lite returns epoch ms (UTC scale)."""
    try:
        picked = event.selection.get('khoang_thoi_gian', {}).get('ts_utc')
    except AttributeError:
        return None
    if not picked or len(picked) != 2:
        return None
    unit = 'ms' if isinstance(picked[0], (int, float)) else None
    t0, t1 = sorted(pd.to_datetime(picked, unit=unit, utc=True).tz_localize(None))
    return t0, t1

table_df = building_df
picked_range = _selected_range(chart_event)
if picked_range:
    t0, t1 = picked_range
    table_df = table_df[(table_df['timestamp'] >= t0) & (table_df['timestamp'] <= t1)]

only_anomaly = st.checkbox("Chỉ hiện giờ bất thường", value=False)
if only_anomaly:
    table_df = table_df[table_df['is_anomaly']]

st.dataframe(
    table_df[display_cols],
    width='stretch',
    hide_index=True,
    column_config={'timestamp': st.column_config.DatetimeColumn('timestamp', format='DD/MM/YYYY HH:mm')},
)

scope = (f"khoảng chọn trên biểu đồ {t0:%d/%m/%Y %H:%M} → {t1:%d/%m/%Y %H:%M}" if picked_range
         else "toàn bộ khoảng ngày ở thanh bên")
st.caption(f"{len(table_df)} dòng, {int(table_df['is_anomaly'].sum())} giờ bất thường — {scope}.")
