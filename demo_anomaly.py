"""Demo: actual vs baseline, with detected anomalies.

Run from the repo root:
    python main.py --pipeline anomaly-cases     # once, builds results/anomaly_cases
    streamlit run demo_anomaly.py

Detectors run live on the selected series, so method, thresholds, budget and
date range can be changed in the sidebar. "Dữ liệu gốc" is the real BDG2
series; "Có ca giả lập" adds the injected cases and shows their labels.
"""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.anomaly.detectors import (
    deviation_scores,
    efficiency_scores,
    fuse_scores,
    hourly_features,
    if_score_features,
    isolation_forest_scores,
    lightgbm_scores,
    normalize_to_train,
    profile_band_scores,
)
from src.anomaly.events import (
    apply_budget,
    classify_hourly,
    interval_events,
    points_to_events,
    split_point_interval,
    suppress_sensor_faults,
)
from src.anomaly.quality import regularize, run_quality
from src.core.metrics import cv_rmse, nmbe
from src.io import bdg2_telemetry

CASE_DIR = Path("results/anomaly_cases")

BASELINE_CSV = Path("results/bdg2/statistical/Rat_office_Colby/our_cleaned_predictions.csv")
TRAIN_END = pd.Timestamp("2017-01-01")
KEY = "power_active_kw"

RULES, IFOREST, LGBM = "Luật (profile band)", "Isolation Forest", "LightGBM"
LGBM_IF, LGBM_IF_LAG = "LightGBM + điểm IF (cùng giờ)", "LightGBM + điểm IF (24h trước)"
FUSION = "Gộp điểm LightGBM + IF (max)"
METHODS = (RULES, IFOREST, LGBM, FUSION, LGBM_IF, LGBM_IF_LAG)
METHOD_NOTES = {
    RULES: "Baseline = median 4 tuần gần nhất cùng giờ và loại ngày. Bất thường khi robust z vượt ngưỡng.",
    IFOREST: "Isolation Forest học trên dữ liệu năm 2016, chấm điểm từng giờ. Không có baseline riêng: "
             "đường nét đứt là profile band để tham chiếu. Bất thường khi điểm vượt phân vị chọn trên dữ liệu học.",
    LGBM: "Baseline = LightGBM dự đoán tải (giờ, thứ, loại ngày, nhiệt độ, tải tuần trước) học trên năm 2016. "
          "Bất thường khi robust z của phần dư vượt ngưỡng.",
    LGBM_IF: "LightGBM có thêm feature là điểm Isolation Forest của chính giờ đó. Điểm IF tính từ tải giờ đó, "
             "nên baseline có thể bám theo bất thường và che bớt nó.",
    LGBM_IF_LAG: "LightGBM có thêm feature là điểm Isolation Forest trung bình 24 giờ trước đó "
                 "(không dùng thông tin của giờ đang xét).",
    FUSION: "Baseline = LightGBM. Điểm = max của robust z phần dư LightGBM và điểm Isolation Forest, "
            "mỗi điểm được chuẩn hóa theo phân bố trên dữ liệu học 2016. IF không đi vào baseline.",
}

TYPE_VI = {
    "spike": "Tăng vọt",
    "sustained_high": "Tải cao kéo dài",
    "after_hours": "Chạy ngoài giờ",
    "efficiency_drop": "Giảm hiệu suất",
    "level_shift": "Tải nền tăng",
    "data_missing": "Mất dữ liệu",
    "stuck_sensor": "Cảm biến đứng yên",
    "out_of_range": "Ngoài dải đo",
}
COLORS = {"actual": "#1f77b4", "baseline": "#7f7f7f", "anomaly": "#d62728", "label": "#ff7f0e", "dq": "#9467bd"}
SCOPE_VI = {"point": "Điểm", "interval": "Khoảng"}

alt.data_transformers.disable_max_rows()
st.set_page_config(page_title="Demo phát hiện bất thường", layout="wide")


# ---------------------------------------------------------------- data

@st.cache_data
def load_case_series() -> pd.DataFrame:
    path = CASE_DIR / "cases_series.parquet"
    _ = path.stat().st_mtime if path.exists() else 0
    return pd.read_parquet(path)


@st.cache_data
def load_labels() -> pd.DataFrame:
    path = CASE_DIR / "cases.csv"
    _ = path.stat().st_mtime if path.exists() else 0
    return pd.read_csv(path, parse_dates=["start", "end"])


@st.cache_data
def load_cooling(entity_id: str) -> pd.Series | None:
    m2 = bdg2_telemetry.load_cooling(buildings=[entity_id])
    if m2.empty:
        return None
    return m2.set_index("ts")["value"].astype(float)


@st.cache_data
def load_temperature(entity_id: str) -> pd.Series:
    m1 = bdg2_telemetry.load_electricity(usage="Office", buildings=[entity_id])
    return m1.set_index("ts")["temperature"].astype(float)


def _entity_frame(entity_id: str, column: str) -> tuple[pd.Series, pd.Series]:
    g = load_case_series().query("entity_id == @entity_id").set_index("ts")
    return g[column].astype(float), g["day_type"]


@st.cache_data
def quality_events(entity_id: str, column: str) -> pd.DataFrame:
    s, _ = _entity_frame(entity_id, column)
    return run_quality(s, entity_id, KEY)


@st.cache_data
def hourly_scores(entity_id: str, column: str, method: str) -> pd.DataFrame:
    """score, baseline, scale, value, day_type per hour for the chosen method."""
    s, day_type = _entity_frame(entity_id, column)
    train = (s.index < TRAIN_END)
    band = profile_band_scores(s, day_type)
    if method == RULES:
        sc = band
    else:
        if_scores = isolation_forest_scores(hourly_features(s, day_type, band), train)
        if method == IFOREST:
            sc = band.assign(score=if_scores, scale=np.nan)
        else:
            extra = {
                LGBM: None,
                LGBM_IF: if_score_features(if_scores, lagged=False),
                LGBM_IF_LAG: if_score_features(if_scores, lagged=True),
            }.get(method)
            sc = lightgbm_scores(
                s, day_type, load_temperature(entity_id), band["baseline"], train, extra_features=extra
            )
            if method == FUSION:
                sc["score"] = fuse_scores({"lightgbm": sc["score"], "iforest": if_scores}, train, how="max")
    return sc.assign(day_type=day_type)


def detect(
    entity_id: str,
    sc: pd.DataFrame,
    quality: pd.DataFrame,
    threshold: float,
    budget: float,
    method: str,
    k_interval: float = 2.5,
    window: int = 6,
    point_max_h: int = 2,
) -> pd.DataFrame:
    """Point events (score >= threshold, <= point_max_h hours) and interval events
    (longer point events, or `window`-hour mean score >= k_interval), merged and budgeted."""
    test = sc.index >= TRAIN_END
    score = sc["score"].where(test)
    point = suppress_sensor_faults(points_to_events(score, threshold, entity_id, method), quality)
    # Isolation Forest scores are not z; put them on the z scale for the sustained rule.
    z = normalize_to_train(sc["score"], ~test).where(test) if method == IFOREST else score
    sustained = suppress_sensor_faults(interval_events(z, window, k_interval, entity_id, f"{method} (khoảng)"), quality)
    ev = split_point_interval(point, sustained, point_max_h)
    if ev.empty:
        return ev.assign(type=pd.Series(dtype=object), scope=pd.Series(dtype=object),
                         value=np.nan, baseline=np.nan, deviation_pct=np.nan, mean_dev_pct=np.nan)
    ev["type"] = classify_hourly(ev, sc["day_type"])
    n_days = (sc.index.max() - TRAIN_END).days + 1
    ev = apply_budget(ev, n_days, budget)
    peak = sc.loc[ev["peak_ts"]]
    ev["value"] = peak["value"].to_numpy()
    ev["baseline"] = peak["baseline"].to_numpy()
    ev["deviation_pct"] = (ev["value"] - ev["baseline"]) / ev["baseline"] * 100
    # Mean deviation over the whole event: the useful number for an interval.
    ev["mean_dev_pct"] = [
        (sc.loc[s:t, "value"].sum() / sc.loc[s:t, "baseline"].sum() - 1) * 100
        for s, t in zip(ev["start"], ev["end"])
    ]
    return ev


def method_threshold(sc: pd.DataFrame, method: str, k: float, if_q: float) -> float:
    if method == IFOREST:
        return float(np.nanquantile(sc.loc[sc.index < TRAIN_END, "score"], if_q))
    return k


def n_hit(events: pd.DataFrame, labels: pd.DataFrame) -> int:
    return sum(((events["start"] <= t) & (events["end"] >= s)).any() for s, t in zip(labels["start"], labels["end"]))


def in_window(df: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame:
    return df[(df["start"] <= hi) & (df["end"] >= lo)]


# ---------------------------------------------------------------- charts

def span_layer(spans: pd.DataFrame, color: str, tooltip: list[str], step: pd.Timedelta) -> alt.Chart:
    spans = spans.assign(end_plot=spans["end"] + step)
    return alt.Chart(spans).mark_rect(opacity=0.18, color=color).encode(
        x="start:T", x2="end_plot:T", tooltip=tooltip
    )


def actual_vs_baseline_chart(
    df: pd.DataFrame,
    x: str,
    actual: str,
    baseline: str,
    anomalies: pd.DataFrame,
    lo: str | None = None,
    hi: str | None = None,
    spans: list[alt.Chart] = (),
    y_title: str = "kW",
    flagged: pd.DataFrame | None = None,
) -> alt.Chart:
    long = df.melt(id_vars=[x], value_vars=[actual, baseline], var_name="series", value_name="val")
    long["series"] = long["series"].map({actual: "Thực tế", baseline: "Baseline"})
    color = alt.Color(
        "series:N",
        scale=alt.Scale(domain=["Thực tế", "Baseline"], range=[COLORS["actual"], COLORS["baseline"]]),
        legend=alt.Legend(title=None, orient="top"),
    )
    layers = list(spans)
    if lo and hi and df[lo].notna().any():
        layers.append(alt.Chart(df).mark_area(opacity=0.15, color=COLORS["baseline"]).encode(
            x=f"{x}:T", y=f"{lo}:Q", y2=f"{hi}:Q"
        ))
    layers.append(alt.Chart(long).mark_line(strokeWidth=1.2).encode(
        x=alt.X(f"{x}:T", title=None),
        y=alt.Y("val:Q", title=y_title),
        color=color,
        strokeDash=alt.condition(alt.datum.series == "Baseline", alt.value([4, 3]), alt.value([1, 0])),
        tooltip=[alt.Tooltip(f"{x}:T"), "series:N", alt.Tooltip("val:Q", format=",.1f")],
    ))
    if flagged is not None and not flagged.empty:
        layers.append(alt.Chart(flagged).mark_circle(size=22, color=COLORS["anomaly"], opacity=0.8).encode(
            x=f"{x}:T", y=f"{actual}:Q",
            tooltip=[alt.Tooltip(f"{x}:T"), alt.Tooltip("score:Q", title="Điểm", format=".3f")],
        ))
    anomalies = anomalies[anomalies["scope"] == "point"] if "scope" in anomalies else anomalies
    if not anomalies.empty:
        layers.append(alt.Chart(anomalies).mark_point(
            filled=True, size=90, color=COLORS["anomaly"], shape="triangle-up"
        ).encode(
            x="peak_ts:T",
            y="value:Q",
            tooltip=[
                alt.Tooltip("loai:N", title="Loại"),
                alt.Tooltip("start:T", title="Bắt đầu"),
                alt.Tooltip("end:T", title="Kết thúc"),
                alt.Tooltip("value:Q", title="Thực tế", format=",.1f"),
                alt.Tooltip("baseline:Q", title="Baseline", format=",.1f"),
                alt.Tooltip("deviation_pct:Q", title="Lệch %", format="+.1f"),
            ],
        ))
    return alt.layer(*layers).properties(height=380).interactive(bind_y=False)


def score_chart(
    df: pd.DataFrame, x: str, y: str, threshold: float, title: str, bars: bool = True, two_sided: bool = True
) -> alt.Chart:
    base = alt.Chart(df).encode(x=alt.X(f"{x}:T", title=None))
    mark = base.mark_bar() if bars else base.mark_line(color=COLORS["actual"])
    test = f"abs(datum.{y}) >= {threshold}" if two_sided else f"datum.{y} >= {threshold}"
    body = mark.encode(
        y=alt.Y(f"{y}:Q", title=title),
        color=alt.condition(test, alt.value(COLORS["anomaly"]), alt.value(COLORS["baseline"]))
        if bars else alt.value(COLORS["actual"]),
        tooltip=[alt.Tooltip(f"{x}:T"), alt.Tooltip(f"{y}:Q", format=".3f")],
    )
    lines = [threshold, -threshold] if two_sided else [threshold]
    rules = alt.Chart(pd.DataFrame({"t": lines})).mark_rule(color=COLORS["anomaly"], strokeDash=[4, 3]).encode(y="t:Q")
    return alt.layer(body, rules).properties(height=180)


def show_events_table(target, events: pd.DataFrame) -> None:
    cols = ["loai", "start", "end", "duration_h", "value", "baseline", "deviation_pct", "peak_score"]
    if "scope" in events:
        events = events.assign(pham_vi=events["scope"].map(SCOPE_VI))
        cols = ["pham_vi", *cols[:4], "mean_dev_pct", *cols[4:]]
    target.dataframe(
        events[cols]
        .sort_values("start")
        .rename(columns={"pham_vi": "Phạm vi", "loai": "Loại", "start": "Bắt đầu", "end": "Kết thúc",
                         "duration_h": "Giờ", "mean_dev_pct": "Lệch TB %", "value": "Thực tế (đỉnh)",
                         "baseline": "Baseline (đỉnh)", "deviation_pct": "Lệch đỉnh %", "peak_score": "Điểm"}),
        hide_index=True, width="stretch",
    )


# ---------------------------------------------------------------- sidebar

st.title("So sánh dữ liệu thật với baseline và phát hiện bất thường")

if not (CASE_DIR / "cases_series.parquet").exists():
    st.error("Chưa có bộ ca. Chạy trước: `python main.py --pipeline anomaly-cases`")
    st.stop()

series = load_case_series()
labels = load_labels()
entities = sorted(series["entity_id"].unique())
year_lo, year_hi = series["ts"].min().date(), series["ts"].max().date()

with st.sidebar:
    st.header("Thiết lập")
    mode = st.radio("Dữ liệu", ["Dữ liệu gốc", "Có ca giả lập"], index=1,
                    help="Dữ liệu gốc: chuỗi BDG2 thật. Có ca giả lập: thêm lỗi đã biết để kiểm tra detector.")
    injected = mode == "Có ca giả lập"
    column = "value" if injected else "value_clean"
    default_idx = entities.index("Rat_office_Colby") if "Rat_office_Colby" in entities else 0
    entity = st.selectbox("Tòa nhà", entities, index=default_idx)

    method = st.radio("Cách phát hiện", METHODS)
    if method == IFOREST:
        if_q = st.slider("Ngưỡng điểm (phân vị trên dữ liệu học)", 0.95, 0.999, 0.99, 0.001, format="%.3f")
        k = 4.0
    else:
        k = st.slider("Ngưỡng robust z", 2.0, 8.0, 4.0, 0.5)
        if_q = 0.99
    budget = st.select_slider("Ngân sách cảnh báo / thiết bị / ngày", [0.02, 0.05, 0.1, 0.2, 0.5], value=0.2)

    st.subheader("Điểm và khoảng")
    point_max_h = st.slider("Bất thường tại điểm: dài tối đa (giờ)", 1, 6, 2,
                            help="Sự kiện dài hơn mức này được xếp là bất thường trên khoảng.")
    window = st.slider("Bất thường trên khoảng: cửa sổ trung bình (giờ)", 3, 24, 6)
    k_interval = st.slider("Bất thường trên khoảng: ngưỡng z trung bình", 1.5, 5.0, 2.5, 0.5,
                           help="Bắt các đoạn lệch vừa phải nhưng kéo dài, dù không giờ nào vượt ngưỡng điểm.")
    show_scopes = st.multiselect("Hiển thị", ["Điểm", "Khoảng"], default=["Điểm", "Khoảng"])

    st.divider()
    ent_labels_all = labels[labels["entity_id"] == entity]
    first = ent_labels_all["start"].min() if injected and not ent_labels_all.empty else TRAIN_END
    default_lo = max(first.normalize() - pd.Timedelta(days=3), pd.Timestamp(year_lo)).date()
    default_hi = min(pd.Timestamp(default_lo) + pd.Timedelta(days=21), pd.Timestamp(year_hi)).date()
    date_lo, date_hi = st.slider(
        "Khoảng ngày xem", min_value=year_lo, max_value=year_hi,
        value=(default_lo, default_hi), format="DD/MM/YYYY", key=f"range_{entity}_{injected}",
    )
    st.caption(f"Dữ liệu trước {TRAIN_END:%d/%m/%Y} dùng làm giai đoạn học, không cảnh báo.")

lo = pd.Timestamp(date_lo)
hi = pd.Timestamp(date_hi) + pd.Timedelta(hours=23)

tab_hourly, tab_eff, tab_daily = st.tabs(
    ["Theo giờ: điện tòa nhà", "Hiệu suất: điện theo tải lạnh", "Theo ngày: CSV baseline"]
)

# ---------------------------------------------------------------- hourly tab
with tab_hourly:
    quality = quality_events(entity, column)
    with st.spinner(f"Đang chạy {method}..."):
        sc = hourly_scores(entity, column, method)
    threshold = method_threshold(sc, method, k, if_q)
    events = detect(entity, sc, quality, threshold, budget, method, k_interval, window, point_max_h)
    events = events[events["scope"].map(SCOPE_VI).isin(show_scopes)]
    ent_labels = ent_labels_all[ent_labels_all["group"].isin(["A", "DQ"])] if injected else labels.iloc[0:0]

    st.info(METHOD_NOTES[method])
    test = sc[sc.index >= TRAIN_END].dropna(subset=["value", "baseline"])
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("CV(RMSE) baseline", f"{cv_rmse(test['value'].to_numpy(), test['baseline'].to_numpy()):.1f}%")
    m2.metric("NMBE baseline", f"{nmbe(test['value'].to_numpy(), test['baseline'].to_numpy()):+.1f}%")
    m3.metric("Bất thường tại điểm (cả năm)", int((events["scope"] == "point").sum()))
    m4.metric("Bất thường trên khoảng (cả năm)", int((events["scope"] == "interval").sum()))
    m5.metric("Lỗi dữ liệu (cả năm)", len(quality))

    view = sc.loc[lo:hi].reset_index(names="ts")
    view["lo"] = view["baseline"] - k * view["scale"]
    view["hi"] = view["baseline"] + k * view["scale"]
    ev_view = in_window(events, lo, hi).assign(loai=lambda d: d["type"].map(TYPE_VI))
    spans = []
    lab_view = in_window(ent_labels, lo, hi).assign(loai=lambda d: d["type"].map(TYPE_VI))
    if not lab_view.empty:
        spans.append(span_layer(lab_view, COLORS["label"], ["case_id", "loai", "start:T", "end:T"], pd.Timedelta(hours=1)))
    dq_view = in_window(quality, lo, hi).assign(loai=lambda d: d["type"].map(TYPE_VI))
    if not dq_view.empty:
        spans.append(span_layer(dq_view, COLORS["dq"], ["loai", "start:T", "end:T"], pd.Timedelta(hours=1)))
    iv_view = ev_view[ev_view["scope"] == "interval"].assign(lech_tb=lambda d: d["mean_dev_pct"].round(1))
    if not iv_view.empty:
        spans.append(span_layer(iv_view, COLORS["anomaly"], ["loai", "start:T", "end:T", "duration_h", "lech_tb"],
                                pd.Timedelta(hours=1)))
    flagged = view[(view["score"] >= threshold) & (view["ts"] >= TRAIN_END)]

    st.altair_chart(
        actual_vs_baseline_chart(view, "ts", "value", "baseline", ev_view, "lo", "hi", spans, flagged=flagged),
        width="stretch",
    )
    st.caption(
        f"{date_lo:%d/%m/%Y} – {date_hi:%d/%m/%Y}. Đường nét đứt: baseline. Vùng xám: baseline ± k·σ. "
        "Tam giác đỏ: bất thường tại điểm. Vùng đỏ: bất thường trên khoảng (sự kiện dài, hoặc đoạn có z trung bình "
        "vượt ngưỡng khoảng). Chấm đỏ: từng giờ vượt ngưỡng điểm. Vùng cam: ca giả lập (nhãn). "
        "Vùng tím: lỗi dữ liệu (cảnh báo trùng vùng này bị bỏ)."
    )
    if method == IFOREST:
        st.altair_chart(score_chart(view, "ts", "score", threshold, "Điểm Isolation Forest", two_sided=False), width="stretch")
    else:
        title = "Điểm gộp (z chuẩn hóa)" if method == FUSION else "Độ lệch (robust z)"
        st.altair_chart(score_chart(view, "ts", "score", threshold, title, two_sided=method != FUSION), width="stretch")

    left, right = st.columns(2)
    left.subheader(f"Cảnh báo trong khoảng ({len(ev_view)})")
    show_events_table(left, ev_view)
    right.subheader(f"Lỗi dữ liệu trong khoảng ({len(dq_view)})")
    right.dataframe(
        dq_view[["loai", "start", "end", "n_points", "detail"]]
        .rename(columns={"loai": "Loại", "start": "Bắt đầu", "end": "Kết thúc", "n_points": "Số điểm", "detail": "Chi tiết"}),
        hide_index=True, width="stretch",
    )

    with st.expander("So sánh các cách phát hiện trên tòa nhà này (cùng ngân sách)"):
        a_labels = ent_labels[ent_labels["group"] == "A"]
        rows = []
        for m in METHODS:
            msc = hourly_scores(entity, column, m)
            mev = detect(entity, msc, quality, method_threshold(msc, m, k, if_q), budget, m, k_interval, window, point_max_h)
            row = {"Cách": m, "Điểm": int((mev["scope"] == "point").sum()),
                   "Khoảng": int((mev["scope"] == "interval").sum()), "Trong khoảng xem": len(in_window(mev, lo, hi))}
            if injected:
                hits = n_hit(mev, a_labels)
                matched = sum(((a_labels["start"] <= t) & (a_labels["end"] >= s)).any()
                              for s, t in zip(mev["start"], mev["end"]))
                row["Ca nhóm A bắt được"] = f"{hits}/{len(a_labels)}"
                row["Precision"] = round(matched / len(mev), 3) if len(mev) else np.nan
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if not injected:
            st.caption("Dữ liệu gốc không có nhãn, nên chỉ so số cảnh báo.")

# ---------------------------------------------------------------- efficiency tab
with tab_eff:
    cooling = load_cooling(entity)
    if cooling is None:
        st.info(f"{entity} không có dữ liệu tải lạnh (M2). Chọn tòa nhà khác ở thanh bên.")
    else:
        eff_threshold = st.slider("Ngưỡng vượt baseline (%)", 3, 20, 8) / 100
        s = sc["value"]
        eff = efficiency_scores(s, regularize(cooling, s.index))
        eff_ev = points_to_events(
            eff["score"].where(eff.index >= TRAIN_END), eff_threshold, entity,
            "load_vs_cooling_regression", max_gap=2, min_duration=3, step_h=24,
        )
        eff_ev = suppress_sensor_faults(eff_ev, quality)
        if not eff_ev.empty:
            peak = eff.loc[eff_ev["peak_ts"]]
            eff_ev = eff_ev.assign(
                value=peak["value"].to_numpy(), baseline=peak["baseline"].to_numpy(),
                deviation_pct=peak["score"].to_numpy() * 100, loai=TYPE_VI["efficiency_drop"],
            )
        eff_view = eff.loc[lo:hi].reset_index(names="date").dropna(subset=["baseline"])
        eff_ev_view = in_window(eff_ev, lo, hi) if not eff_ev.empty else eff_ev
        eff_labels = in_window(ent_labels_all[ent_labels_all["type"] == "efficiency_drop"], lo, hi) if injected else labels.iloc[0:0]
        spans = [span_layer(eff_labels.assign(loai=TYPE_VI["efficiency_drop"]), COLORS["label"],
                            ["case_id", "start:T", "end:T", "magnitude"], pd.Timedelta(hours=1))] if not eff_labels.empty else []
        if eff_view.empty:
            st.warning("Khoảng ngày đang chọn chưa có baseline hiệu suất (cần ~7 tuần dữ liệu trước đó). Chọn khoảng sau tháng 2/2016.")
        else:
            st.altair_chart(
                actual_vs_baseline_chart(eff_view, "date", "value", "baseline", eff_ev_view, spans=spans, y_title="kWh/ngày"),
                width="stretch",
            )
            eff_view["excess_pct"] = eff_view["score"] * 100
            st.altair_chart(score_chart(eff_view, "date", "excess_pct", eff_threshold * 100, "Vượt baseline (%)",
                                        bars=False, two_sided=False), width="stretch")
        st.caption(
            "Baseline: điện năng ngày dự đoán từ tải lạnh cùng ngày, hồi quy trên cửa sổ tham chiếu t-49..t-21 ngày. "
            "Cùng tải lạnh mà tốn điện hơn = giảm hiệu suất."
        )
        if not eff_ev_view.empty:
            show_events_table(st, eff_ev_view)

# ---------------------------------------------------------------- daily tab
with tab_daily:
    csv_path = CASE_DIR / "baseline_daily_cases.csv" if injected else BASELINE_CSV
    daily = pd.read_csv(csv_path)
    models = [c.removeprefix("pred_") for c in daily.columns if c.startswith("pred_")]
    c1, c2, c3 = st.columns(3)
    model = c1.selectbox("Mô hình baseline", models)
    z_thr = c2.slider("Ngưỡng robust z (ngày)", 2.0, 6.0, 3.0, 0.5)
    cusum_thr = c3.slider("Ngưỡng CUSUM", 2.0, 15.0, 5.0, 0.5)
    st.caption(
        f"Tòa nhà Rat_office_Colby, nguồn `{csv_path}`. Nhánh so sai lệch đọc mọi CSV theo cấu trúc "
        "date, actual_kWh, pred_*, residual_*. Cách phát hiện ở thanh bên không áp dụng cho tab này."
    )

    dsc = deviation_scores(daily, model=model)
    building = "Rat_office_Colby"
    spikes = points_to_events(dsc["score"], z_thr, building, "residual_robust_z", step_h=24).assign(loai="Tại điểm: lệch ngắn (z)")
    shifts = points_to_events(dsc["cusum"], cusum_thr, building, "residual_cusum", step_h=24).assign(loai="Trên khoảng: lệch kéo dài (CUSUM)")
    dev = pd.concat([spikes, shifts], ignore_index=True)
    if not dev.empty:
        peak = dsc.loc[dev["peak_ts"]]
        dev["value"] = peak["value"].to_numpy()
        dev["baseline"] = peak["baseline"].to_numpy()
        dev["deviation_pct"] = (dev["value"] - dev["baseline"]) / dev["baseline"] * 100

    d_labels = pd.read_csv(CASE_DIR / "baseline_daily_labels.csv", parse_dates=["start", "end"]) if injected else labels.iloc[0:0]
    d_labels = in_window(d_labels, lo, hi)
    spans = [span_layer(d_labels.assign(loai=d_labels["type"].map(TYPE_VI)), COLORS["label"],
                        ["case_id", "loai", "start:T", "end:T"], pd.Timedelta(days=1))] if not d_labels.empty else []
    ok = dsc.dropna(subset=["value"])
    k1, k2, k3 = st.columns(3)
    k1.metric("CV(RMSE)", f"{cv_rmse(ok['value'].to_numpy(), ok['baseline'].to_numpy()):.1f}%")
    k2.metric("NMBE", f"{nmbe(ok['value'].to_numpy(), ok['baseline'].to_numpy()):+.1f}%")
    k3.metric("Sự kiện bất thường (cả năm)", len(dev))

    dview = dsc.loc[lo:hi].reset_index(names="date")
    dev_view = in_window(dev, lo, hi) if not dev.empty else dev
    if dview.empty:
        st.warning("CSV baseline không có dữ liệu trong khoảng ngày đang chọn.")
    else:
        st.altair_chart(
            actual_vs_baseline_chart(dview, "date", "value", "baseline", dev_view, spans=spans, y_title="kWh/ngày"),
            width="stretch",
        )
        left, right = st.columns(2)
        left.altair_chart(score_chart(dview, "date", "z", z_thr, "Phần dư (robust z)"), width="stretch")
        right.altair_chart(score_chart(dview, "date", "cusum", cusum_thr, "CUSUM", bars=False, two_sided=False),
                           width="stretch")
    if not dev_view.empty:
        show_events_table(st, dev_view)
