from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import beta as beta_dist
from streamlit_gsheets import GSheetsConnection


st.set_page_config(
    page_title="Dashboard Risiko Energi Primer",
    page_icon="⚡",
    layout="wide",
)


# =========================================================
# FUNGSI BANTU
# =========================================================
def format_number(value, decimals=2):
    """Format angka dengan koma ribuan dan titik desimal."""
    return f"{value:,.{decimals}f}"


def moving_block_bootstrap(values, horizon, simulations, block_length, seed):
    """Simulasikan lintasan HOP sambil mempertahankan pola harian berurutan."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.empty((0, horizon))

    effective_block = min(block_length, values.size)
    max_start = values.size - effective_block
    blocks_needed = int(np.ceil(horizon / effective_block))
    generator = np.random.default_rng(seed)
    result = np.empty((simulations, horizon), dtype=float)

    for simulation_index in range(simulations):
        starts = generator.integers(0, max_start + 1, size=blocks_needed)
        path = np.concatenate(
            [values[start : start + effective_block] for start in starts]
        )
        result[simulation_index] = path[:horizon]

    return result


@st.cache_data
def load_model_data():
    file_path = Path(__file__).parent / "data" / "model_input.csv"
    return pd.read_csv(file_path)


@st.cache_data(ttl=60)
def load_hop_data():
    connection = st.connection("gsheets", type=GSheetsConnection)
    data = connection.read(worksheet=305648890, ttl=60)
    data.columns = data.columns.astype(str).str.strip().str.lower()

    required = {"tanggal", "unit", "hop"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            "Kolom Google Sheets belum lengkap: " + ", ".join(sorted(missing))
        )

    data = data[["tanggal", "unit", "hop"]].copy()
    data["tanggal"] = pd.to_datetime(data["tanggal"], errors="coerce")
    data["unit"] = data["unit"].astype("string").str.strip()
    data["hop"] = pd.to_numeric(data["hop"], errors="coerce")
    data = data.dropna(subset=["tanggal", "unit", "hop"])
    data = data[data["unit"] != ""]
    return data.sort_values(["unit", "tanggal"]).reset_index(drop=True)


# =========================================================
# MEMBACA DAN MEMVALIDASI DATA MODEL
# =========================================================
try:
    model_data = load_model_data()
except FileNotFoundError:
    st.error("File data/model_input.csv tidak ditemukan.")
    st.stop()
except Exception as error:
    st.error(f"Data model tidak dapat dibaca: {error}")
    st.stop()


required_model_columns = [
    "model_id",
    "risk_code",
    "unit",
    "commodity",
    "period",
    "forecast_name",
    "minimum",
    "most_likely",
    "maximum",
    "lambda",
    "threshold",
    "uom",
    "active",
]

missing_model_columns = [
    column for column in required_model_columns if column not in model_data.columns
]

if missing_model_columns:
    st.error(
        "Kolom berikut belum tersedia pada CSV: "
        + ", ".join(missing_model_columns)
    )
    st.stop()

model_data = model_data[
    model_data["active"].astype(str).str.strip().str.lower() == "ya"
].copy()

if model_data.empty:
    st.error("Tidak ada model aktif pada data/model_input.csv.")
    st.stop()


try:
    hop_data = load_hop_data()
    hop_error = None
except Exception as error:
    hop_data = pd.DataFrame(columns=["tanggal", "unit", "hop"])
    hop_error = str(error)


# =========================================================
# JUDUL
# =========================================================
st.title("⚡ Dashboard Risiko Hambatan Energi Primer")
st.caption(
    "Pemodelan probabilistik Beta-PERT dan simulasi Monte Carlo, "
    "dilengkapi pemantauan HOP dari Google Sheets."
)


# =========================================================
# DATA HISTORIS HOP
# =========================================================
st.subheader("Data Historis HOP")

if hop_error:
    st.warning(f"Data HOP Google Sheets belum dapat dibaca: {hop_error}")

if hop_data.empty:
    st.info("Data HOP belum tersedia atau belum berhasil dibaca.")
else:
    jumlah_unit_hop = hop_data["unit"].nunique()
    tanggal_awal_hop = hop_data["tanggal"].min()
    tanggal_akhir_hop = hop_data["tanggal"].max()

    hop_col1, hop_col2, hop_col3 = st.columns(3)
    hop_col1.metric("Jumlah Data", f"{len(hop_data):,} baris")
    hop_col2.metric("Jumlah Unit", f"{jumlah_unit_hop:,} unit")
    hop_col3.metric(
        "Periode Data",
        f"{tanggal_awal_hop:%d-%m-%Y} s.d. {tanggal_akhir_hop:%d-%m-%Y}",
    )

    st.subheader("Tren HOP Harian per Unit")

    control_1, control_2, control_3 = st.columns(3)
    hop_unit_options = sorted(hop_data["unit"].astype(str).unique())
    selected_hop_unit = control_1.selectbox(
        "Pilih Unit",
        options=hop_unit_options,
        key="selected_hop_unit",
    )
    batas_aman = control_2.number_input(
        "Batas Aman HOP (hari)",
        min_value=0.1,
        value=7.0,
        step=0.5,
        key="batas_aman_hop",
    )
    batas_kritis = control_3.number_input(
        "Batas Kritis HOP (hari)",
        min_value=0.0,
        value=3.0,
        step=0.5,
        key="batas_kritis_hop",
    )

    if batas_kritis >= batas_aman:
        st.error("Batas kritis harus lebih kecil daripada batas aman.")
    else:
        # =================================================
        # RINGKASAN RISIKO HOP SELURUH UNIT
        # =================================================
        hop_with_status = hop_data.copy()
        hop_with_status["status"] = np.select(
            [
                hop_with_status["hop"] <= batas_kritis,
                hop_with_status["hop"] < batas_aman,
            ],
            ["Kritis", "Waspada"],
            default="Aman",
        )

        latest_index = hop_with_status.groupby("unit")["tanggal"].idxmax()
        latest_hop = (
            hop_with_status.loc[latest_index, ["unit", "tanggal", "hop", "status"]]
            .rename(
                columns={
                    "tanggal": "tanggal_terkini",
                    "hop": "hop_terkini",
                    "status": "status_terkini",
                }
            )
            .set_index("unit")
        )

        hop_summary = hop_with_status.groupby("unit").agg(
            jumlah_observasi=("hop", "size"),
            hop_rata_rata=("hop", "mean"),
            hop_minimum=("hop", "min"),
            hari_di_bawah_aman=("hop", lambda values: int((values < batas_aman).sum())),
            hari_kritis=("hop", lambda values: int((values <= batas_kritis).sum())),
        )
        hop_summary = hop_summary.join(latest_hop)
        hop_summary["frekuensi_di_bawah_aman_pct"] = (
            hop_summary["hari_di_bawah_aman"]
            / hop_summary["jumlah_observasi"]
            * 100
        )
        hop_summary["frekuensi_kritis_pct"] = (
            hop_summary["hari_kritis"]
            / hop_summary["jumlah_observasi"]
            * 100
        )
        hop_summary = hop_summary.sort_values(
            ["frekuensi_kritis_pct", "frekuensi_di_bawah_aman_pct"],
            ascending=[False, False],
        ).reset_index()
        hop_summary.insert(0, "peringkat", range(1, len(hop_summary) + 1))

        st.markdown("#### Peringkat Risiko HOP Seluruh Unit")

        ranking_chart = go.Figure()
        ranking_chart.add_trace(
            go.Bar(
                y=hop_summary["unit"],
                x=hop_summary["frekuensi_di_bawah_aman_pct"],
                name="Di Bawah Batas Aman",
                orientation="h",
                marker_color="#F59E0B",
                customdata=np.stack(
                    [
                        hop_summary["hari_di_bawah_aman"],
                        hop_summary["jumlah_observasi"],
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "Unit: %{y}"
                    "<br>Frekuensi: %{x:.2f}%"
                    "<br>Jumlah: %{customdata[0]} dari %{customdata[1]} hari"
                    "<extra></extra>"
                ),
            )
        )
        ranking_chart.add_trace(
            go.Bar(
                y=hop_summary["unit"],
                x=hop_summary["frekuensi_kritis_pct"],
                name="Kritis",
                orientation="h",
                marker_color="#E53935",
                customdata=np.stack(
                    [
                        hop_summary["hari_kritis"],
                        hop_summary["jumlah_observasi"],
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "Unit: %{y}"
                    "<br>Frekuensi: %{x:.2f}%"
                    "<br>Jumlah: %{customdata[0]} dari %{customdata[1]} hari"
                    "<extra></extra>"
                ),
            )
        )
        ranking_chart.update_layout(
            title="Frekuensi HOP di Bawah Batas per Unit",
            xaxis_title="Frekuensi terhadap Jumlah Observasi (%)",
            yaxis_title="Unit",
            barmode="group",
            yaxis=dict(autorange="reversed"),
            height=max(420, 65 * len(hop_summary)),
            margin=dict(l=30, r=30, t=70, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(ranking_chart, use_container_width=True)

        ranking_display = hop_summary.rename(
            columns={
                "peringkat": "Peringkat",
                "unit": "Unit",
                "tanggal_terkini": "Tanggal Terkini",
                "hop_terkini": "HOP Terkini",
                "status_terkini": "Status Terkini",
                "jumlah_observasi": "Jumlah Observasi",
                "hop_rata_rata": "Rata-rata HOP",
                "hop_minimum": "HOP Minimum",
                "hari_di_bawah_aman": "Hari di Bawah Aman",
                "hari_kritis": "Hari Kritis",
                "frekuensi_di_bawah_aman_pct": "Frekuensi di Bawah Aman (%)",
                "frekuensi_kritis_pct": "Frekuensi Kritis (%)",
            }
        )
        ranking_display = ranking_display[
            [
                "Peringkat",
                "Unit",
                "Tanggal Terkini",
                "HOP Terkini",
                "Status Terkini",
                "Rata-rata HOP",
                "HOP Minimum",
                "Hari di Bawah Aman",
                "Hari Kritis",
                "Frekuensi di Bawah Aman (%)",
                "Frekuensi Kritis (%)",
                "Jumlah Observasi",
            ]
        ]
        st.dataframe(
            ranking_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Tanggal Terkini": st.column_config.DateColumn(format="DD-MM-YYYY"),
                "HOP Terkini": st.column_config.NumberColumn(format="%.2f hari"),
                "Rata-rata HOP": st.column_config.NumberColumn(format="%.2f hari"),
                "HOP Minimum": st.column_config.NumberColumn(format="%.2f hari"),
                "Frekuensi di Bawah Aman (%)": st.column_config.NumberColumn(format="%.2f%%"),
                "Frekuensi Kritis (%)": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )

        st.markdown("#### Detail Unit Terpilih")

        unit_hop_data = hop_data[
            hop_data["unit"].astype(str) == selected_hop_unit
        ].copy()
        unit_hop_data = unit_hop_data.sort_values("tanggal")

        unit_hop_data["status"] = np.select(
            [
                unit_hop_data["hop"] <= batas_kritis,
                unit_hop_data["hop"] < batas_aman,
            ],
            ["Kritis", "Waspada"],
            default="Aman",
        )

        hop_terkini = unit_hop_data.iloc[-1]
        hop_rata_rata = unit_hop_data["hop"].mean()
        hop_minimum = unit_hop_data["hop"].min()
        jumlah_di_bawah_aman = int((unit_hop_data["hop"] < batas_aman).sum())
        probabilitas_kritis = (
            (unit_hop_data["hop"] <= batas_kritis).mean() * 100
        )

        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric(
            "HOP Terkini",
            f"{format_number(hop_terkini['hop'])} hari",
            help=f"Data tanggal {hop_terkini['tanggal']:%d-%m-%Y}",
        )
        kpi2.metric("Rata-rata HOP", f"{format_number(hop_rata_rata)} hari")
        kpi3.metric("HOP Minimum", f"{format_number(hop_minimum)} hari")
        kpi4.metric("Hari di Bawah Aman", f"{jumlah_di_bawah_aman} hari")
        kpi5.metric(
            "Frekuensi Kondisi Kritis",
            f"{format_number(probabilitas_kritis)}%",
        )

        latest_status = str(hop_terkini["status"])
        if latest_status == "Kritis":
            st.error(f"Status HOP terkini {selected_hop_unit}: KRITIS")
        elif latest_status == "Waspada":
            st.warning(f"Status HOP terkini {selected_hop_unit}: WASPADA")
        else:
            st.success(f"Status HOP terkini {selected_hop_unit}: AMAN")

        hop_chart = go.Figure()
        hop_chart.add_trace(
            go.Scatter(
                x=unit_hop_data["tanggal"],
                y=unit_hop_data["hop"],
                mode="lines",
                name="HOP Aktual",
                line=dict(color="#94A3B8", width=2),
                hoverinfo="skip",
            )
        )

        status_colors = {
            "Aman": "#16A34A",
            "Waspada": "#F59E0B",
            "Kritis": "#E53935",
        }
        for status_name, color in status_colors.items():
            status_data = unit_hop_data[unit_hop_data["status"] == status_name]
            if not status_data.empty:
                hop_chart.add_trace(
                    go.Scatter(
                        x=status_data["tanggal"],
                        y=status_data["hop"],
                        mode="markers",
                        name=status_name,
                        marker=dict(color=color, size=10),
                        customdata=status_data["status"],
                        hovertemplate=(
                            "Tanggal: %{x|%d-%m-%Y}"
                            "<br>HOP: %{y:.2f} hari"
                            "<br>Status: %{customdata}"
                            "<extra></extra>"
                        ),
                    )
                )

        hop_chart.add_hline(
            y=batas_aman,
            line_color="#16A34A",
            line_dash="dash",
            annotation_text=f"Batas Aman: {batas_aman:g} hari",
            annotation_position="top left",
        )
        hop_chart.add_hline(
            y=batas_kritis,
            line_color="#E53935",
            line_dash="dash",
            annotation_text=f"Batas Kritis: {batas_kritis:g} hari",
            annotation_position="bottom left",
        )
        hop_chart.update_layout(
            title=f"Tren HOP — {selected_hop_unit}",
            xaxis_title="Tanggal",
            yaxis_title="Hari Operasi Pembangkit",
            hovermode="x unified",
            height=460,
            margin=dict(l=30, r=30, t=70, b=30),
        )
        st.plotly_chart(hop_chart, use_container_width=True)

        # =================================================
        # PROBABILITAS DAN SIMULASI MONTE CARLO HOP
        # =================================================
        st.markdown("#### Probabilitas dan Simulasi HOP")
        st.caption(
            "Simulasi menggunakan moving-block bootstrap agar pola HOP "
            "beberapa hari berturut-turut tetap terwakili."
        )

        simulation_col1, simulation_col2, simulation_col3 = st.columns(3)
        horizon_hop = simulation_col1.selectbox(
            "Horizon Simulasi",
            options=[30, 90, 180, 365],
            index=1,
            format_func=lambda value: f"{value} hari",
            key="horizon_simulasi_hop",
        )
        jumlah_simulasi_hop = simulation_col2.selectbox(
            "Jumlah Skenario HOP",
            options=[500, 1000, 5000, 10000],
            index=1,
            format_func=lambda value: f"{value:,} skenario",
            key="jumlah_simulasi_hop",
        )
        panjang_blok = simulation_col3.selectbox(
            "Panjang Blok Historis",
            options=[3, 7, 14, 30],
            index=1,
            format_func=lambda value: f"{value} hari",
            key="panjang_blok_hop",
            help=(
                "Blok 7 hari mempertahankan pola perubahan HOP mingguan "
                "saat sampel historis diacak."
            ),
        )

        historical_hop = unit_hop_data["hop"].to_numpy(dtype=float)
        historical_p5, historical_p10, historical_p50 = np.percentile(
            historical_hop, [5, 10, 50]
        )
        historical_below_safe = (historical_hop < batas_aman).mean() * 100
        historical_critical = (historical_hop <= batas_kritis).mean() * 100

        hop_paths = moving_block_bootstrap(
            values=historical_hop,
            horizon=horizon_hop,
            simulations=jumlah_simulasi_hop,
            block_length=panjang_blok,
            seed=2027,
        )

        scenario_minimum = hop_paths.min(axis=1)
        scenario_mean = hop_paths.mean(axis=1)
        scenario_below_safe_days = (hop_paths < batas_aman).sum(axis=1)
        scenario_critical_days = (hop_paths <= batas_kritis).sum(axis=1)

        probability_any_below_safe = (
            (scenario_below_safe_days > 0).mean() * 100
        )
        probability_any_critical = (
            (scenario_critical_days > 0).mean() * 100
        )
        expected_below_safe_days = scenario_below_safe_days.mean()
        expected_critical_days = scenario_critical_days.mean()

        hist_kpi1, hist_kpi2, hist_kpi3, hist_kpi4, hist_kpi5 = st.columns(5)
        hist_kpi1.metric("P5 Historis", f"{format_number(historical_p5)} hari")
        hist_kpi2.metric("P10 Historis", f"{format_number(historical_p10)} hari")
        hist_kpi3.metric("P50 Historis", f"{format_number(historical_p50)} hari")
        hist_kpi4.metric(
            "Historis di Bawah Aman",
            f"{format_number(historical_below_safe)}%",
        )
        hist_kpi5.metric(
            "Historis Kritis",
            f"{format_number(historical_critical)}%",
        )

        mc_kpi1, mc_kpi2, mc_kpi3, mc_kpi4 = st.columns(4)
        mc_kpi1.metric(
            "Peluang Ada Hari di Bawah Aman",
            f"{format_number(probability_any_below_safe)}%",
            help=f"Sedikitnya satu hari dalam horizon {horizon_hop} hari.",
        )
        mc_kpi2.metric(
            "Peluang Ada Hari Kritis",
            f"{format_number(probability_any_critical)}%",
            help=f"Sedikitnya satu hari dalam horizon {horizon_hop} hari.",
        )
        mc_kpi3.metric(
            "Ekspektasi Hari di Bawah Aman",
            f"{format_number(expected_below_safe_days)} hari",
        )
        mc_kpi4.metric(
            "Ekspektasi Hari Kritis",
            f"{format_number(expected_critical_days)} hari",
        )

        future_dates = pd.date_range(
            start=unit_hop_data["tanggal"].max() + pd.Timedelta(days=1),
            periods=horizon_hop,
            freq="D",
        )
        path_p10 = np.percentile(hop_paths, 10, axis=0)
        path_p50 = np.percentile(hop_paths, 50, axis=0)
        path_p90 = np.percentile(hop_paths, 90, axis=0)

        fan_chart = go.Figure()
        fan_chart.add_trace(
            go.Scatter(
                x=future_dates,
                y=path_p90,
                mode="lines",
                line=dict(width=0),
                name="P90",
                hovertemplate="P90: %{y:.2f} hari<extra></extra>",
            )
        )
        fan_chart.add_trace(
            go.Scatter(
                x=future_dates,
                y=path_p10,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(20, 115, 230, 0.22)",
                name="Rentang P10–P90",
                hovertemplate="P10: %{y:.2f} hari<extra></extra>",
            )
        )
        fan_chart.add_trace(
            go.Scatter(
                x=future_dates,
                y=path_p50,
                mode="lines",
                line=dict(color="#1473E6", width=3),
                name="P50",
                hovertemplate="P50: %{y:.2f} hari<extra></extra>",
            )
        )
        fan_chart.add_hline(
            y=batas_aman,
            line_color="#16A34A",
            line_dash="dash",
            annotation_text="Batas Aman",
        )
        fan_chart.add_hline(
            y=batas_kritis,
            line_color="#E53935",
            line_dash="dash",
            annotation_text="Batas Kritis",
        )
        fan_chart.update_layout(
            title=f"Proyeksi Monte Carlo HOP {horizon_hop} Hari — {selected_hop_unit}",
            xaxis_title="Tanggal Simulasi",
            yaxis_title="HOP (hari)",
            hovermode="x unified",
            height=470,
            margin=dict(l=30, r=30, t=70, b=30),
        )
        st.plotly_chart(fan_chart, use_container_width=True)

        distribution_col1, distribution_col2 = st.columns(2)

        minimum_histogram = go.Figure(
            go.Histogram(
                x=scenario_minimum,
                nbinsx=45,
                marker_color="#E53935",
                opacity=0.82,
                hovertemplate=(
                    "Minimum HOP: %{x:.2f} hari"
                    "<br>Frekuensi: %{y}"
                    "<extra></extra>"
                ),
            )
        )
        minimum_histogram.add_vline(
            x=batas_kritis,
            line_color="#7F1D1D",
            line_dash="dash",
            annotation_text="Batas Kritis",
        )
        minimum_histogram.update_layout(
            title="Distribusi Minimum HOP per Skenario",
            xaxis_title="Minimum HOP (hari)",
            yaxis_title="Frekuensi",
            height=400,
            showlegend=False,
            margin=dict(l=30, r=30, t=70, b=30),
        )
        distribution_col1.plotly_chart(
            minimum_histogram, use_container_width=True
        )

        average_histogram = go.Figure(
            go.Histogram(
                x=scenario_mean,
                nbinsx=45,
                marker_color="#4EC1C1",
                opacity=0.82,
                hovertemplate=(
                    "Rata-rata HOP: %{x:.2f} hari"
                    "<br>Frekuensi: %{y}"
                    "<extra></extra>"
                ),
            )
        )
        average_histogram.add_vline(
            x=batas_aman,
            line_color="#166534",
            line_dash="dash",
            annotation_text="Batas Aman",
        )
        average_histogram.update_layout(
            title="Distribusi Rata-rata HOP per Skenario",
            xaxis_title="Rata-rata HOP (hari)",
            yaxis_title="Frekuensi",
            height=400,
            showlegend=False,
            margin=dict(l=30, r=30, t=70, b=30),
        )
        distribution_col2.plotly_chart(
            average_histogram, use_container_width=True
        )

        with st.expander("Metode simulasi HOP"):
            st.markdown(
                f"""
                - Sumber simulasi: **{len(historical_hop):,} observasi**
                  milik **{selected_hop_unit}**.
                - Horizon: **{horizon_hop} hari** dengan
                  **{jumlah_simulasi_hop:,} skenario**.
                - Panjang blok: **{panjang_blok} hari**. Potongan data historis
                  yang berurutan dipilih secara acak dan disambungkan sampai
                  memenuhi horizon simulasi.
                - **Peluang ada hari kritis** adalah persentase skenario yang
                  memiliki sedikitnya satu nilai HOP ≤ {batas_kritis:g} hari.
                - Simulasi ini merepresentasikan pola historis. Model belum
                  memasukkan rencana pasokan, konsumsi, cuaca, atau jadwal kapal.
                """
            )

        with st.expander("Lihat data HOP dari Google Sheets"):
            st.dataframe(
                unit_hop_data[["tanggal", "unit", "hop", "status"]],
                use_container_width=True,
                hide_index=True,
            )


# =========================================================
# PEMILIHAN MODEL
# =========================================================
st.divider()
st.sidebar.header("Pemilihan Model")

unit_options = sorted(model_data["unit"].dropna().astype(str).unique())
selected_unit = st.sidebar.selectbox("Unit", options=unit_options)

unit_data = model_data[model_data["unit"].astype(str) == selected_unit]
commodity_options = sorted(
    unit_data["commodity"].dropna().astype(str).unique()
)
selected_commodity = st.sidebar.selectbox(
    "Komoditas", options=commodity_options
)

commodity_data = unit_data[
    unit_data["commodity"].astype(str) == selected_commodity
]
period_options = sorted(commodity_data["period"].dropna().unique())
selected_period = st.sidebar.selectbox("Periode", options=period_options)

filtered_data = commodity_data[commodity_data["period"] == selected_period]
if filtered_data.empty:
    st.warning("Model untuk pilihan tersebut tidak ditemukan.")
    st.stop()

selected_model = filtered_data.iloc[0]
model_id = str(selected_model["model_id"])
st.sidebar.info(
    f"Model: {model_id}\n\nRisiko: {selected_model['risk_code']}"
)


# =========================================================
# INPUT PARAMETER BETA-PERT
# =========================================================
st.sidebar.header("Parameter Beta-PERT")

minimum = st.sidebar.number_input(
    "Minimum (Rp miliar)",
    value=float(selected_model["minimum"]),
    step=1.0,
    key=f"minimum_{model_id}",
)
most_likely = st.sidebar.number_input(
    "Most Likely (Rp miliar)",
    value=float(selected_model["most_likely"]),
    step=1.0,
    key=f"most_likely_{model_id}",
)
maximum = st.sidebar.number_input(
    "Maximum (Rp miliar)",
    value=float(selected_model["maximum"]),
    step=1.0,
    key=f"maximum_{model_id}",
)
lambda_value = st.sidebar.number_input(
    "Lambda Beta-PERT",
    min_value=1.0,
    max_value=10.0,
    value=float(selected_model["lambda"]),
    step=0.5,
    key=f"lambda_{model_id}",
)
threshold = st.sidebar.number_input(
    "Threshold Risiko (Rp miliar)",
    value=float(selected_model["threshold"]),
    step=1.0,
    key=f"threshold_{model_id}",
)

simulation_options = [1000, 5000, 10000, 25000, 50000]
default_simulation = int(selected_model.get("simulation_count", 10000))
if default_simulation not in simulation_options:
    simulation_options.append(default_simulation)
    simulation_options.sort()

jumlah_simulasi = st.sidebar.selectbox(
    "Jumlah Simulasi Monte Carlo",
    options=simulation_options,
    index=simulation_options.index(default_simulation),
    key=f"simulation_count_{model_id}",
)

if not minimum < most_likely < maximum:
    st.error(
        "Parameter tidak valid. Nilainya harus memenuhi: "
        "Minimum < Most Likely < Maximum."
    )
    st.stop()


# =========================================================
# PERHITUNGAN BETA-PERT
# =========================================================
rentang = maximum - minimum
alpha = 1 + lambda_value * ((most_likely - minimum) / rentang)
beta_shape = 1 + lambda_value * ((maximum - most_likely) / rentang)
expected_loss = (
    minimum + lambda_value * most_likely + maximum
) / (lambda_value + 2)


def calculate_percentile(probability):
    return minimum + rentang * beta_dist.ppf(
        probability, alpha, beta_shape
    )


p20 = calculate_percentile(0.20)
p50 = calculate_percentile(0.50)
p90 = calculate_percentile(0.90)
p95 = calculate_percentile(0.95)

if threshold <= minimum:
    probability_exceed = 100.0
elif threshold >= maximum:
    probability_exceed = 0.0
else:
    threshold_normalized = (threshold - minimum) / rentang
    probability_exceed = (
        1 - beta_dist.cdf(threshold_normalized, alpha, beta_shape)
    ) * 100


# =========================================================
# IDENTITAS DAN KPI MODEL
# =========================================================
st.subheader(str(selected_model["forecast_name"]))
identity_1, identity_2, identity_3, identity_4 = st.columns(4)
identity_1.info(f"**Unit**\n\n{selected_unit}")
identity_2.info(f"**Komoditas**\n\n{selected_commodity}")
identity_3.info(f"**Periode**\n\n{selected_period}")
identity_4.info(f"**Model ID**\n\n{model_id}")

st.subheader("Ringkasan Forecast")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Expected Loss", f"Rp{format_number(expected_loss)} M")
col2.metric("P20", f"Rp{format_number(p20)} M")
col3.metric("P50", f"Rp{format_number(p50)} M")
col4.metric("P90", f"Rp{format_number(p90)} M")
col5.metric("P95", f"Rp{format_number(p95)} M")

if probability_exceed >= 20:
    exposure_status = "Tinggi"
elif probability_exceed >= 10:
    exposure_status = "Menengah"
else:
    exposure_status = "Rendah"

exceed_col, status_col, shape_col = st.columns(3)
exceed_col.metric(
    "Probabilitas Melampaui Threshold",
    f"{format_number(probability_exceed)}%",
)
status_col.metric("Status Eksposur", exposure_status)
shape_col.metric("Alpha / Beta", f"{alpha:.2f} / {beta_shape:.2f}")


# =========================================================
# KURVA BETA-PERT
# =========================================================
x = np.linspace(minimum, maximum, 800)
x_normalized = (x - minimum) / rentang
density = beta_dist.pdf(x_normalized, alpha, beta_shape) / rentang

curve = go.Figure()
zones = [
    (x <= p20, "≤P20", "#EF6677"),
    ((x > p20) & (x <= p50), "P20–P50", "#F7B547"),
    ((x > p50) & (x <= p90), "P50–P90", "#4EC1C1"),
    (x > p90, ">P90", "#78C787"),
]

for mask, zone_name, color in zones:
    curve.add_trace(
        go.Scatter(
            x=x[mask],
            y=density[mask],
            mode="lines",
            name=zone_name,
            line=dict(color=color, width=2),
            fill="tozeroy",
            fillcolor=color,
            hovertemplate=(
                "Loss: Rp%{x:,.2f} miliar"
                "<br>Density: %{y:.5f}"
                "<extra></extra>"
            ),
        )
    )

reference_lines = [
    (p20, "P20", "#C7354C"),
    (p50, "P50", "#D88708"),
    (p90, "P90", "#178F91"),
    (p95, "P95", "#E53935"),
    (threshold, "Threshold", "#16A34A"),
]
for value, label, color in reference_lines:
    curve.add_vline(
        x=value,
        line_width=2,
        line_dash="dash",
        line_color=color,
        annotation_text=label,
        annotation_position="top",
    )

curve.update_layout(
    title="Distribusi Beta-PERT — Loss Opportunity",
    xaxis_title="Loss Opportunity (Rp miliar)",
    yaxis_title="Kepadatan Probabilitas",
    legend_title="Zona Percentile",
    hovermode="x unified",
    height=530,
    margin=dict(l=30, r=30, t=70, b=30),
)
st.plotly_chart(curve, use_container_width=True)


# =========================================================
# SIMULASI MONTE CARLO
# =========================================================
st.subheader("Simulasi Monte Carlo")
random_seed = int(selected_model.get("random_seed", 2027))
random_generator = np.random.default_rng(random_seed)
simulation = minimum + rentang * random_generator.beta(
    alpha, beta_shape, jumlah_simulasi
)

histogram = go.Figure()
histogram.add_trace(
    go.Histogram(
        x=simulation,
        nbinsx=50,
        marker_color="#1473E6",
        opacity=0.80,
        name="Hasil Simulasi",
    )
)

histogram_lines = [
    (p50, "P50", "#F59E0B", "dash"),
    (p90, "P90", "#E53935", "dash"),
    (p95, "P95", "#8B0000", "dash"),
    (threshold, "Threshold", "#16A34A", "dot"),
]
for value, label, color, dash in histogram_lines:
    histogram.add_vline(
        x=value,
        line_color=color,
        line_dash=dash,
        annotation_text=label,
    )

histogram.update_layout(
    xaxis_title="Loss Opportunity (Rp miliar)",
    yaxis_title="Frekuensi",
    height=430,
    showlegend=False,
)
st.plotly_chart(histogram, use_container_width=True)


# =========================================================
# DATA DAN PENJELASAN
# =========================================================
with st.expander("Lihat sumber data model"):
    st.dataframe(filtered_data, use_container_width=True, hide_index=True)

with st.expander("Cara membaca dashboard"):
    st.markdown(
        f"""
        - **P20 Rp{format_number(p20)} miliar**: sekitar 20% hasil berada
          pada atau di bawah nilai tersebut.
        - **P50 Rp{format_number(p50)} miliar**: median atau skenario tengah.
        - **P90 Rp{format_number(p90)} miliar**: skenario konservatif.
        - **P95 Rp{format_number(p95)} miliar**: kondisi yang lebih ekstrem.
        - Peluang melampaui threshold **Rp{format_number(threshold)} miliar**
          adalah **{format_number(probability_exceed)}%**.
        """
    )

st.caption(
    "Sumber parameter model: data/model_input.csv pada repository GitHub. "
    "Sumber historis HOP: Google Sheets. Model Beta-PERT menggunakan "
    "minimum, most likely, maksimum, dan lambda."
)

