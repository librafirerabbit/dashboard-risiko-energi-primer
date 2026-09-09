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
    layout="wide"
)


# =========================================================
# MEMBACA DATA CSV
# =========================================================
@st.cache_data
def load_model_data():
    file_path = Path(__file__).parent / "data" / "model_input.csv"
    return pd.read_csv(file_path)


try:
    model_data = load_model_data()
except FileNotFoundError:
    st.error("File data/model_input.csv tidak ditemukan.")
    st.stop()
except Exception as error:
    st.error(f"Data tidak dapat dibaca: {error}")
    st.stop()

# =========================================================
# MEMBACA DATA HOP DARI GOOGLE SHEETS
# =========================================================
GOOGLE_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1K-tMzdJNhsRBMDyJPPc4rI-spO2EwbHdnzcavLsEFBM/edit"
)

try:
    sheets_connection = st.connection(
        "gsheets",
        type=GSheetsConnection
    )

  hop_data = sheets_connection.read(
    worksheet=305648890,
    ttl=60

    )

    hop_data["tanggal"] = pd.to_datetime(
        hop_data["tanggal"],
        errors="coerce"
    )

    hop_data["hop"] = pd.to_numeric(
        hop_data["hop"],
        errors="coerce"
    )

    hop_data = hop_data.dropna(
        subset=["tanggal", "unit", "hop"]
    )

except Exception as error:
    st.warning(
        f"Data HOP Google Sheets belum dapat dibaca: {error}"
    )
    hop_data = pd.DataFrame(
        columns=["tanggal", "unit", "hop"]
    )

required_columns = [
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

missing_columns = [
    column
    for column in required_columns
    if column not in model_data.columns
]

if missing_columns:
    st.error(
        "Kolom berikut belum tersedia pada CSV: "
        + ", ".join(missing_columns)
    )
    st.stop()


model_data = model_data[
    model_data["active"].astype(str).str.lower() == "ya"
].copy()


# =========================================================
# JUDUL
# =========================================================
st.title("⚡ Dashboard Risiko Hambatan Energi Primer")
st.caption(
    "Pemodelan probabilistik Beta-PERT dan simulasi Monte Carlo "
    "berdasarkan data pada GitHub"
)


# =========================================================
# FILTER DATA
# =========================================================
st.sidebar.header("Pemilihan Model")

unit_options = sorted(
    model_data["unit"].dropna().astype(str).unique()
)

selected_unit = st.sidebar.selectbox(
    "Unit",
    options=unit_options
)

unit_data = model_data[
    model_data["unit"].astype(str) == selected_unit
]

commodity_options = sorted(
    unit_data["commodity"].dropna().astype(str).unique()
)

selected_commodity = st.sidebar.selectbox(
    "Komoditas",
    options=commodity_options
)

commodity_data = unit_data[
    unit_data["commodity"].astype(str) == selected_commodity
]

period_options = sorted(
    commodity_data["period"].dropna().unique()
)

selected_period = st.sidebar.selectbox(
    "Periode",
    options=period_options
)

filtered_data = commodity_data[
    commodity_data["period"] == selected_period
]

if filtered_data.empty:
    st.warning("Model untuk pilihan tersebut tidak ditemukan.")
    st.stop()


selected_model = filtered_data.iloc[0]
model_id = str(selected_model["model_id"])

st.sidebar.info(
    f"Model: {model_id}\n\n"
    f"Risiko: {selected_model['risk_code']}"
)


# =========================================================
# INPUT PARAMETER
# =========================================================
st.sidebar.header("Parameter Beta-PERT")

minimum = st.sidebar.number_input(
    "Minimum (Rp miliar)",
    value=float(selected_model["minimum"]),
    step=1.0,
    key=f"minimum_{model_id}"
)

most_likely = st.sidebar.number_input(
    "Most Likely (Rp miliar)",
    value=float(selected_model["most_likely"]),
    step=1.0,
    key=f"most_likely_{model_id}"
)

maximum = st.sidebar.number_input(
    "Maximum (Rp miliar)",
    value=float(selected_model["maximum"]),
    step=1.0,
    key=f"maximum_{model_id}"
)

lambda_value = st.sidebar.number_input(
    "Lambda Beta-PERT",
    min_value=1.0,
    max_value=10.0,
    value=float(selected_model["lambda"]),
    step=0.5,
    key=f"lambda_{model_id}"
)

threshold = st.sidebar.number_input(
    "Threshold Risiko (Rp miliar)",
    value=float(selected_model["threshold"]),
    step=1.0,
    key=f"threshold_{model_id}"
)

simulation_options = [1000, 5000, 10000, 25000, 50000]

default_simulation = int(
    selected_model.get("simulation_count", 10000)
)

if default_simulation not in simulation_options:
    simulation_options.append(default_simulation)
    simulation_options.sort()

jumlah_simulasi = st.sidebar.selectbox(
    "Jumlah Simulasi Monte Carlo",
    options=simulation_options,
    index=simulation_options.index(default_simulation),
    key=f"simulation_count_{model_id}"
)


# =========================================================
# VALIDASI
# =========================================================
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

alpha = 1 + lambda_value * (
    (most_likely - minimum) / rentang
)

beta_shape = 1 + lambda_value * (
    (maximum - most_likely) / rentang
)

expected_loss = (
    minimum
    + lambda_value * most_likely
    + maximum
) / (lambda_value + 2)


def calculate_percentile(probability):
    return minimum + rentang * beta_dist.ppf(
        probability,
        alpha,
        beta_shape
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
    threshold_normalized = (
        threshold - minimum
    ) / rentang

    probability_exceed = (
        1
        - beta_dist.cdf(
            threshold_normalized,
            alpha,
            beta_shape
        )
    ) * 100


# =========================================================
# IDENTITAS MODEL
# =========================================================
st.subheader(str(selected_model["forecast_name"]))

identity_1, identity_2, identity_3, identity_4 = st.columns(4)

identity_1.info(f"**Unit**\n\n{selected_unit}")
identity_2.info(f"**Komoditas**\n\n{selected_commodity}")
identity_3.info(f"**Periode**\n\n{selected_period}")
identity_4.info(f"**Model ID**\n\n{model_id}")


# =========================================================
# KPI
# =========================================================
st.subheader("Ringkasan Forecast")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Expected Loss",
    f"Rp{expected_loss:,.2f} M"
)

col2.metric(
    "P20",
    f"Rp{p20:,.2f} M"
)

col3.metric(
    "P50",
    f"Rp{p50:,.2f} M"
)

col4.metric(
    "P90",
    f"Rp{p90:,.2f} M"
)

col5.metric(
    "P95",
    f"Rp{p95:,.2f} M"
)

if probability_exceed >= 20:
    status = "Tinggi"
elif probability_exceed >= 10:
    status = "Menengah"
else:
    status = "Rendah"

exceed_col, status_col, shape_col = st.columns(3)

exceed_col.metric(
    "Probabilitas Melampaui Threshold",
    f"{probability_exceed:,.2f}%"
)

status_col.metric(
    "Status Eksposur",
    status
)

shape_col.metric(
    "Alpha / Beta",
    f"{alpha:.2f} / {beta_shape:.2f}"
)


# =========================================================
# KURVA BETA-PERT
# =========================================================
x = np.linspace(minimum, maximum, 800)
x_normalized = (x - minimum) / rentang

density = beta_dist.pdf(
    x_normalized,
    alpha,
    beta_shape
) / rentang

curve = go.Figure()

zones = [
    (x <= p20, "≤P20", "#EF6677"),
    (
        (x > p20) & (x <= p50),
        "P20–P50",
        "#F7B547"
    ),
    (
        (x > p50) & (x <= p90),
        "P50–P90",
        "#4EC1C1"
    ),
    (x > p90, ">P90", "#78C787")
]

for mask, zone_name, color in zones:
    curve.add_trace(
        go.Scatter(
            x=x[mask],
            y=density[mask],
            mode="lines",
            name=zone_name,
            line=dict(
                color=color,
                width=2
            ),
            fill="tozeroy",
            fillcolor=color,
            hovertemplate=(
                "Loss: Rp%{x:,.2f} miliar"
                "<br>Density: %{y:.5f}"
                "<extra></extra>"
            )
        )
    )

reference_lines = [
    (p20, "P20", "#C7354C"),
    (p50, "P50", "#D88708"),
    (p90, "P90", "#178F91"),
    (p95, "P95", "#E53935"),
    (threshold, "Threshold", "#16A34A")
]

for value, label, color in reference_lines:
    curve.add_vline(
        x=value,
        line_width=2,
        line_dash="dash",
        line_color=color,
        annotation_text=label,
        annotation_position="top"
    )

curve.update_layout(
    title="Distribusi Beta-PERT — Loss Opportunity",
    xaxis_title="Loss Opportunity (Rp miliar)",
    yaxis_title="Kepadatan Probabilitas",
    legend_title="Zona Percentile",
    hovermode="x unified",
    height=530,
    margin=dict(
        l=30,
        r=30,
        t=70,
        b=30
    )
)

st.plotly_chart(
    curve,
    use_container_width=True
)


# =========================================================
# SIMULASI MONTE CARLO
# =========================================================
st.subheader("Simulasi Monte Carlo")

random_seed = int(
    selected_model.get("random_seed", 2027)
)

random_generator = np.random.default_rng(random_seed)

simulation = minimum + rentang * random_generator.beta(
    alpha,
    beta_shape,
    jumlah_simulasi
)

histogram = go.Figure()

histogram.add_trace(
    go.Histogram(
        x=simulation,
        nbinsx=50,
        marker_color="#1473E6",
        opacity=0.80,
        name="Hasil Simulasi"
    )
)

histogram.add_vline(
    x=p50,
    line_color="#F59E0B",
    line_dash="dash",
    annotation_text="P50"
)

histogram.add_vline(
    x=p90,
    line_color="#E53935",
    line_dash="dash",
    annotation_text="P90"
)

histogram.add_vline(
    x=p95,
    line_color="#8B0000",
    line_dash="dash",
    annotation_text="P95"
)

histogram.add_vline(
    x=threshold,
    line_color="#16A34A",
    line_dash="dot",
    annotation_text="Threshold"
)

histogram.update_layout(
    xaxis_title="Loss Opportunity (Rp miliar)",
    yaxis_title="Frekuensi",
    height=430,
    showlegend=False
)

st.plotly_chart(
    histogram,
    use_container_width=True
)


# =========================================================
# DATA DAN PENJELASAN
# =========================================================
with st.expander("Lihat sumber data model"):
    st.dataframe(
        filtered_data,
        use_container_width=True,
        hide_index=True
    )

with st.expander("Cara membaca dashboard"):
    st.markdown(
        f"""
        - **P20 Rp{p20:,.2f} miliar**: sekitar 20% hasil
          berada pada atau di bawah nilai tersebut.
        - **P50 Rp{p50:,.2f} miliar**: median atau skenario
          tengah.
        - **P90 Rp{p90:,.2f} miliar**: skenario konservatif.
        - **P95 Rp{p95:,.2f} miliar**: kondisi yang lebih
          ekstrem.
        - Peluang melampaui threshold
          **Rp{threshold:,.2f} miliar** adalah
          **{probability_exceed:,.2f}%**.
        """
    )

st.caption(
    "Sumber parameter: data/model_input.csv pada repository GitHub. "
    "Model Beta-PERT merupakan pendekatan berdasarkan minimum, "
    "most likely, maksimum, dan lambda."
)
