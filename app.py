import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import beta as beta_dist

st.set_page_config(
    page_title="Dashboard Risiko Energi Primer",
    page_icon="⚡",
    layout="wide"
)

# =========================================================
# JUDUL DASHBOARD
# =========================================================
st.title("⚡ Dashboard Risiko Hambatan Energi Primer")
st.caption(
    "Pemodelan probabilistik Beta-PERT dan simulasi Monte Carlo "
    "untuk Loss Opportunity Hambatan Energi Primer"
)

# =========================================================
# INPUT PARAMETER
# =========================================================
st.sidebar.header("Parameter Model")

minimum = st.sidebar.number_input(
    "Minimum (Rp miliar)",
    value=264.82,
    step=1.0
)

most_likely = st.sidebar.number_input(
    "Most Likely (Rp miliar)",
    value=345.51,
    step=1.0
)

maximum = st.sidebar.number_input(
    "Maximum (Rp miliar)",
    value=446.39,
    step=1.0
)

lambda_value = st.sidebar.number_input(
    "Lambda Beta-PERT",
    min_value=1.0,
    max_value=10.0,
    value=4.0,
    step=0.5
)

threshold = st.sidebar.number_input(
    "Threshold Risiko (Rp miliar)",
    value=389.20,
    step=1.0
)

jumlah_simulasi = st.sidebar.selectbox(
    "Jumlah Simulasi Monte Carlo",
    options=[1000, 5000, 10000, 25000, 50000],
    index=2
)

# =========================================================
# VALIDASI INPUT
# =========================================================
if not minimum < most_likely < maximum:
    st.error(
        "Parameter tidak valid. Harus memenuhi: "
        "Minimum < Most Likely < Maximum."
    )
    st.stop()

# =========================================================
# PARAMETER BETA-PERT
# =========================================================
rentang = maximum - minimum

alpha = 1 + lambda_value * (
    (most_likely - minimum) / rentang
)

beta_shape = 1 + lambda_value * (
    (maximum - most_likely) / rentang
)

mean_value = (
    minimum
    + lambda_value * most_likely
    + maximum
) / (lambda_value + 2)

def percentile_beta(probability):
    return minimum + rentang * beta_dist.ppf(
        probability,
        alpha,
        beta_shape
    )

p20 = percentile_beta(0.20)
p50 = percentile_beta(0.50)
p90 = percentile_beta(0.90)
p95 = percentile_beta(0.95)

if threshold <= minimum:
    probability_exceed = 100.0
elif threshold >= maximum:
    probability_exceed = 0.0
else:
    threshold_normalized = (
        threshold - minimum
    ) / rentang

    probability_exceed = (
        1 - beta_dist.cdf(
            threshold_normalized,
            alpha,
            beta_shape
        )
    ) * 100

# =========================================================
# KPI UTAMA
# =========================================================
st.subheader("Ringkasan Forecast")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Expected Loss",
    f"Rp{mean_value:,.2f} M"
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

st.metric(
    "Probabilitas Melampaui Threshold",
    f"{probability_exceed:,.2f}%"
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

fig = go.Figure()

zones = [
    (
        x <= p20,
        "≤P20",
        "#EF6677"
    ),
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
    (
        x > p90,
        ">P90",
        "#78C787"
    )
]

for mask, zone_name, color in zones:
    fig.add_trace(
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

percentile_lines = [
    (p20, "P20", "#C7354C"),
    (p50, "P50", "#D88708"),
    (p90, "P90", "#178F91"),
    (p95, "P95", "#E53935"),
    (threshold, "Threshold", "#16A34A")
]

for value, label, color in percentile_lines:
    fig.add_vline(
        x=value,
        line_width=2,
        line_dash="dash",
        line_color=color,
        annotation_text=label,
        annotation_position="top"
    )

fig.update_layout(
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
    fig,
    use_container_width=True
)

# =========================================================
# HISTOGRAM MONTE CARLO
# =========================================================
st.subheader("Simulasi Monte Carlo")

random_generator = np.random.default_rng(2026)

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
# PENJELASAN
# =========================================================
with st.expander("Cara membaca dashboard"):
    st.markdown(
        f"""
        - **P20 sebesar Rp{p20:,.2f} miliar** berarti sekitar
          20% hasil berada pada atau di bawah nilai tersebut.
        - **P50 sebesar Rp{p50:,.2f} miliar** merupakan median
          atau skenario tengah.
        - **P90 sebesar Rp{p90:,.2f} miliar** dapat digunakan
          sebagai skenario konservatif.
        - **P95 sebesar Rp{p95:,.2f} miliar** menggambarkan
          kondisi yang lebih ekstrem.
        - Peluang kerugian melampaui threshold
          **Rp{threshold:,.2f} miliar** adalah
          **{probability_exceed:,.2f}%**.
        """
    )

st.caption(
    "Model Beta-PERT merupakan pendekatan berdasarkan nilai "
    "minimum, most likely, maksimum, dan lambda."
)
