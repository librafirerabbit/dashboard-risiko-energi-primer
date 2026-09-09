import streamlit as st

st.set_page_config(
    page_title="Dashboard Risiko Energi Primer",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Dashboard Risiko Hambatan Energi Primer")
st.subheader("PLN Nusantara Power")

st.success("Aplikasi Streamlit berhasil dijalankan.")

st.write(
    "Dashboard ini akan digunakan untuk menampilkan pemodelan "
    "Beta-PERT, simulasi Monte Carlo, percentile risiko, dan dampak finansial."
)
