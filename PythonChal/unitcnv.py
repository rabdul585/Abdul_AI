# unitcnv.py
# Minimal guard to avoid running with `python unitcnv.py` directly (no Streamlit import here)
import os, sys
if "STREAMLIT_RUN_MAIN" not in os.environ and __name__ == "__main__":
    print("\nThis is a Streamlit app. Run it with:\n\n    streamlit run unitcnv.py\n")
    sys.exit(0)

# ---------- App code ----------
import streamlit as st
import requests

# -------------------------
# Page config & CSS
# -------------------------
st.set_page_config(page_title="🔁 Unit Converter — Dark", page_icon="🔁", layout="wide")

st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    .stApp {
        background-color: #0b1220;
        color: #e6eef8;
        font-family: "Inter", sans-serif;
    }
    .card {
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 6px 18px rgba(2,6,23,0.55);
        margin-bottom: 12px;
    }
    .title { font-size:1.25rem; font-weight:700; margin-bottom:6px; }
    .muted { color:#9aa6b2; font-size:0.9rem; }
    .small { font-size:0.85rem; color:#9aa6b2; }
    .result { font-size:1.05rem; font-weight:600; margin-top:8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Helper functions
# -------------------------
def get_inr_to_usd_rate(fallback=0.01128, timeout=5):
    """Return (rate, date) or (fallback, None) on failure."""
    try:
        resp = requests.get("https://api.exchangerate.host/latest?base=INR&symbols=USD", timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        rate = data.get("rates", {}).get("USD")
        if rate:
            return float(rate), data.get("date")
    except Exception:
        pass
    return float(fallback), None

def c_to_f(c): return c * 9.0 / 5.0 + 32.0
def cm_to_in(cm): return cm / 2.54
def kg_to_lb(kg): return kg * 2.20462262185

# -------------------------
# Header
# -------------------------
st.markdown(
    '<div class="card"><div style="display:flex;justify-content:space-between;align-items:center;">'
    '<div><div class="title">🔁 Unit Converter — Dark</div><div class="muted">Compact • Side-by-side • Friendly ✨</div></div>'
    '<div style="text-align:right"><div class="small">Built with Python + Streamlit</div></div>'
    '</div></div>',
    unsafe_allow_html=True,
)

# -------------------------
# Side-by-side layout (2 columns)
# -------------------------
left_col, right_col = st.columns(2, gap="large")

# LEFT column: Currency & Temperature
with left_col:
    # Currency card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 💱 Currency — INR → USD")
    inr_amount = st.number_input("Amount (₹ INR)", min_value=0.0, value=1000.0, step=100.0, format="%.2f", key="inr_amount")
    use_live = st.checkbox("Use live rate", value=True, help="Fetch fresh INR→USD rate from exchangerate.host", key="use_live")
    live_rate, rate_date = get_inr_to_usd_rate()
    fallback_rate = 0.01128
    effective_rate = live_rate if use_live else fallback_rate
    usd_amount = inr_amount * effective_rate
    st.markdown(f"<div class='result'>₹{inr_amount:,.2f} → <strong>${usd_amount:,.4f}</strong></div>", unsafe_allow_html=True)
    src = "Live API" if use_live else "Fallback rate"
    rate_info = f"1 INR = {effective_rate:.6f} USD  ·  Source: {src}"
    if use_live and rate_date:
        rate_info += f" (date: {rate_date})"
    st.markdown(f"<div class='small'>{rate_info}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Temperature card (stacked below currency)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🌡️ Temperature — °C → °F")
    celsius = st.number_input("Degrees Celsius (°C)", value=25.0, format="%.2f", key="celsius")
    fahrenheit = c_to_f(celsius)
    st.markdown(f"<div class='result'>{celsius:.2f} °C → <strong>{fahrenheit:.2f} °F</strong></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# RIGHT column: Length & Weight
with right_col:
    # Length card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📏 Length — cm → in")
    cm_value = st.number_input("Centimeters (cm)", value=100.0, format="%.2f", key="cm_value")
    inches = cm_to_in(cm_value)
    st.markdown(f"<div class='result'>{cm_value:.2f} cm → <strong>{inches:.4f} in</strong></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Weight card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ⚖️ Weight — kg → lb")
    kg_value = st.number_input("Kilograms (kg)", value=70.0, format="%.2f", key="kg_value")
    pounds = kg_to_lb(kg_value)
    st.markdown(f"<div class='result'>{kg_value:.2f} kg → <strong>{pounds:.2f} lb</strong></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown(
    '<div style="margin-top:10px;"><div class="small" style="text-align:center;">'
    'Tip: This page shows all converters side-by-side. Toggle <strong>Use live rate</strong> to fetch latest INR→USD.</div></div>',
    unsafe_allow_html=True,
)
