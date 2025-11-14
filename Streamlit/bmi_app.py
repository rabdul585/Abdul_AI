# bmi_app.py
"""
Complete Streamlit BMI Calculator (auto-calc)
Save as `bmi_app.py` and run:
    streamlit run bmi_app.py

If you get a blank page: make sure you run with `streamlit run` (not `python`),
and open the Local URL (usually http://localhost:8501) printed by Streamlit.
"""

import streamlit as st
import pandas as pd
import datetime
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="BMI Calculator", layout="centered")

# --- Utilities ---
def calculate_bmi(weight, height, units="Metric"):
    """Return BMI rounded to 2 decimals. Metric: kg & cm. Imperial: lbs & inches."""
    try:
        if units == "Metric":
            h_m = height / 100.0  # cm -> meters
            bmi = weight / (h_m * h_m)
        else:
            bmi = (weight / (height * height)) * 703
        return round(bmi, 2)
    except Exception:
        return None

def bmi_category(bmi):
    if bmi is None:
        return ("Unknown", "gray")
    if bmi < 18.5:
        return ("Underweight", "#3b82f6")
    if 18.5 <= bmi < 25:
        return ("Normal (Healthy)", "#10b981")
    if 25 <= bmi < 30:
        return ("Overweight", "#f59e0b")
    return ("Obese", "#ef4444")

def short_tip(category):
    tips = {
        "Underweight": "Consider a nutrient-rich diet and consult a nutritionist if needed.",
        "Normal": "Maintain a balanced diet and regular physical activity.",
        "Overweight": "Try increasing physical activity and adjust calorie intake.",
        "Obese": "Consult healthcare provider for personalized guidance."
    }
    # match by first word to support "Normal (Healthy)" etc.
    return tips.get(category.split()[0], "")

# --- Sidebar inputs ---
st.sidebar.title("Inputs")
units = st.sidebar.selectbox(
    "Units", ["Metric", "Imperial"], index=0,
    help="Metric: weight in kg, height in cm. Imperial: weight in lbs, height in inches."
)

col1, col2 = st.sidebar.columns(2)
with col1:
    weight = st.number_input(
        "Weight",
        min_value=1.0, max_value=635.0, value=70.0, step=0.1, format="%.1f",
        help="Enter weight in selected units"
    )
with col2:
    if units == "Metric":
        height = st.number_input(
            "Height (cm)",
            min_value=50.0, max_value=300.0, value=170.0, step=0.1, format="%.1f"
        )
    else:
        height = st.number_input(
            "Height (in)",
            min_value=20.0, max_value=100.0, value=67.0, step=0.1, format="%.1f"
        )

age = st.sidebar.number_input("Age (optional)", min_value=1, max_value=120, value=30)
sex = st.sidebar.selectbox("Sex (optional)", ["Prefer not to say", "Male", "Female", "Other"])

# Initialize session history list
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts

# Debug / help: quick link to show run info (hidden by default)
with st.expander("Run & debug info (expand if needed)"):
    st.write("Streamlit run command: `streamlit run bmi_app.py`")
    st.write("If the page is blank: ensure you opened the exact Local URL printed by Streamlit (e.g., http://localhost:8501).")

# --- Main UI ---
st.title("BMI Calculator")
st.markdown("Auto-calculates BMI as you change inputs. Session history and CSV download included.")

# Input validation
error = None
if weight <= 0:
    error = "Weight must be greater than zero."
if height <= 0:
    error = "Height must be greater than zero."

if error:
    st.sidebar.error(error)
    st.error(error)
else:
    bmi = calculate_bmi(weight, height, units)
    category, color = bmi_category(bmi)
    tip = short_tip(category)

    # Results layout
    colA, colB = st.columns([2, 1])
    with colA:
        st.subheader(f"Your BMI: {bmi}")
        st.markdown(
            f"**Category:** <span style='color:{color};font-weight:600'>{category}</span>",
            unsafe_allow_html=True
        )
        st.write(f"Age: {age} • Sex: {sex}")
        if tip:
            st.info(tip)
        st.caption("BMI is a screening measure — it doesn't measure body fat directly. For personalized advice consult a healthcare professional.")
    with colB:
        st.metric(label="BMI value", value=f"{bmi}")
        # small color bar
        st.markdown(
            f"<div style='width:100%;height:18px;background:{color};border-radius:6px'></div>",
            unsafe_allow_html=True
        )

    # Prepare entry and add to session history (avoid duplicate entries on immediate reruns)
    entry = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "units": units,
        "weight": float(weight),
        "height": float(height),
        "age": int(age),
        "sex": sex,
        "bmi": float(bmi),
        "category": category
    }

    # Append if different from last saved entry (prevents duplicates when Streamlit reruns)
    if not st.session_state.history:
        st.session_state.history.append(entry)
    else:
        last = st.session_state.history[-1]
        # compare relevant fields (not timestamp)
        compare_keys = ["units", "weight", "height", "age", "sex", "bmi", "category"]
        changed = any(last.get(k) != entry.get(k) for k in compare_keys)
        if changed:
            st.session_state.history.append(entry)

# Show session history
if st.session_state.history:
    st.markdown("---")
    st.subheader("Session History")
    df = pd.DataFrame(st.session_state.history)
    # reorder columns
    df = df[["timestamp", "units", "weight", "height", "age", "sex", "bmi", "category"]]
    st.dataframe(df.sort_values("timestamp", ascending=False).reset_index(drop=True))

    # BMI over time chart
    try:
        fig, ax = plt.subplots()
        df_plot = df.copy()
        df_plot["timestamp_parsed"] = pd.to_datetime(df_plot["timestamp"])
        df_plot = df_plot.sort_values("timestamp_parsed")
        ax.plot(df_plot["timestamp_parsed"], df_plot["bmi"], marker="o")
        ax.set_title("BMI over time")
        ax.set_xlabel("Time")
        ax.set_ylabel("BMI")
        ax.grid(True)
        st.pyplot(fig)
    except Exception as e:
        st.write("Could not render chart:", e)

    # Download CSV
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download history (CSV)", csv, "bmi_history.csv", "text/csv")
else:
    st.info("No entries yet. Adjust the inputs in the sidebar to create the first entry.")

# Footer
st.markdown("---")
st.caption("This tool is for informational purposes only and not a medical device.")
