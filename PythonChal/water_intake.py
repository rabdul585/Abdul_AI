import streamlit as st
import pandas as pd
import altair as alt
from datetime import date, timedelta
import os

# ----------------------
# Configuration
# ----------------------
DATA_FILE = "water_log.csv"  # local persistence
DAILY_GOAL_ML = 3000  # 3L in ml

st.set_page_config(
    page_title="Hydrate — Water Intake Tracker",
    page_icon="💧",
    layout="centered",
)

# ----------------------
# Utility functions
# ----------------------

def ensure_datafile(filepath=DATA_FILE):
    # Ensure the folder exists (helps when running from other dirs)
    folder = os.path.dirname(os.path.abspath(filepath))
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)


def load_data(filepath=DATA_FILE):
    ensure_datafile(filepath)
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        try:
            df = pd.read_csv(filepath, parse_dates=["date"]) 
            # normalize to date (no time component)
            df["date"] = pd.to_datetime(df["date"]).dt.date
            # make sure types are correct
            df = df.assign(ml=df["ml"].astype(int))
            return df
        except Exception:
            # If file is corrupted, start fresh but keep a backup
            try:
                os.rename(filepath, filepath + ".bak")
            except Exception:
                pass
            return pd.DataFrame(columns=["date", "ml"]).astype({"date":"datetime64[ns]","ml":"int"})
    else:
        return pd.DataFrame(columns=["date", "ml"]).astype({"date":"datetime64[ns]","ml":"int"})


def save_data(df, filepath=DATA_FILE):
    ensure_datafile(filepath)
    # Convert date objects to ISO strings for CSV stability
    out = df.copy()
    if not out.empty:
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(filepath, index=False)


def add_entry(amount_ml, when: date = None):
    if when is None:
        when = date.today()
    new = {"date": when, "ml": int(amount_ml)}
    df = load_data()
    df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
    save_data(df)
    return df


def todays_total(df):
    today = date.today()
    if df.empty:
        return 0
    # df['date'] contains date objects
    return int(df[df["date"] == today]["ml"].sum())


def last_n_days_totals(df, n=7):
    today = date.today()
    days = [today - timedelta(days=i) for i in range(n-1, -1, -1)]
    totals = []
    for d in days:
        s = int(df[df["date"] == d]["ml"].sum()) if not df.empty else 0
        totals.append({"date": d, "ml": s})
    return pd.DataFrame(totals)


# ----------------------
# App styling
# ----------------------

st.markdown(
    """
    <style>
    /* Page background */
    .stApp {
        background: linear-gradient(180deg, #0f1724 0%, #071029 100%);
        color: #e6eef8;
        font-family: 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', Arial;
    }

    /* Card like containers */
    .card {
        background: rgba(255,255,255,0.03);
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 6px 18px rgba(2,6,23,0.6);
        border: 1px solid rgba(255,255,255,0.03);
        margin-bottom: 12px;
    }

    .big-number {
        font-size: 36px;
        font-weight: 700;
    }

    .muted {
        color: #9fb1c8;
    }

    /* Button styling: make quick-add and control buttons clear and prominent */
    .stButton>button, .stDownloadButton>button {
        background: linear-gradient(90deg, #00b4d8, #0077b6);
        color: white;
        padding: 10px 14px;
        border-radius: 10px;
        font-weight: 700;
        border: none;
        box-shadow: 0 6px 18px rgba(0,0,0,0.4);
        min-width: 100%;
    }

    .stButton>button:hover, .stDownloadButton>button:hover {
        transform: translateY(-1px);
        filter: brightness(1.05);
    }

    /* Make quick-add buttons slightly larger text */
    .element-container .stButton>button {
        font-size: 14px;
    }

    /* Adjust table/card text contrast */
    .stTable td, .stTable th {
        color: #e6eef8;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------
# Load data
# ----------------------

df = load_data()

# ----------------------
# Header
# ----------------------

st.markdown(
    "<div style='display:flex; align-items:center; gap:12px'>"
    "<div style='font-size:36px'>💧</div>"
    "<div style='line-height:1'>"
    "<div style='font-size:20px; font-weight:700'>Hydrate — Water Intake Tracker</div>"
    "<div class='muted' style='font-size:13px'>Log daily water (ml). Goal: 3.0 L</div>"
    "</div></div>",
    unsafe_allow_html=True,
)

st.write("")

# ----------------------
# Main layout: left (log + quick actions), right (progress + chart)
# ----------------------

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Log water")

    # Quick-add buttons must NOT be inside st.form()
    st.write("Quick add")
    qcol1, qcol2, qcol3, qcol4 = st.columns(4)
    if qcol1.button("+200 ml"):
        df = add_entry(200, date.today())
        st.rerun()
    if qcol2.button("+250 ml"):
        df = add_entry(250, date.today())
        st.rerun()
    if qcol3.button("+500 ml"):
        df = add_entry(500, date.today())
        st.rerun()
    if qcol4.button("+1000 ml"):
        df = add_entry(1000, date.today())
        st.rerun()

    # Manual entry form
    with st.form("add_water_form", clear_on_submit=True):
        col1, col2 = st.columns([2, 1])
        amount = col1.number_input(
            "Amount (ml)", min_value=1, max_value=10000, value=250, step=50, format="%d"
        )
        when = col2.date_input("When", value=date.today())

        submitted = st.form_submit_button("Add entry")
        if submitted:
            if amount <= 0:
                st.error("Enter a positive amount")
            else:
                df = add_entry(amount, when)
                st.success(f"Added {amount} ml for {when}")
                st.rerun()

    st.write("</div>", unsafe_allow_html=True)

    # History and controls
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("History & Controls")

    if df.empty:
        st.info("No water logged yet. Use the buttons or the form to add entries.")
    else:
        # Show recent logs (most recent first)
        recent = df.sort_values(by=["date"], ascending=False).head(14)
        recent_display = recent.copy()
        recent_display["date"] = pd.to_datetime(recent_display["date"]).dt.strftime("%Y-%m-%d")
        st.table(recent_display.rename(columns={"date": "Date", "ml": "Amount (ml)"}))

        col_a, col_b = st.columns([2, 1])
        if col_a.button("Delete last entry"):
            # Delete the most recent entry by date (safer than relying on file order)
            df_all = load_data()
            if not df_all.empty:
                df_all = df_all.sort_values(by=["date"], ascending=False).reset_index(drop=True)
                # remove first row (most recent)
                df_all = df_all.iloc[1:].reset_index(drop=True)
                save_data(df_all)
                st.rerun()
            else:
                st.warning("Nothing to delete")

        if col_b.button("Clear all data"):
            save_data(pd.DataFrame(columns=["date", "ml"]))
            st.rerun()

    st.write("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Today's progress")

    today_ml = todays_total(df)
    percent = min(int((today_ml / DAILY_GOAL_ML) * 100), 100) if DAILY_GOAL_ML > 0 else 0

    # Big numbers + metric
    st.markdown(f"<div class='big-number'>{today_ml:,} ml</div>", unsafe_allow_html=True)
    st.write(f"{percent}% of {DAILY_GOAL_ML/1000:.1f} L goal")

    # Progress bar
    try:
        st.progress(percent / 100)
    except Exception:
        st.progress(0)

    # Helpful tips
    if percent >= 100:
        st.success("Nice! You've reached your daily goal 🎉")
    else:
        remaining = DAILY_GOAL_ML - today_ml
        st.info(f"{remaining:,} ml to reach 3.0 L")
        st.write("- Try sipping regularly — set a reminder every hour.")
        st.write("- Start the day with a glass of water (250 ml).")

    st.write("</div>", unsafe_allow_html=True)

    # Weekly chart
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Weekly hydration")

    last7 = last_n_days_totals(df, n=7)
    # Convert to datetime for altair
    last7_plot = last7.copy()
    last7_plot["date"] = pd.to_datetime(last7_plot["date"]) 

    base = alt.Chart(last7_plot).encode(
        x=alt.X("date:T", title="Date", axis=alt.Axis(format="%b %d")),
    )

    bars = base.mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
        y=alt.Y("ml:Q", title="ml"),
        tooltip=[alt.Tooltip("date:T", title="Date", format="%Y-%m-%d"), alt.Tooltip("ml:Q", title="Intake (ml)")],
    )

    target_line = base.mark_rule(color="#00b4d8", strokeDash=[4,4]).encode(y=alt.value(DAILY_GOAL_ML))
    target_text = base.mark_text(align='left', dx=3, dy=-10).encode(text=alt.value(f"Daily target: {DAILY_GOAL_ML} ml"))

    chart = (bars + target_line + target_text).properties(height=240, width="container")

    st.altair_chart(chart, use_container_width=True)

    # Weekly stats
    weekly_total = int(last7_plot["ml"].sum())
    weekly_avg = int(last7_plot["ml"].mean()) if not last7_plot["ml"].empty else 0
    st.write(f"**This week's total:** {weekly_total:,} ml — **Avg/day:** {weekly_avg:,} ml")

    st.write("</div>", unsafe_allow_html=True)

# ----------------------
# Footer: small guidance
# ----------------------

st.markdown("---")
col1, col2 = st.columns([3, 1])
with col1:
    st.write("Need to export or import data? Use the controls below.")
    df_all = load_data()
    if not df_all.empty:
        csv_bytes = df_all.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv_bytes, file_name="water_log.csv", mime="text/csv")

with col2:
    st.write("")

st.caption("Built with ❤️ — Track your hydration and stay healthy. Data is stored locally in the app folder.")

# ----------------------
# End of file
# ----------------------
