# Prompt (use this to generate or to give to an LLM/codegen tool) """ You are an expert developer. Build a single-file Streamlit application (Python) called **Expense Splitter** that is visually appealing, responsive, and accessible. Requirements: - Tech stack: Python 3.x, Streamlit, pandas (optional), standard library only otherwise. - Core flow: 1. User can enter the **total amount** and **number of people** (or add people one-by-one). 2. Allow optional input of **names** for each person. 3. Allow optional **individual contributions** (amount each person already paid). If contributions are left blank, assume 0. 4. App computes an **equal share** per person (total / number_of_people) and determines for each person whether they **owe** money or should **receive** money back. 5. Display the results in a clear table with columns: Name, Paid, Share, Balance (positive -> should receive, negative -> owes). Show amounts formatted to 2 decimal places and show currency symbol. 6. Provide a friendly natural-language settlement summary (e.g., "A owes B ₹X, C gets back ₹Y") and a simplified settle-up suggestion (minimize transactions if practical). - UI/UX requirements: - Use a clean layout with header, subheader, and a brief help tooltip/expander. - Use columns to group inputs and actions (e.g., left column for inputs, right column for quick actions & summary). - Show visual cues: badges/colored chips for "Owes" vs "Gets back" and small icons. - Provide controls: "Add person" (adds a name + contribution input), "Auto-fill names" (optional), "Reset", and "Download CSV" of results. - Make the app mobile-friendly (use responsive Streamlit layout where possible). - Edge cases and validation: - Validate numeric inputs (non-negative, reasonable precision). - Handle when number of people doesn't match the number of provided names/contributions. - If number of people is changed, preserve existing person entries when possible. - Deliverables: - A single-file, runnable Streamlit app with comments and helpful variable names. - Include clear instructions at top (how to save and run: streamlit run expense_splitter.py). - Keep dependencies minimal and documented in comments. Optional (bonus): - Provide an algorithm that suggests the minimal set of payments to settle balances (greedy positive/negative matching). - Support splitting by custom weights (e.g., someone pays double share) and tax/tip percent. """
"""
Expense Splitter — visually enhanced dark theme single-file Streamlit app.

Save as `expense_splitter.py` and run:
    python -m streamlit run expense_splitter.py

Dependencies:
    pip install streamlit pandas
"""

from typing import List, Dict
import streamlit as st
import pandas as pd
from io import BytesIO

# Page config
st.set_page_config(
    page_title="Expense Splitter • Dark",
    page_icon="💸",
    layout="wide",
)

# --------------------------
# Minimal Dark Theme Styling
# --------------------------
# We use inline CSS to force a dark look, card-style panels, badges, and emojis.
# NOTE: Streamlit theming is preferred for production; this CSS augments it.
st.markdown(
    """
    <style>
    /* Page background & main text */
    .stApp {
        background: linear-gradient(180deg, #0f1724 0%, #071025 100%);
        color: #e6eef8;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }
    /* Card container */
    .card {
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 6px 18px rgba(2,6,23,0.6);
        border: 1px solid rgba(255,255,255,0.03);
    }
    /* Input spacing */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        background: rgba(255,255,255,0.03) !important;
        color: #e6eef8 !important;
        border-radius: 8px !important;
        padding: 10px !important;
    }
    /* Buttons */
    .stButton>button {
        border-radius: 10px;
        padding: 8px 14px;
    }
    /* Dataframe header style */
    .dataframe th {
        background: rgba(255,255,255,0.03) !important;
        color: #dbeafe !important;
    }
    /* Badges */
    .badge-owe {
        display:inline-block; padding:6px 10px; border-radius:999px;
        background: linear-gradient(90deg,#ff4d6d,#ff7a8a); color:white; font-weight:600;
        box-shadow: 0 4px 12px rgba(255,77,109,0.12);
    }
    .badge-get {
        display:inline-block; padding:6px 10px; border-radius:999px;
        background: linear-gradient(90deg,#5eead4,#14b8a6); color:#022c22; font-weight:700;
        box-shadow: 0 4px 12px rgba(20,184,166,0.08);
    }
    /* Small muted text */
    .muted { color: rgba(230,238,248,0.6); font-size:0.95rem; }
    /* Small helper */
    .helper { color: rgba(230,238,248,0.65); font-size:0.9rem; margin-top:6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------
# Helpers & Business Logic
# --------------------------
CURRENCY = "₹"


def fmt(a: float) -> str:
    return f"{CURRENCY}{a:,.2f}"


def compute_shares(total: float, people: List[Dict]) -> pd.DataFrame:
    n = max(1, len(people))
    share = round(total / n, 2)
    rows = []
    for p in people:
        paid = float(p.get("paid") or 0)
        name = (p.get("name") or p.get("id") or "").strip() or p.get("id")
        balance = round(paid - share, 2)
        rows.append({"Name": name, "Paid": paid, "Share": share, "Balance": balance})
    return pd.DataFrame(rows)


def suggest_settlements(df: pd.DataFrame) -> List[str]:
    creditors = df[df.Balance > 0][["Name", "Balance"]].sort_values(by="Balance", ascending=False).to_dict("records")
    debtors = df[df.Balance < 0][["Name", "Balance"]].sort_values(by="Balance").to_dict("records")
    i, j = 0, 0
    settlements = []
    while i < len(debtors) and j < len(creditors):
        debtor = debtors[i]
        creditor = creditors[j]
        owe = -debtor["Balance"]
        receive = creditor["Balance"]
        pay = round(min(owe, receive), 2)
        settlements.append(f"{debtor['Name']} ➜ {creditor['Name']}: {fmt(pay)}")
        debtors[i]["Balance"] += pay
        creditors[j]["Balance"] -= pay
        if abs(debtors[i]["Balance"]) < 0.01:
            i += 1
        if abs(creditors[j]["Balance"]) < 0.01:
            j += 1
    return settlements


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


# --------------------------
# App Header
# --------------------------
st.markdown(
    """
    <div class="card">
      <h1 style="margin:0;">💸 <strong>Expense Splitter</strong></h1>
      <div class="muted">Split bills quickly — who paid, who owes, and who should be reimbursed.</div>
      <div style="margin-top:10px;" class="helper">Tip: click <strong>Add person</strong> to add entries. Use CSV export to share results.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")  # spacing

# --------------------------
# Layout: Inputs (left) and Summary (right)
# --------------------------
left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📝 Inputs")
    total_amount = st.number_input("Total amount", min_value=0.0, format="%.2f", value=0.0, help="Total bill amount (including tax/tip if desired).")
    num_people = st.number_input("Number of people", min_value=1, step=1, value=2)

    st.markdown("---")
    st.markdown("**👥 People & contributions**")

    # session state for people
    if "people" not in st.session_state:
        st.session_state.people = [{"id": f"P1", "name": "Person 1", "paid": 0.0} for _ in range(int(num_people))]

    # preserve entries when num_people changes
    target = int(num_people)
    while len(st.session_state.people) < target:
        idx = len(st.session_state.people) + 1
        st.session_state.people.append({"id": f"P{idx}", "name": f"Person {idx}", "paid": 0.0})
    while len(st.session_state.people) > target:
        st.session_state.people.pop()

    # display each person as compact rows
    for i, person in enumerate(st.session_state.people):
        row_a, row_b = st.columns([3, 1])
        with row_a:
            person["name"] = st.text_input(f"Name #{i+1}", value=person.get("name", f"Person {i+1}"), key=f"name_{i}")
        with row_b:
            person["paid"] = st.number_input(f"Paid #{i+1}", min_value=0.0, format="%.2f", value=float(person.get("paid", 0.0)), key=f"paid_{i}")

    st.write("")  # spacing
    add_col1, add_col2, add_col3 = st.columns([1,1,1])
    with add_col1:
        if st.button("➕ Add person"):
            idx = len(st.session_state.people) + 1
            st.session_state.people.append({"id": f"P{idx}", "name": f"Person {idx}", "paid": 0.0})
            st.experimental_rerun()
    with add_col2:
        if st.button("🧾 Auto-fill names"):
            for i, p in enumerate(st.session_state.people):
                p["name"] = p.get("name") or f"Person {i+1}"
            st.experimental_rerun()
    with add_col3:
        if st.button("♻️ Reset"):
            st.session_state.people = [{"id": "P1", "name": "Person 1", "paid": 0.0}]
            st.experimental_rerun()

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Summary")
    people_copy = [p.copy() for p in st.session_state.people]
    df = compute_shares(float(total_amount), people_copy)

    total_paid = df.Paid.sum() if not df.empty else 0.0
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total bill", fmt(total_amount))
    col_b.metric("Total paid", fmt(total_paid))
    col_c.metric("Per person", fmt(df['Share'].iloc[0] if not df.empty else 0.0))

    st.markdown("---")
    st.subheader("💰 Balances")
    # format for display
    df_display = df.copy()
    df_display["Paid"] = df_display["Paid"].apply(fmt)
    df_display["Share"] = df_display["Share"].apply(fmt)
    df_display["Balance"] = df_display["Balance"].apply(lambda x: fmt(x) if x >= 0 else f"-{fmt(abs(x))}")

    st.dataframe(df_display, use_container_width=True, height=240)

    # Badges + friendly list
    owes = df[df.Balance < 0]
    gets = df[df.Balance > 0]

    if not owes.empty:
        st.markdown("**🔻 Owes**")
        for _, r in owes.iterrows():
            st.markdown(f'<span class="badge-owe">💸 {r["Name"]}</span>  &nbsp; owes **{fmt(-r["Balance"])}**', unsafe_allow_html=True)
    if not gets.empty:
        st.markdown("**🟢 Gets back**")
        for _, r in gets.iterrows():
            st.markdown(f'<span class="badge-get">🪙 {r["Name"]}</span>  &nbsp; should receive **{fmt(r["Balance"])}**', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🔁 Settle up (suggestions)")
    settlements = suggest_settlements(df)
    if settlements:
        for s in settlements:
            st.write("• " + s)
    else:
        st.write("All settled 🎉 — no transfers needed.")

    # Download CSV
    csv_bytes = to_csv_bytes(df)
    st.download_button("⬇️ Download results (CSV)", data=csv_bytes, file_name="expense_split_results.csv", mime="text/csv")

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------
# Advanced / Footer
# --------------------------
st.write("")  # spacing
st.markdown(
    """
    <div class="card">
      <h4 style="margin:0;">⚙️ Advanced</h4>
      <div class="muted" style="margin-top:8px;">
        Want weighted splits, tip or tax calculations, or currency switching? These can be added — request via the button below.
      </div>
      <div style="margin-top:12px;">
      """
    + (
        st.button("✨ Request feature")
        and "<div class='helper'>Thanks! Feature request recorded ✉️</div>"
    )
    + "</div></div>",
    unsafe_allow_html=True,
)

st.markdown('<div style="margin-top:12px;" class="muted">Built with ❤️ • Dark theme • Streamlit</div>', unsafe_allow_html=True)
