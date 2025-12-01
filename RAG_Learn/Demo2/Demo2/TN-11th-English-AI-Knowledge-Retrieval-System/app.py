import streamlit as st
from hybrid_rag import hybrid_answer
from kg_visual import show_graph

# ----------------------------
# Page config (Material-ish)
# ----------------------------
st.set_page_config(
    page_title="TN 11th English Knowledge Retrieval System",
    layout="wide",
    page_icon="📘",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Material 3 CSS
# ----------------------------
st.markdown("""
<style>
:root {
  --primary: #1a73e8;
  --primary-weak: #e8f0fe;
  --surface: #ffffff;
  --surface-2: #f6f8fb;
  --text: #0f172a;
  --text-weak: #475569;
  --border: #e5e7eb;
  --success: #0f9d58;
  --warning: #f29900;
}

/* App background */
.stApp {
  background: var(--surface-2);
}

/* Main container width */
.block-container {
  padding-top: 1.2rem;
  padding-bottom: 2rem;
  max-width: 1200px;
}

/* Header */
.header-wrap {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 20px 22px;
  box-shadow: 0 2px 8px rgba(16,24,40,0.06);
  margin-bottom: 14px;
}
.header-title {
  font-size: 1.9rem;
  font-weight: 800;
  color: var(--text);
  margin-bottom: 2px;
}
.header-sub {
  font-size: 0.98rem;
  color: var(--text-weak);
  line-height: 1.5;
}
.header-meta {
  margin-top: 10px;
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}
.chip {
  font-size: 0.82rem;
  padding: 4px 10px;
  border-radius: 999px;
  background: var(--primary-weak);
  color: var(--primary);
  border: 1px solid #dbeafe;
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

/* Section cards */
.section-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 14px 16px;
  box-shadow: 0 1px 6px rgba(16,24,40,0.05);
}

/* Input styling */
.stTextInput>div>div>input {
  background: #fff;
  border-radius: 10px;
  border: 1px solid var(--border);
  padding: 0.7rem 0.9rem;
}

/* Buttons */
.stButton>button {
  background: var(--primary);
  color: white;
  border: none;
  border-radius: 10px;
  padding: 0.55rem 1rem;
  font-weight: 600;
  box-shadow: 0 2px 6px rgba(26,115,232,0.25);
}
.stButton>button:hover {
  background: #1662c4;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  gap: 8px;
}
.stTabs [data-baseweb="tab"] {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 8px 14px;
  font-weight: 600;
  color: var(--text-weak);
}
.stTabs [aria-selected="true"] {
  color: var(--text);
  border-bottom: 2px solid var(--primary);
}

/* Evidence expander */
.streamlit-expanderHeader {
  font-weight: 700 !important;
  color: var(--text);
}
hr {
  border-color: var(--border);
}
.small-muted {
  color: var(--text-weak);
  font-size: 0.9rem;
}
.badge-success {
  display:inline-block;
  font-size:0.82rem;
  padding:4px 10px;
  border-radius:999px;
  background:#e7f6ee;
  color:var(--success);
  border:1px solid #bfe8d0;
}
.badge-warn {
  display:inline-block;
  font-size:0.82rem;
  padding:4px 10px;
  border-radius:999px;
  background:#fff4e5;
  color:var(--warning);
  border:1px solid #ffe0b2;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Research / Lab Header (Style 3)
# ----------------------------
st.markdown("""
<div class="header-wrap">
  <div class="header-title">TN 11th English Knowledge Retrieval System</div>
  <div class="header-sub">
    Hybrid Vector Search + Knowledge Graph RAG for
    <b>Higher Secondary – First Year, English</b><br/>
    Resource PDF: Government of Tamil Nadu, Department of School Education
  </div>
  <div class="header-meta">
    <span class="chip">Zero-Hallucination Policy</span>
    <span class="chip">Neon pgvector</span>
    <span class="chip">Neo4j Knowledge Graph</span>
    <span class="chip">Team 01 — Natraj · Araul · Mohamed Munawfer</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# Sidebar (Material filter panel)
# ----------------------------
st.sidebar.markdown("### Filters")
chapter_filter = st.sidebar.selectbox(
    "Chapter scope",
    ["All", "Prose", "Poem", "Supplementary", "Unit", "Grammar", "General"],
    index=0
)

section_filter = st.sidebar.selectbox(
    "Section type (optional)",
    ["All", "Prose", "Poem", "Supplementary", "Grammar", "Warm-up", "Other"],
    index=0
)

top_k = st.sidebar.slider("Evidence chunks (k)", 3, 8, 5)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div class='small-muted'>The assistant answers only from textbook evidence. "
    "If evidence is not found, it will refuse.</div>",
    unsafe_allow_html=True
)

# ----------------------------
# Tabs
# ----------------------------
tab_ask, tab_evidence, tab_kg = st.tabs(
    ["💬 Ask Question", "🔍 Evidence", "🧠 Knowledge Graph"]
)

# Session state
if "last_query" not in st.session_state:
    st.session_state.last_query = None
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "last_docs" not in st.session_state:
    st.session_state.last_docs = []
if "last_kg" not in st.session_state:
    st.session_state.last_kg = ""

# ----------------------------
# Ask tab
# ----------------------------
with tab_ask:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)

    st.subheader("Ask any question from the textbook")
    query = st.text_input(
        "Enter your question",
        placeholder="Example: What is the theme of the poem 'The Earth'?"
    )

    colA, colB = st.columns([1, 5])
    with colA:
        ask_btn = st.button("Ask")

    st.markdown("</div>", unsafe_allow_html=True)

    if ask_btn and query:
        with st.spinner("Retrieving evidence and generating grounded answer..."):
            # Note: section_filter isn't a DB filter yet; it’s UI-level.
            # You can add true DB filtering later if needed.
            answer, docs, kg = hybrid_answer(
                query,
                chapter_filter=chapter_filter,
                return_evidence=True
            )

        st.session_state.last_query = query
        st.session_state.last_answer = answer
        st.session_state.last_docs = docs[:top_k]
        st.session_state.last_kg = kg

        # Grounding badge
        refused = "I don't have enough evidence" in answer
        badge = "<span class='badge-warn'>Insufficient evidence</span>" if refused \
                else "<span class='badge-success'>Grounded answer</span>"

        st.markdown(f"**Status:** {badge}", unsafe_allow_html=True)
        st.markdown("### Answer")
        st.write(answer)

        st.caption("Open the Evidence tab to see exactly what was used.")

# ----------------------------
# Evidence tab
# ----------------------------
with tab_evidence:
    st.subheader("Evidence used")

    if not st.session_state.last_query:
        st.warning("Ask a question first.")
    else:
        st.markdown(f"**Question:** {st.session_state.last_query}")
        st.markdown("---")

        st.markdown("#### Vector Evidence (Textbook chunks)")
        for i, d in enumerate(st.session_state.last_docs, start=1):
            meta = d.metadata or {}
            chap = meta.get("chapter", "General")
            sec_type = meta.get("section_type", "Unknown")
            unit = meta.get("unit", "")

            header = f"chunk-{i} • {chap}"
            if sec_type and sec_type != "Unknown":
                header += f" • {sec_type}"
            if unit:
                header += f" • {unit}"

            with st.expander(header):
                st.write(d.page_content)

        st.markdown("---")
        st.markdown("#### Knowledge Graph Evidence")
        st.text(st.session_state.last_kg or "None")

# ----------------------------
# KG tab
# ----------------------------
with tab_kg:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("### Knowledge Graph View")
    st.caption("Interactive Neo4j graph extracted from the textbook.")
    st.markdown("</div>", unsafe_allow_html=True)

    show_graph()
