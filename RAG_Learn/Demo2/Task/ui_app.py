"""
ui_app.py
=====================================
Streamlit UI for the Vector + KG RAG backend.

Features:
- Document upload -> /ingest
- Query input -> /ask
- Display:
    * final answer
    * refusal badge (if any)
    * confidence score
    * sources (Doc/Chunk)
    * KG entities and relations

Configuration:
- BACKEND_URL env var (e.g. http://localhost:8000 or http://rag-backend:8000 in Docker)

Run locally:
    export BACKEND_URL=http://localhost:8000
    streamlit run ui_app.py

In Docker:
    docker-compose up --build
"""

import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Vector + KG RAG", layout="wide")

st.sidebar.title("Configuration")
backend_url = st.sidebar.text_input("Backend URL", BACKEND_URL)
if backend_url.endswith("/"):
    backend_url = backend_url[:-1]

st.sidebar.markdown("---")
st.sidebar.markdown("**Info**")
st.sidebar.write("Backend health check:")

try:
    resp = requests.get(f"{backend_url}/health", timeout=3)
    if resp.status_code == 200:
        st.sidebar.success("Backend is up ✅")
    else:
        st.sidebar.error(f"Backend responded with status {resp.status_code}")
except Exception as e:
    st.sidebar.error(f"Backend not reachable: {e}")

st.title("🔎 Vector + Knowledge Graph RAG Demo")

tab_ingest, tab_query = st.tabs(["📄 Document Ingestion", "❓ Ask a Question"])

# ---------------------------------------------------------
# Ingestion UI
# ---------------------------------------------------------
with tab_ingest:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF / DOCX / TXT files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Ingest Selected Documents"):
        for file in uploaded_files:
            st.write(f"Uploading **{file.name}**...")
            files = {"file": (file.name, file.getvalue(), file.type)}
            try:
                r = requests.post(f"{backend_url}/ingest", files=files)
                if r.status_code == 200:
                    data = r.json()
                    st.success(
                        f"Ingested {data['filename']} as document ID {data['document_id']} "
                        f"with {data['num_chunks']} chunks."
                    )
                else:
                    st.error(f"Failed to ingest {file.name}: {r.text}")
            except Exception as e:
                st.error(f"Error ingesting {file.name}: {e}")


# ---------------------------------------------------------
# Query UI
# ---------------------------------------------------------
with tab_query:
    st.header("Ask the Knowledge Base")

    query = st.text_area("Enter your question", height=120)
    top_k = st.number_input("Top-K chunks", min_value=1, max_value=20, value=5, step=1)

    if st.button("Ask"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Querying RAG backend..."):
                try:
                    payload = {"query": query, "top_k": top_k}
                    r = requests.post(f"{backend_url}/ask", json=payload, timeout=60)
                except Exception as e:
                    st.error(f"Error calling backend: {e}")
                    r = None

            if r is not None:
                if r.status_code != 200:
                    st.error(f"Backend error: {r.status_code} - {r.text}")
                else:
                    data = r.json()

                    # Top row: answer + confidence
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.subheader("Answer")
                        st.write(data["answer"])

                    with col2:
                        st.subheader("Confidence")
                        conf = data["confidence_score"]
                        st.metric("Confidence Score", f"{conf:.2f}")

                        if data.get("refused", False):
                            st.error("Refusal: Model chose not to answer.")
                            if data.get("refusal_reason"):
                                st.write(data["refusal_reason"])
                        else:
                            st.success("Answer is grounded.")

                    st.markdown("---")

                    # Sources
                    st.subheader("Sources (Vector Store)")
                    sources = data.get("sources", [])
                    if not sources:
                        st.info("No sources returned.")
                    else:
                        for s in sources:
                            with st.expander(
                                f"Doc {s['document_id']} | Chunk {s['chunk_id']} | Score {s['score']:.3f}"
                            ):
                                st.write(s["content_preview"])

                    # KG Evidence
                    st.markdown("---")
                    st.subheader("Knowledge Graph Evidence")

                    kg_nodes = data.get("kg_nodes", [])
                    kg_edges = data.get("kg_edges", [])

                    col_nodes, col_edges = st.columns(2)

                    with col_nodes:
                        st.markdown("**Nodes (Entities)**")
                        if not kg_nodes:
                            st.info("No KG nodes used.")
                        else:
                            for n in kg_nodes:
                                st.write(f"- **{n['name']}** (ID {n['id']}, type: {n.get('type')})")

                    with col_edges:
                        st.markdown("**Edges (Relations)**")
                        if not kg_edges:
                            st.info("No KG edges used.")
                        else:
                            for e in kg_edges:
                                st.write(
                                    f"- Edge {e['id']}: {e['source_node_id']} --{e['relation_type']}--> {e['target_node_id']} "
                                    f"(Doc {e.get('document_id')}, Chunk {e.get('chunk_id')})"
                                )

                    # Meta / Debug info
                    st.markdown("---")
                    with st.expander("Debug / Evaluation Metadata"):
                        meta = data.get("meta", {})
                        st.json(meta)

    st.markdown(
        """
        **Refusal behavior:**  
        If the system cannot find sufficient, consistent evidence in the vector store and KG,  
        it will explicitly refuse to answer and show a warning badge above.
        """
    )
