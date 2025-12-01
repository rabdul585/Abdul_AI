import os
import json
from typing import List, Dict, Tuple, Any

import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
import faiss

from openai import OpenAI
from dotenv import load_dotenv

# =========================================================
# Global config: load .env and initialize OpenAI client
# =========================================================

# Load environment variables from .env file in project root
load_dotenv()

# Read config from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Initialize OpenAI client (reads OPENAI_API_KEY from env)
client = OpenAI()

# =========================================================
# Helper: Web search stub
# =========================================================

def search_web(query: str) -> str:
    """
    Stub for web search.
    Replace this with a real search API call (Tavily, SerpAPI, Bing, etc.).

    Expected behavior:
      - Take `query` as input
      - Return a TEXT summary of relevant web results
    """
    # TODO: integrate your real web search here.
    # For now, this returns a placeholder string so the app runs end-to-end.
    return (
        f"[Stubbed web search results for query: '{query}']\n\n"
        "Replace `search_web` with an actual search API integration."
    )

# =========================================================
# Helper: Document loading
# =========================================================

def extract_text_from_pdf(uploaded_file) -> List[Tuple[int, str]]:
    """
    Extract text page-by-page from a PDF file.
    Returns a list of (page_index, page_text).
    """
    reader = PdfReader(uploaded_file)
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append((i, text))
    return pages


def extract_text_from_txt(uploaded_file) -> List[Tuple[int, str]]:
    """
    Extract text from a TXT file.
    Treats the whole file as a single "page" (page_index = 0).
    """
    content_bytes = uploaded_file.read()
    try:
        text = content_bytes.decode("utf-8", errors="ignore")
    except Exception:
        text = content_bytes.decode("latin-1", errors="ignore")
    return [(0, text)]


def load_documents(uploaded_files: List[Any]) -> List[Dict[str, Any]]:
    """
    Load and extract text from uploaded PDF/TXT files.

    Returns a list of document dicts:
    {
        "file_name": str,
        "pages": List[{"page_index": int, "text": str}]
    }
    """
    docs = []
    for f in uploaded_files:
        file_name = f.name
        if file_name.lower().endswith(".pdf"):
            pages_raw = extract_text_from_pdf(f)
        elif file_name.lower().endswith(".txt"):
            pages_raw = extract_text_from_txt(f)
        else:
            continue

        pages = []
        for page_idx, text in pages_raw:
            if text and text.strip():
                pages.append({
                    "page_index": page_idx,
                    "text": text
                })

        docs.append({
            "file_name": file_name,
            "pages": pages
        })
    return docs

# =========================================================
# Helper: Chunking
# =========================================================

def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[str]:
    """
    Simple character-based chunking with overlap.
    """
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        # move start forward, but keep overlap
        start += chunk_size - chunk_overlap
        if start <= 0:  # safety
            break
    return chunks


def build_chunks(
    docs: List[Dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int
) -> List[Dict[str, Any]]:
    """
    Build chunks with metadata from loaded documents.

    Returns list of:
    {
      "text": str,
      "metadata": {
          "file_name": str,
          "page_index": int,
          "chunk_index": int
      }
    }
    """
    all_chunks = []
    for doc in docs:
        file_name = doc["file_name"]
        for page in doc["pages"]:
            page_idx = page["page_index"]
            text = page["text"]
            raw_chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            for ci, ch in enumerate(raw_chunks):
                all_chunks.append({
                    "text": ch,
                    "metadata": {
                        "file_name": file_name,
                        "page_index": page_idx,
                        "chunk_index": ci
                    }
                })
    return all_chunks

# =========================================================
# Helper: Embeddings & FAISS
# =========================================================

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed a list of texts using the embedding model from .env.
    Returns a 2D numpy array of shape (len(texts), dim).
    """
    if not texts:
        return np.empty((0, 0), dtype="float32")

    resp = client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=texts
    )
    vectors = [item.embedding for item in resp.data]
    arr = np.array(vectors, dtype="float32")
    return arr


def normalize_embeddings(embs: np.ndarray) -> np.ndarray:
    """
    L2-normalize embeddings so inner product ~ cosine similarity.
    """
    if embs.size == 0:
        return embs
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
    return embs / norms


def build_faiss_index(embs: np.ndarray) -> faiss.Index:
    """
    Build an in-memory FAISS index for cosine-similarity retrieval.
    We store normalized vectors and use IndexFlatIP (inner product).
    """
    if embs.size == 0:
        return None
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    return index


def retrieve_chunks(
    query: str,
    index: faiss.Index,
    chunk_texts: List[str],
    chunk_metadata: List[Dict[str, Any]],
    top_k: int
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Retrieve top_k chunks for the given query.

    Returns (retrieved_chunks, avg_similarity) where retrieved_chunks is a list of:
    {
      "text": str,
      "metadata": {...},
      "score": float
    }
    """
    if index is None or not chunk_texts:
        return [], 0.0

    q_emb = embed_texts([query])
    if q_emb.size == 0:
        return [], 0.0

    q_emb_norm = normalize_embeddings(q_emb)
    scores, indices = index.search(q_emb_norm, top_k)
    scores = scores[0]
    indices = indices[0]

    retrieved = []
    valid_scores = []
    for score, idx in zip(scores, indices):
        if idx == -1:
            continue
        retrieved.append({
            "text": chunk_texts[idx],
            "metadata": chunk_metadata[idx],
            "score": float(score)
        })
        valid_scores.append(score)

    if valid_scores:
        avg_score = float(np.mean(valid_scores))
    else:
        avg_score = 0.0

    return retrieved, avg_score

# =========================================================
# Helper: LLM QA (using OPENAI_MODEL)
# =========================================================

def llm_answer_from_docs(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Ask the LLM to answer using ONLY the provided document chunks.

    If the answer is not answerable from these chunks, the LLM must respond
    with EXACTLY: 'NOT_ANSWERABLE'
    """
    context_snippets = []
    for rc in retrieved_chunks:
        meta = rc["metadata"]
        context_snippets.append(
            f"[File: {meta['file_name']} | Page: {meta['page_index']} | Chunk: {meta['chunk_index']}]\n"
            f"{rc['text']}"
        )

    context_str = "\n\n---\n\n".join(context_snippets) if context_snippets else "NO_CONTEXT"

    system_prompt = (
        "You are a precise Q&A assistant that answers ONLY from the provided context.\n"
        "If the answer CANNOT be inferred from the context, respond with EXACTLY:\n"
        "NOT_ANSWERABLE\n"
        "Do NOT hallucinate. Do NOT use external knowledge."
    )

    user_prompt = (
        f"User question:\n{query}\n\n"
        f"Context:\n{context_str}\n\n"
        "Remember: If the question cannot be answered from the context, respond ONLY with:\n"
        "NOT_ANSWERABLE"
    )

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    answer = resp.choices[0].message.content.strip()
    return answer


def llm_answer_fusion(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    web_results: str
) -> str:
    """
    LLM answer using both document context and web search results.

    NOTE: UPDATED SYSTEM PROMPT
    - Always produce THREE sections in the answer:
      1) 'From your documents:'
      2) 'From web search:'
      3) 'Final answer:'
    """
    context_snippets = []
    for rc in retrieved_chunks:
        meta = rc["metadata"]
        context_snippets.append(
            f"[File: {meta['file_name']} | Page: {meta['page_index']} | Chunk: {meta['chunk_index']}]\n"
            f"{rc['text']}"
        )

    docs_context = "\n\n---\n\n".join(context_snippets) if context_snippets else "NO_DOCUMENT_CONTEXT"

    # 🔴 CHANGED SYSTEM PROMPT HERE
    system_prompt = (
        "You are a helpful assistant that synthesizes answers from two sources:\n"
        "1) Uploaded DOCUMENT CONTEXT\n"
        "2) WEB SEARCH RESULTS\n\n"
        "Use BOTH when relevant. Prefer document context for document-specific details, "
        "and web results for general knowledge.\n\n"
        "IMPORTANT OUTPUT FORMAT:\n"
        "- Always structure your response in exactly three sections, in this order:\n"
        "  1) 'From your documents:' - Summarize only what is supported by DOCUMENT CONTEXT.\n"
        "  2) 'From web search:' - Summarize only what is supported by WEB SEARCH RESULTS.\n"
        "  3) 'Final answer:' - Provide a concise, helpful answer combining both when useful.\n"
        "- If a section has no relevant information, still include the heading and say "
        "'No relevant information found.'\n"
    )

    user_prompt = (
        f"User question:\n{query}\n\n"
        f"DOCUMENT CONTEXT:\n{docs_context}\n\n"
        f"WEB SEARCH RESULTS:\n{web_results}\n\n"
        "Follow the required three-section format."
    )

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    answer = resp.choices[0].message.content.strip()
    return answer

# =========================================================
# Helper: Evaluation (Relevance / Accuracy / Completeness)
# =========================================================

def evaluate_answer(
    query: str,
    context_snippets: List[str],
    web_snippets: List[str],
    answer: str
) -> Dict[str, float]:
    """
    LLM-based self-evaluation.

    Returns:
    {
      "relevance": float (0-1),
      "accuracy": float (0-1),
      "completeness": float (0-1)
    }
    """
    context_str = "\n\n---\n\n".join(context_snippets) if context_snippets else "NO_DOCUMENT_CONTEXT"
    web_str = "\n\n---\n\n".join(web_snippets) if web_snippets else "NO_WEB_CONTEXT"

    system_prompt = (
        "You are an evaluation assistant. You will receive:\n"
        "- A user query\n"
        "- A final answer\n"
        "- Supporting document snippets\n"
        "- Supporting web snippets\n\n"
        "You must rate THREE dimensions (0.0 to 1.0):\n"
        "1) relevance: How relevant are the provided contexts to the query?\n"
        "2) accuracy: How factually correct is the answer given the contexts?\n"
        "3) completeness: How fully does the answer address all parts of the query?\n\n"
        "Scoring rubric (guideline):\n"
        "- 0.0–0.3: Poor\n"
        "- 0.3–0.6: Partial / mixed\n"
        "- 0.6–0.9: Good\n"
        "- 0.9–1.0: Excellent\n\n"
        "IMPORTANT:\n"
        "- Output ONLY valid JSON with float values between 0.0 and 1.0.\n"
        "- JSON keys: relevance, accuracy, completeness.\n"
        '- Example: {\"relevance\": 0.8, \"accuracy\": 0.9, \"completeness\": 0.7}\n'
        "- Do NOT add any explanation outside the JSON. No markdown."
    )

    user_prompt = json.dumps({
        "query": query,
        "answer": answer,
        "document_snippets": context_str,
        "web_snippets": web_str
    })

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        raw = resp.choices[0].message.content.strip()
        scores = json.loads(raw)
        relevance = float(scores.get("relevance", 0.5))
        accuracy = float(scores.get("accuracy", 0.5))
        completeness = float(scores.get("completeness", 0.5))
    except Exception:
        # Fallback if parsing fails
        relevance = accuracy = completeness = 0.5

    return {
        "relevance": max(0.0, min(1.0, relevance)),
        "accuracy": max(0.0, min(1.0, accuracy)),
        "completeness": max(0.0, min(1.0, completeness))
    }

# =========================================================
# Streamlit: session state helpers
# =========================================================

def init_session_state():
    """
    Initialize session_state keys if not present.
    """
    if "docs" not in st.session_state:
        st.session_state.docs = []
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "chunk_texts" not in st.session_state:
        st.session_state.chunk_texts = []
    if "chunk_metadata" not in st.session_state:
        st.session_state.chunk_metadata = []
    if "faiss_index" not in st.session_state:
        st.session_state.faiss_index = None
    if "last_answer" not in st.session_state:
        st.session_state.last_answer = None
    if "last_source" not in st.session_state:
        st.session_state.last_source = None
    if "last_scores" not in st.session_state:
        st.session_state.last_scores = None
    if "last_retrieved" not in st.session_state:
        st.session_state.last_retrieved = []
    if "last_web_snippet" not in st.session_state:
        st.session_state.last_web_snippet = ""
    # 🔴 NEW: store document-only answer
    if "last_doc_answer" not in st.session_state:
        st.session_state.last_doc_answer = None


def build_knowledge_base(
    uploaded_files: List[Any],
    chunk_size: int,
    chunk_overlap: int
):
    """
    Load docs, create chunks, embed, and build FAISS index.
    Stores everything in st.session_state.
    """
    with st.spinner("Processing documents and building vector index..."):
        docs = load_documents(uploaded_files)
        chunks = build_chunks(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunk_texts = [c["text"] for c in chunks]
        chunk_metadata = [c["metadata"] for c in chunks]

        embeddings = embed_texts(chunk_texts)
        embeddings_norm = normalize_embeddings(embeddings)
        index = build_faiss_index(embeddings_norm)

        st.session_state.docs = docs
        st.session_state.chunks = chunks
        st.session_state.chunk_texts = chunk_texts
        st.session_state.chunk_metadata = chunk_metadata
        st.session_state.faiss_index = index

    st.success(f"Knowledge base built with {len(chunks)} chunks from {len(docs)} document(s).")

# =========================================================
# Streamlit main app
# =========================================================

def main():
    st.set_page_config(
        page_title="RAG Q&A Assistant",
        page_icon="🔎",
        layout="wide"
    )

    init_session_state()

    # Basic Google-like styling
    st.markdown(
        """
        <style>
        .main {
            background-color: #ffffff;
        }
        .search-box input {
            border-radius: 999px !important;
            padding: 0.75rem 1.25rem !important;
            border: 1px solid #e0e0e0 !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }
        .search-box input:focus {
            border-color: #4285f4 !important;
            box-shadow: 0 0 0 1px #4285f4;
        }
        .answer-card {
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid #e0e0e0;
            box-shadow: 0 2px 6px rgba(0,0,0,0.04);
            background-color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Warn if API key missing
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY is not set. Please add it to your .env file.")
        return

    # Title & subtitle (centered feel using columns)
    top_col1, top_col2, top_col3 = st.columns([1, 2, 1])
    with top_col2:
        st.markdown("### 🔎 RAG Q&A Assistant")
        st.caption("Ask questions about your uploaded documents. If not found, I’ll search the web.")

    st.write("")

    # Main layout
    left_col, right_col = st.columns([2, 1])

    # ---------------- Left column: documents + query ----------------
    with left_col:
        st.subheader("Documents & Query")

        uploaded_files = st.file_uploader(
            "Upload PDF or TXT files",
            type=["pdf", "txt"],
            accept_multiple_files=True
        )

        with st.expander("Advanced settings"):
            chunk_size = st.slider("Chunk size (characters)", min_value=500, max_value=3000, value=1000, step=100)
            chunk_overlap = st.slider("Chunk overlap (characters)", min_value=50, max_value=1000, value=200, step=50)
            top_k = st.slider("Top K chunks to retrieve", min_value=3, max_value=15, value=5, step=1)
            doc_relevance_threshold = st.slider(
                "Document relevance threshold to trigger web search",
                min_value=0.0, max_value=1.0, value=0.35, step=0.05
            )

        # Button to build/reset index
        if uploaded_files:
            if st.button("📚 Build / Rebuild Knowledge Base"):
                build_knowledge_base(uploaded_files, chunk_size, chunk_overlap)
        else:
            st.info(
                "Upload one or more documents to build a local knowledge base. "
                "If you don't upload anything, I'll answer via web search only (stub)."
            )

        st.write("")
        st.markdown("#### Ask a question")

        # Search bar style container
        with st.container():
            query = st.text_input(
                "",
                placeholder="Type your question here...",
                key="user_query",
                label_visibility="collapsed",
            )
            ask_col1, ask_col2 = st.columns([4, 1])
            with ask_col1:
                st.markdown(
                    '<div class="search-box"></div>',
                    unsafe_allow_html=True
                )
            with ask_col2:
                ask_button = st.button("Search", use_container_width=True)

        if ask_button and query.strip():
            # Core RAG pipeline for a single query
            faiss_index = st.session_state.faiss_index
            chunk_texts = st.session_state.chunk_texts
            chunk_metadata = st.session_state.chunk_metadata

            # 1) Retrieve chunks
            retrieved, avg_sim = retrieve_chunks(
                query=query,
                index=faiss_index,
                chunk_texts=chunk_texts,
                chunk_metadata=chunk_metadata,
                top_k=top_k
            )

            # 2) Decide whether docs are strong enough
            use_docs = bool(retrieved)
            doc_relevance = avg_sim  # 0–1 cosine-ish
            doc_answer = None
            final_answer = None
            source_label = None
            web_snippet = ""
            doc_only_context_snippets = [r["text"] for r in retrieved]

            # 3) First attempt: docs-only QA (if we have any)
            if use_docs:
                doc_answer = llm_answer_from_docs(query, retrieved)

            # 4) Decision: web search fallback & fusion
            need_web = False

            # Condition 1: no docs or low similarity
            if not retrieved or doc_relevance < doc_relevance_threshold:
                need_web = True

            # Condition 2: LLM explicitly says not answerable
            if doc_answer is not None and doc_answer.strip() == "NOT_ANSWERABLE":
                need_web = True

            if need_web:
                # Call stubbed web search
                web_snippet = search_web(query)

                # Fusion of docs + web
                final_answer = llm_answer_fusion(query, retrieved, web_snippet)

                if retrieved and doc_relevance > 0.15:
                    source_label = "Source: Uploaded documents + Web search"
                else:
                    source_label = "Source: Web search"
            else:
                # Docs are good & LLM answered from docs
                final_answer = doc_answer
                source_label = "Source: Uploaded documents"

            # 5) Evaluation (relevance / accuracy / completeness) on final answer
            web_snippets_list = [web_snippet] if web_snippet else []
            scores = evaluate_answer(query, doc_only_context_snippets, web_snippets_list, final_answer)

            # Store for display
            st.session_state.last_answer = final_answer
            st.session_state.last_source = source_label
            st.session_state.last_scores = scores
            st.session_state.last_retrieved = retrieved
            st.session_state.last_web_snippet = web_snippet
            # 🔴 NEW: store docs-only answer (RAG answer)
            if doc_answer and doc_answer.strip() != "NOT_ANSWERABLE":
                st.session_state.last_doc_answer = doc_answer
            else:
                st.session_state.last_doc_answer = None

    # ---------------- Right column: metrics & context inspectors ----------------
    with right_col:
        st.subheader("Answer Quality")

        if st.session_state.last_scores:
            scores = st.session_state.last_scores

            st.markdown("**Relevance**")
            st.progress(scores["relevance"])
            st.caption(f"{scores['relevance']:.2f} / 1.00")

            st.markdown("**Accuracy**")
            st.progress(scores["accuracy"])
            st.caption(f"{scores['accuracy']:.2f} / 1.00")

            st.markdown("**Completeness**")
            st.progress(scores["completeness"])
            st.caption(f"{scores['completeness']:.2f} / 1.00")
        else:
            st.info("Ask a question to see quality scores here.")

        st.write("")
        st.subheader("Context Inspectors")

        if st.session_state.last_retrieved:
            with st.expander("Retrieved document chunks"):
                for rc in st.session_state.last_retrieved:
                    meta = rc["metadata"]
                    st.markdown(
                        f"**File:** {meta['file_name']} | "
                        f"Page: {meta['page_index']} | "
                        f"Chunk: {meta['chunk_index']} | "
                        f"Score: {rc['score']:.3f}"
                    )
                    st.write(rc["text"])
                    st.markdown("---")
        else:
            st.caption("No document chunks used for the last answer.")

        if st.session_state.last_web_snippet:
            with st.expander("Web search snippet"):
                st.write(st.session_state.last_web_snippet)
        else:
            st.caption("No web search used for the last answer.")

    # Final answer panel at bottom
    st.write("")
    st.markdown("---")

    # 🔴 CHANGED: show BOTH RAG (docs) result and fusion/web result
    doc_ans = st.session_state.last_doc_answer
    final_ans = st.session_state.last_answer
    src_label = st.session_state.last_source

    if doc_ans or final_ans:
        # Show document-based answer if available
        if doc_ans:
            st.markdown("### Answer from uploaded documents (RAG)")
            st.markdown(
                f'<div class="answer-card">{doc_ans}</div>',
                unsafe_allow_html=True
            )

        # Show final / fused answer
        if final_ans:
            # If web was used, label clearly
            if src_label and "Web" in src_label:
                st.markdown("### Answer using web search + documents")
            else:
                st.markdown("### Final Answer")

            st.markdown(
                f'<div class="answer-card">{final_ans}</div>',
                unsafe_allow_html=True
            )
            if src_label:
                st.caption(src_label)


if __name__ == "__main__":
    main()
