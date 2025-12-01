# RAG Chatbot — Step-by-step Procedure

This `Readme.md` provides a concise step-by-step procedure to build a Retrieval-Augmented Generation (RAG) chatbot that accepts document uploads and uses a web-search fallback when local retrieval is insufficient.

Prerequisites
- Python 3.8+
- API keys: OpenAI (or another LLM provider) and Bing Search (or SerpAPI) for web fallback.

1) Project scaffold
- Create project folder:
```powershell
mkdir D:\Abdul_AI\RAG_Learn\rag_project
cd D:\Abdul_AI\RAG_Learn\rag_project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
- Create `requirements.txt` (minimal):
```
langchain
openai
faiss-cpu
sentence-transformers
fastapi
uvicorn[standard]
python-multipart
requests
python-dotenv
streamlit  # optional for UI
```
Install dependencies:
```powershell
pip install -r requirements.txt
```

2) Environment variables (example)
```powershell
setx OPENAI_API_KEY "sk-..."
setx BING_SEARCH_KEY "your_bing_key"
# For current session
$env:OPENAI_API_KEY="sk-..."
$env:BING_SEARCH_KEY="your_bing_key"
```

3) Ingestion pipeline (create `ingest.py`)
- Tasks:
  - Accept files (txt, pdf, docx)
  - Extract text (use `pdfplumber`, `python-docx` if needed)
  - Chunk text with overlap (e.g., 800 tokens, 200 overlap)
  - Create embeddings (OpenAI or sentence-transformers)
  - Store vectors in FAISS and metadata in a JSON/SQLite

4) Retriever (create `retriever.py`)
- Tasks:
  - Given a query, embed it
  - Search vector DB top-k (k=5)
  - Return chunks + similarity scores
  - Decide confidence threshold (e.g., top score < 0.55 triggers fallback)

5) QA orchestration (create `qa.py`)
- Build prompt template that includes sources and instructions to cite sources and say "I don't know" if not found.
- Call the LLM with retrieved context.
- Inspect answer for low-confidence markers (e.g., "I don't know", hedging language) to trigger fallback.

6) Web fallback (create `web_search.py`)
- Use Bing Web Search API or SerpAPI to fetch top N web results and snippets.
- Option A: Embed snippets and add them temporarily to FAISS, then re-run retrieval.
- Option B: Inject web snippets directly into the prompt as additional sources.
- Keep a rate limit and caching to save costs.

7) API / UI (create `api.py` or `app_streamlit.py`)
- Minimal FastAPI endpoints:
  - `POST /upload` -> accepts file, calls ingestion
  - `POST /ask` -> accepts question, runs `qa.answer(question)` and returns answer + sources + fallback flag
- Example run (FastAPI):
```powershell
uvicorn api:app --reload --port 8000
```
- Streamlit quick UI:
```powershell
streamlit run app_streamlit.py
```

8) Testing & evaluation
- Unit tests for ingestion, retrieval, and fallback trigger logic.
- Create a small test corpus with labeled Q/A pairs.
- Track metrics: fallback rate, accuracy (F1 or exact match), latency, API cost.

9) Deployment notes
- Store API keys as secrets (environment vars, Azure Key Vault, AWS Secrets Manager).
- Use a hosted vector DB for scale (Pinecone, Weaviate) for multi-user deployments.
- Dockerize the app for reproducible deployment.

10) Quick commands (copyable)
```powershell
# create venv and activate
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# install
pip install -r requirements.txt
# run API
uvicorn api:app --reload --port 8000
# run streamlit UI
streamlit run app_streamlit.py
```

11) Next steps & improvements
- Add a re-ranker (cross-encoder) for higher precision.
- Add session-based multi-turn context handling.
- Add provenance UI showing which document/snippet/URL the answer used.
- Add a verifier step that asks the LLM to check its answer against provided sources and output a confidence score.

---
Files you should add next (starter):
- `ingest.py`, `retriever.py`, `qa.py`, `web_search.py`, `api.py`, `.env.example`, `requirements.txt`

If you want, I can scaffold these files inside `D:\Abdul_AI\RAG_Learn\rag_project` now.

-------------------------------------------

3. How to Run

Install dependencies (example):

pip install streamlit openai faiss-cpu PyPDF2 numpy


Set your OpenAI API key (and optionally override model names):

setx OPENAI_API_KEY "your_key_here"      # Windows
export OPENAI_API_KEY="your_key_here"    # macOS / Linux

# Optional:
export EMBEDDING_MODEL="text-embedding-3-small"
export CHAT_MODEL="gpt-4o-mini"


(Optional) Integrate real web search:
In search_web(query: str), replace the stub with calls to Tavily, SerpAPI, Bing, etc., and return a text summary of results.

Run the app:

streamlit run app.py


Open the local URL shown in the terminal, upload documents, ask questions, and you’ll see:

Answer

Source label

Relevance / Accuracy / Completeness scores

Retrieved chunks & web snippets for transparency.