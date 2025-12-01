# Self-RAG — Explanation & Quick Start

This README explains the "Self-RAG" pattern used in this folder and shows how to run the provided example script `self_rag.py`.

**What is Self-RAG?**

- Self-RAG is a lightweight, pragmatic pattern that combines a standard LLM call with retrieval as a fallback.
- Instead of always performing retrieval first, the system: 1) asks the LLM directly for an answer, and 2) if the answer appears low-confidence or too short, runs a retrieval-augmented query to improve the response.
- This reduces unnecessary retrieval calls and latency/cost when the LLM can answer confidently from its internal knowledge, while still providing an evidence-backed fallback when needed.

**Why use Self-RAG?**

- Lower cost: fewer retrieval queries when the LLM already knows the answer.
- Simpler flow for quick queries and demonstrations.
- Still provides evidence and context when the LLM is unsure or returns a short/low-confidence answer.

**Pattern overview**

1. LLM-first: call the LLM directly with the question.
2. Check the LLM response for signs of low confidence (heuristics: phrase like "I’m not sure", very short answers, or explicit disclaimers).
3. If low confidence, run a retriever to fetch relevant document chunks and call a RetrievalQA (RAG) chain to produce an improved answer.
4. Return the best answer (either the original LLM answer or the retrieval-augmented answer).

**Files in this folder**

- `self_rag.py` — the example implementation of the Self-RAG pattern.
- `my_notes.txt` — (not included by default) small sample document you can create to test retrieval.
- `README.md` — this file.

**Prerequisites**

- Python 3.10+ recommended.
- An OpenAI API key if you plan to use `OpenAIEmbeddings` and `ChatOpenAI`.
- A virtual environment for isolation (recommended).
- Basic packages: `langchain`, `openai`, and FAISS (or another vector store).

Note: FAISS can be platform-specific. On Windows, try `pip install faiss-cpu` or use an alternative vector store if FAISS installation fails.

**Quick setup (PowerShell)**

```powershell
# change to the folder
cd 'D:\Abdul_AI\RAG_Learn\Self_RAG'

# create + activate venv (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# upgrade pip and install required packages
python -m pip install --upgrade pip
pip install langchain openai faiss-cpu

# or install from a requirements file if present
# pip install -r ..\requirements.txt
```

Set your OpenAI API key into the environment (temporary for current session):

```powershell
$env:OPENAI_API_KEY = 'sk-...'
```

**Run the example**

1. Ensure you have a small `my_notes.txt` in the same folder (or edit `self_rag.py` to point to a different file).
2. Run:

```powershell
python .\self_rag.py
```

You should see a printed "First attempt without retrieval:" output and either the LLM-only answer or a message indicating low confidence followed by the retrieval-based (improved) answer.

**Short example (behavior)**n
The script implements logic similar to:

```python
# pseudocode of the approach
first_answer = llm.predict(f"Q: {question}\nA:")
if "I'm not sure" in first_answer or len(first_answer) < 30:
    improved_answer = qa.run(question)  # retrieval + LLM
    return improved_answer
else:
    return first_answer
```

**Troubleshooting**

- If `faiss` fails to install on Windows, try `pip install faiss-cpu` or use `Chroma` / `sqlite` backed vector DB.
- If the script fails with embedding or OpenAI errors, confirm `OPENAI_API_KEY` is present and network access is allowed.
- If `llm.predict` isn’t available in your LangChain version, the example script contains a safe fallback that calls the LLM as a callable and extracts text.

**Next improvements you can make**

- Replace `OpenAIEmbeddings` with a local embedding model (sentence-transformers) to avoid OpenAI costs.
- Add a re-ranker (cross-encoder) to improve retrieval precision.
- Persist the FAISS index to disk to avoid rebuilding on each run.
- Add unit tests and a small `my_notes.txt` dataset to reproduce behavior consistently.

If you want, I can:
- create a small `my_notes.txt` example file here,
- add a `run_self_rag.ps1` script that automates venv creation and running, or
- adapt `self_rag.py` to use local embeddings to avoid external API calls.

---

Created to help you understand and run the Self-RAG example in this folder.