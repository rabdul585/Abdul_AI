# RAG Learn — Project README

This file documents the folder structure and gives a clear step-by-step procedure to set up and run the material in `D:\Abdul_AI\RAG_Learn`.

## Project Purpose
This workspace contains learning materials, demos, and example code for Retrieval-Augmented Generation (RAG). Use it to experiment with document ingestion, embeddings, vector stores, retrieval, and LLM orchestration.

## Folder Structure (in this workspace)
- `D:\Abdul_AI\RAG_Learn` — root for this learning project
	- `demo_learn.py` — small runnable demo script that illustrates core RAG steps.
	- `learning.md` — this file (overview, setup steps, usage instructions).
	- `pre_buildrequirement.md` — notes and pre-requirements for the environment.
	- `Readme.md` — (if present) general README for broader context.
	- `requirements.txt` — Python dependencies for demos in this folder.

Other related folders in your broader workspace (quick reference):
- `D:\Abdul_AI\AI` — various small Python apps (gym workout, water tracker).
- `D:\Abdul_AI\Demo\Knowledge-Graph-RAG` — complete demo project with pipeline, queries, and docker setup.
- `D:\Abdul_AI\KG_RAG\Knowledge-Graph-RAG` — copy of the demo project (alternate location).

Only the `RAG_Learn` folder is covered by the step-by-step instructions below; references to other folders are for exploration.

## Quick Prerequisites
- Python 3.10 or later.
- Git (optional, if you track changes remotely).
- For Windows PowerShell (pwsh) users: this guide uses PowerShell commands.

## Step-by-step Setup (PowerShell)
1. Open PowerShell and change to the project folder:

```powershell
cd 'D:\Abdul_AI\RAG_Learn'
```

2. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Upgrade pip and install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you plan to run components in `Demo\Knowledge-Graph-RAG`, install that folder's requirements too:

```powershell
pip install -r ..\Demo\Knowledge-Graph-RAG\requirements.txt
```

4. Configure any API keys needed by your demos (example environment variables):

```powershell
$env:OPENAI_API_KEY = 'sk-...'
# or persist in your user env vars for longer-term use
```

5. Run the demo script:

```powershell
python .\demo_learn.py
```

The demo script should demonstrate a minimal RAG flow (ingest, embed, index, query). Check console logs for retrieved chunks and model responses.

## What to expect / Troubleshooting
- If installation fails on a package: check the Python version and install appropriate wheels for Windows.
- If `demo_learn.py` requires external services (OpenAI, Pinecone), ensure API keys and network access are configured.
- If embeddings or vector store usage is slow, you can switch to local lightweight models (see `pre_buildrequirement.md`).

## Recommended Workflow
- Start small: run `demo_learn.py` with a short local document to verify end-to-end behavior.
- Log retrieved chunks, similarity scores, and the final LLM answers for debugging.
- Iterate on chunk size and overlap when ingesting long documents.

## Next Steps / Improvements
- Add a simple `run_demo.ps1` script to automate venv creation and demo runs.
- Add a `tests/` directory with a few sample docs and queries for regression testing.
- Document which demo scripts require which API keys in `pre_buildrequirement.md`.

---
If you'd like, I can also:
- add a `run_demo.ps1` convenience script,
- update `requirements.txt` with any missing packages discovered while running the demo, or
- run `demo_learn.py` now (with your confirmation and available API keys).

Notes: keep this file as your quick start and update it with any environment-specific steps you discover.