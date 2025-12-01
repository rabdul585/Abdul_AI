OPTION 1 – Run directly with Python (no Docker)
1️⃣ Open Command Prompt in your project folder
Win + R  →  type: cmd  →  Enter


Then:

cd /d D:\RAG_Project


(Adjust path to your real folder.)

2️⃣ Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate


You should see (venv) at the start of the prompt.

3️⃣ Install required Python packages

From the same folder (with venv active):

pip install fastapi uvicorn[standard] streamlit python-multipart pydantic psycopg2-binary SQLAlchemy pgvector python-dotenv openai python-docx PyPDF2 requests


If something fails, run Command Prompt as Administrator and retry.

4️⃣ Start PostgreSQL with pgvector (via Docker, simplest way)

If you already have Postgres with pgvector installed, you can skip this and just set POSTGRES_URL correctly.

Otherwise, from another Command Prompt window:

docker run --name pgvector ^
  -e POSTGRES_USER=rag_user ^
  -e POSTGRES_PASSWORD=rag_password ^
  -e POSTGRES_DB=rag_db ^
  -p 5432:5432 ^
  -d ankane/pgvector


This starts Postgres on port 5432 with pgvector enabled.

Database URL (we’ll use it next):
postgresql://rag_user:rag_password@localhost:5432/rag_db

5️⃣ Set environment variables (in the venv CMD window)

In the window where your venv is active:

set POSTGRES_URL=postgresql://rag_user:rag_password@localhost:5432/rag_db
set OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
set OPENAI_MODEL=gpt-4.1-mini
set OPENAI_EMBEDDING_MODEL=text-embedding-3-small


Replace sk-xxxxxxxx... with your actual OpenAI key.

You can also (optional) tune chunking:

set CHUNK_SIZE=800
set CHUNK_OVERLAP=150
set TOP_K=5

6️⃣ Run the backend with Uvicorn

Still in the same CMD (venv active, env vars set):

uvicorn rag_backend:app --reload --host 0.0.0.0 --port 8000


You should see logs like:

INFO:     Uvicorn running on http://0.0.0.0:8000


Test in browser: http://localhost:8000/docs

7️⃣ Run the UI with Streamlit

Open a second Command Prompt:

cd /d D:\RAG_Project
venv\Scripts\activate
set BACKEND_URL=http://localhost:8000
streamlit run ui_app.py


Streamlit will start on: http://localhost:8501

Now in your browser:

Go to http://localhost:8501

Use Document Ingestion tab to upload files

Use Ask a Question tab to query

OPTION 2 – Run everything via Docker / docker-compose from CMD

If you want to use the Dockerfile and docker-compose.yml I gave you:

1️⃣ Open Command Prompt in project folder
cd /d D:\RAG_Project

2️⃣ Set your OpenAI API key (so docker-compose can see it)
set OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx


(Optional) You can also create a .env file instead:

.env (in D:\RAG_Project):

OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx


docker-compose will automatically read it.

3️⃣ Build and start containers

From the same folder:

docker-compose up --build


What this does:

Starts postgres with pgvector

Builds and runs rag-backend (FastAPI on port 8000)

Builds and runs rag-ui (Streamlit on port 8501)

When logs stabilize, open:

UI: http://localhost:8501

Backend docs (optional): http://localhost:8000/docs

To stop everything: press Ctrl + C in that window, then:

docker-compose down

Quick Checklist (Command Prompt)

Run with Python (no Docker):

cd /d D:\RAG_Project

python -m venv venv

venv\Scripts\activate

pip install ... (libs list above)

Start pgvector container (one time): docker run ... ankane/pgvector

set POSTGRES_URL=...

set OPENAI_API_KEY=...

Backend: uvicorn rag_backend:app --reload --port 8000

New CMD → activate venv → set BACKEND_URL=http://localhost:8000

UI: streamlit run ui_app.py

Run with Docker / docker-compose:

cd /d D:\RAG_Project

set OPENAI_API_KEY=...

docker-compose up --build

Open http://localhost:8501