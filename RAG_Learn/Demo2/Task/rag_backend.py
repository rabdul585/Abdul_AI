"""
rag_backend.py
=====================================
FastAPI-based RAG backend with:
- PostgreSQL + pgvector vector store (documents/chunks/embeddings)
- Knowledge Graph (kg_nodes, kg_edges) in PostgreSQL
- Strict anti-hallucination logic:
    * Vector retrieval + KG retrieval
    * LLM answer generation constrained to context
    * Self-check step to validate support
    * Refusal when context is insufficient

Run locally (without Docker):
    export POSTGRES_URL=postgresql://USER:PASS@localhost:5432/DB
    export OPENAI_API_KEY=...
    uvicorn rag_backend:app --reload

In Docker:
    docker-compose up --build
"""

import os
import io
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

import psycopg2
from psycopg2.pool import SimpleConnectionPool
from pgvector.psycopg2 import register_vector
from pgvector import Vector

from PyPDF2 import PdfReader
import docx  # python-docx

# ---------------------------------------------------------
# Config & Globals
# ---------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-backend")

POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://rag_user:rag_password@localhost:5432/rag_db")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
DEFAULT_TOP_K = int(os.getenv("TOP_K", "5"))

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set. LLM calls will fail until you set it.")

client = OpenAI(api_key=OPENAI_API_KEY)

# Simple connection pool
pool: SimpleConnectionPool = SimpleConnectionPool(
    minconn=1,
    maxconn=5,
    dsn=POSTGRES_URL,
)

# Register pgvector type once
conn = pool.getconn()
try:
    with conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            # Documents
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    title TEXT,
                    original_filename TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                );
                """
            )
            # Chunks
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER,
                    content TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                );
                """
            )
            # Embeddings table (vector store)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chunk_embeddings (
                    chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
                    embedding vector(1536)
                );
                """
            )
            # Index for vector similarity
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_embedding
                ON chunk_embeddings USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
                """
            )
            # Knowledge graph nodes
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS kg_nodes (
                    id SERIAL PRIMARY KEY,
                    name TEXT UNIQUE,
                    type TEXT
                );
                """
            )
            # Knowledge graph edges
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS kg_edges (
                    id SERIAL PRIMARY KEY,
                    source_node_id INTEGER REFERENCES kg_nodes(id) ON DELETE CASCADE,
                    target_node_id INTEGER REFERENCES kg_nodes(id) ON DELETE CASCADE,
                    relation_type TEXT,
                    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_id INTEGER REFERENCES chunks(id) ON DELETE CASCADE
                );
                """
            )
        register_vector(conn)
finally:
    pool.putconn(conn)

# ---------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------

app = FastAPI(title="Vector + KG RAG Backend", version="0.1.0")

# CORS for UI in another container
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in real deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Pydantic models for API
# ---------------------------------------------------------

class IngestResponse(BaseModel):
    document_id: int
    filename: str
    num_chunks: int


class AskRequest(BaseModel):
    query: str
    top_k: int = DEFAULT_TOP_K


class SourceChunk(BaseModel):
    document_id: int
    chunk_id: int
    score: float
    content_preview: str


class KGNode(BaseModel):
    id: int
    name: str
    type: Optional[str] = None


class KGEdge(BaseModel):
    id: int
    source_node_id: int
    target_node_id: int
    relation_type: str
    document_id: Optional[int] = None
    chunk_id: Optional[int] = None


class AskResponse(BaseModel):
    answer: str
    refused: bool
    refusal_reason: Optional[str] = None
    confidence_score: float
    sources: List[SourceChunk]
    kg_nodes: List[KGNode]
    kg_edges: List[KGEdge]
    meta: Dict[str, Any]


# ---------------------------------------------------------
# Helpers: DB
# ---------------------------------------------------------

def get_conn():
    return pool.getconn()


def release_conn(conn):
    pool.putconn(conn)


# ---------------------------------------------------------
# Helpers: Text extraction & chunking
# ---------------------------------------------------------

def extract_text_from_file(upload_file: UploadFile) -> str:
    """Extract raw text from PDF, DOCX, or text."""
    filename = upload_file.filename.lower()
    content = upload_file.file.read()

    if filename.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(content))
        texts = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(texts)

    elif filename.endswith(".docx"):
        doc = docx.Document(io.BytesIO(content))
        return "\n".join([p.text for p in doc.paragraphs])

    else:
        # assume utf-8 text file
        return content.decode("utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Simple character-based chunking with overlap."""
    tokens = list(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = "".join(tokens[start:end])
        chunks.append(chunk)
        if end == len(tokens):
            break
        start = end - overlap
    return chunks


# ---------------------------------------------------------
# Helpers: Embeddings & LLM
# ---------------------------------------------------------

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Call OpenAI embedding API."""
    if not texts:
        return []
    response = client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=texts,
    )
    # Return in same order
    return [item.embedding for item in response.data]


def call_llm_for_answer(system_prompt: str, user_prompt: str) -> str:
    """Generic LLM call for answer generation."""
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,  # Critical for hallucination reduction
    )
    return resp.choices[0].message.content.strip()


def call_llm_for_json(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """LLM call expecting JSON object in response."""
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )
    content = resp.choices[0].message.content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        logger.warning("LLM JSON parse failed; raw: %s", content[:2000])
        return {}


# ---------------------------------------------------------
# Helpers: KG extraction using LLM
# ---------------------------------------------------------

def extract_entities_and_relations(text: str) -> List[Dict[str, Any]]:
    """
    Use LLM to extract triples (entity1, relation, entity2, type1, type2).
    Returned format: {"triples": [{"head": "...", "head_type": "...", "relation": "...",
                                  "tail": "...", "tail_type": "..."}]}
    """
    system_prompt = (
        "You are an information extraction system. "
        "Extract factual relationships as triples from the given text. "
        "Return JSON with key 'triples', value is a list of objects with keys: "
        "head, head_type, relation, tail, tail_type. "
        "Use brief type labels like PERSON, ORG, PLACE, CONCEPT."
    )
    user_prompt = f"Text:\n{text}\n\nReturn triples JSON only."

    data = call_llm_for_json(system_prompt, user_prompt)
    triples = data.get("triples", [])
    if not isinstance(triples, list):
        return []
    return triples


def ensure_node(conn, name: str, type_: str) -> int:
    """Insert or get KG node id."""
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM kg_nodes WHERE name = %s;", (name,))
        row = cur.fetchone()
        if row:
            return row[0]
        cur.execute(
            "INSERT INTO kg_nodes (name, type) VALUES (%s, %s) RETURNING id;",
            (name, type_),
        )
        node_id = cur.fetchone()[0]
    return node_id


def insert_kg_triples(conn, document_id: int, chunk_id: int, triples: List[Dict[str, Any]]):
    """Store KG triples referencing document + chunk for traceability."""
    with conn:
        with conn.cursor() as cur:
            for t in triples:
                head = t.get("head")
                tail = t.get("tail")
                relation = t.get("relation")
                head_type = t.get("head_type", "UNKNOWN")
                tail_type = t.get("tail_type", "UNKNOWN")
                if not head or not tail or not relation:
                    continue
                head_id = ensure_node(conn, head, head_type)
                tail_id = ensure_node(conn, tail, tail_type)
                cur.execute(
                    """
                    INSERT INTO kg_edges (source_node_id, target_node_id, relation_type, document_id, chunk_id)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING;
                    """,
                    (head_id, tail_id, relation, document_id, chunk_id),
                )


# ---------------------------------------------------------
# Retrieval: Vector store & KG
# ---------------------------------------------------------

def retrieve_from_vector_store(query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    """Semantic search from pgvector."""
    query_emb = embed_texts([query])[0]
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    c.id,
                    c.document_id,
                    c.content,
                    1 - (ce.embedding <=> %s::vector) AS score  -- cosine similarity to score
                FROM chunks c
                JOIN chunk_embeddings ce ON ce.chunk_id = c.id
                ORDER BY ce.embedding <=> %s::vector
                LIMIT %s;
                """,
                (Vector(query_emb), Vector(query_emb), top_k),
            )
            rows = cur.fetchall()
        results = []
        for chunk_id, doc_id, content, score in rows:
            results.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": doc_id,
                    "content": content,
                    "score": float(score),
                }
            )
        return results
    finally:
        release_conn(conn)


def extract_query_entities(query: str, chunks: List[Dict[str, Any]]) -> List[str]:
    """Use LLM to extract entity names from query + top chunks."""
    text_for_entities = query + "\n\n" + "\n\n".join(
        c["content"][:500] for c in chunks  # limit to keep prompt small
    )
    system_prompt = (
        "You are an NER system. Extract key entities from the text. "
        "Return JSON with key 'entities' as a list of unique string names."
    )
    data = call_llm_for_json(system_prompt, text_for_entities)
    entities = data.get("entities", [])
    if not isinstance(entities, list):
        return []
    # normalize
    return list({e.strip() for e in entities if isinstance(e, str) and e.strip()})


def retrieve_from_knowledge_graph(entities: List[str], max_hops: int = 2) -> Tuple[List[KGNode], List[KGEdge]]:
    """
    Retrieve KG nodes & edges related to the entities.
    For simplicity, we fetch:
      - nodes whose name is in entities
      - edges incident to those nodes (1-hop), optionally their connected nodes.
    """
    if not entities:
        return [], []

    conn = get_conn()
    try:
        node_ids = []
        nodes: Dict[int, KGNode] = {}
        edges: Dict[int, KGEdge] = {}

        with conn.cursor() as cur:
            # base nodes
            cur.execute(
                "SELECT id, name, type FROM kg_nodes WHERE name = ANY(%s);",
                (entities,),
            )
            for nid, name, type_ in cur.fetchall():
                node_ids.append(nid)
                nodes[nid] = KGNode(id=nid, name=name, type=type_)

            if not node_ids:
                return list(nodes.values()), list(edges.values())

            # edges 1-hop
            cur.execute(
                """
                SELECT id, source_node_id, target_node_id, relation_type, document_id, chunk_id
                FROM kg_edges
                WHERE source_node_id = ANY(%s) OR target_node_id = ANY(%s);
                """,
                (node_ids, node_ids),
            )
            for row in cur.fetchall():
                eid, sid, tid, rtype, doc_id, chunk_id = row
                edges[eid] = KGEdge(
                    id=eid,
                    source_node_id=sid,
                    target_node_id=tid,
                    relation_type=rtype,
                    document_id=doc_id,
                    chunk_id=chunk_id,
                )
                # also add connected nodes
                for nid in (sid, tid):
                    if nid not in nodes:
                        cur.execute(
                            "SELECT id, name, type FROM kg_nodes WHERE id = %s;",
                            (nid,),
                        )
                        r2 = cur.fetchone()
                        if r2:
                            nid2, name2, type2 = r2
                            nodes[nid2] = KGNode(id=nid2, name=name2, type=type2)

        return list(nodes.values()), list(edges.values())
    finally:
        release_conn(conn)


# ---------------------------------------------------------
# RAG core: answer_question with self-check
# ---------------------------------------------------------

def build_context(chunks: List[Dict[str, Any]], kg_nodes: List[KGNode], kg_edges: List[KGEdge]) -> str:
    """Create unified text context for LLM (vector + KG)."""
    context_parts = []

    # Vector store context
    context_parts.append("=== DOCUMENT CHUNKS ===")
    for c in chunks:
        context_parts.append(
            f"[Doc {c['document_id']} | Chunk {c['chunk_id']} | Score {c['score']:.3f}]\n{c['content']}\n"
        )

    # KG context
    context_parts.append("\n=== KNOWLEDGE GRAPH FACTS ===")
    node_map = {n.id: n for n in kg_nodes}
    for e in kg_edges:
        src = node_map.get(e.source_node_id)
        tgt = node_map.get(e.target_node_id)
        if not src or not tgt:
            continue
        context_parts.append(
            f"(Edge {e.id}) [{src.name} --{e.relation_type}--> {tgt.name}] "
            f"(Doc {e.document_id}, Chunk {e.chunk_id})"
        )

    return "\n".join(context_parts)


def generate_candidate_answer(query: str, context: str) -> str:
    """
    Primary answer generation step.

    *** ZERO-HALLUCINATION HOOK ***
    System prompt explicitly restricts answer to provided context and graph.
    """
    system_prompt = (
        "You are a rigorous QA assistant grounded ONLY in the supplied CONTEXT and GRAPH FACTS.\n"
        "RULES:\n"
        "1. Use ONLY the information in the provided context and graph.\n"
        "2. If the answer is not directly supported, say:\n"
        "   'I don’t know based on the current knowledge base.'\n"
        "3. Do NOT invent new facts, numbers, or names.\n"
        "4. When you state a fact, whenever possible include citations in the form "
        "[Doc X, Chunk Y].\n"
        "5. If there are conflicting facts, mention the conflict and do NOT choose a side.\n"
    )
    user_prompt = f"CONTEXT AND GRAPH:\n{context}\n\nUSER QUESTION:\n{query}\n\nAnswer:"
    return call_llm_for_answer(system_prompt, user_prompt)


def self_check_answer(query: str, context: str, candidate_answer: str) -> Dict[str, Any]:
    """
    Self-check step: ask LLM to evaluate support of each claim.
    Returns a JSON object like:
    {
      "overall_support_score": 0.0-1.0,
      "unsupported_claims": [...],
      "corrected_answer": "..."
    }

    *** ZERO-HALLUCINATION HOOK ***
    This step identifies unsupported claims and rewrites the answer to remove them.
    """
    system_prompt = (
        "You are a fact-checker. Your job is to verify whether a candidate answer "
        "is fully supported by the given context and graph.\n"
        "You must NOT introduce new facts.\n"
        "Return a JSON object with keys:\n"
        "  - overall_support_score (0.0 to 1.0)\n"
        "  - unsupported_claims (list of strings)\n"
        "  - corrected_answer (string) where all unsupported claims are removed "
        "    or clearly marked as 'Not supported by current knowledge base'.\n"
    )
    user_prompt = (
        f"CONTEXT AND GRAPH:\n{context}\n\n"
        f"USER QUESTION:\n{query}\n\n"
        f"CANDIDATE ANSWER:\n{candidate_answer}\n"
        "Now evaluate and return JSON as specified."
    )
    result = call_llm_for_json(system_prompt, user_prompt)
    # Provide defaults if something missing
    result.setdefault("overall_support_score", 0.0)
    result.setdefault("unsupported_claims", [])
    result.setdefault("corrected_answer", candidate_answer)
    return result


def compute_confidence(retrieved_chunks: List[Dict[str, Any]], support_score: float, kg_nodes: List[KGNode]) -> float:
    """
    Simple heuristic for confidence:
    - avg retrieval score (0-1)
    - LLM support score (0-1)
    - presence of KG evidence (0 or 1)
    """
    if not retrieved_chunks:
        avg_retrieval = 0.0
    else:
        avg_retrieval = sum(c["score"] for c in retrieved_chunks) / len(retrieved_chunks)
    kg_boost = 1.0 if kg_nodes else 0.5  # small bump if KG used

    confidence = (0.4 * avg_retrieval) + (0.5 * support_score) + (0.1 * kg_boost)
    # clamp
    confidence = max(0.0, min(1.0, confidence))
    return confidence


def answer_question(query: str, top_k: int = DEFAULT_TOP_K) -> AskResponse:
    """
    Full RAG flow:
    1. Vector search
    2. Entity extraction
    3. KG retrieval
    4. Unified context
    5. LLM candidate answer (strict grounding)
    6. Self-check/verifier step
    7. Confidence + possible refusal

    *** VECTOR + KG FUSION ***
    Happens in steps 2-4 where we combine semantic chunks and KG facts.

    *** ZERO-HALLUCINATION POLICY ***
    - Candidate generation constrained by context
    - Self-check removes unsupported claims
    - If no relevant context or low support, we refuse.
    """
    # 1. Vector search
    chunks = retrieve_from_vector_store(query, top_k=top_k)
    logger.info("Retrieved %d chunks for query.", len(chunks))

    if not chunks:
        # No grounding available at all
        return AskResponse(
            answer="I don’t have enough information in the current knowledge base to answer that reliably.",
            refused=True,
            refusal_reason="No relevant chunks retrieved.",
            confidence_score=0.0,
            sources=[],
            kg_nodes=[],
            kg_edges=[],
            meta={"stage": "no_chunks"},
        )

    # 2. Entity extraction
    entities = extract_query_entities(query, chunks)
    logger.info("Extracted entities: %s", entities)

    # 3. KG retrieval
    kg_nodes, kg_edges = retrieve_from_knowledge_graph(entities, max_hops=2)
    logger.info("KG retrieval: %d nodes, %d edges", len(kg_nodes), len(kg_edges))

    # 4. Unified context
    context = build_context(chunks, kg_nodes, kg_edges)

    # 5. Candidate answer
    candidate_answer = generate_candidate_answer(query, context)

    # 6. Self-check
    self_check = self_check_answer(query, context, candidate_answer)
    support_score = float(self_check.get("overall_support_score", 0.0))
    corrected_answer = self_check.get("corrected_answer", candidate_answer)
    unsupported_claims = self_check.get("unsupported_claims", [])

    # 7. Confidence + possible refusal
    confidence = compute_confidence(chunks, support_score, kg_nodes)

    refused = False
    refusal_reason = None
    final_answer = corrected_answer

    # Refusal policy:
    # - low support score
    # - or very low confidence
    if support_score < 0.4 or confidence < 0.4:
        refused = True
        refusal_reason = (
            "Self-check indicates that the answer is not sufficiently supported by the knowledge base."
        )
        final_answer = "I don’t have enough information in the current knowledge base to answer that reliably."

    sources = [
        SourceChunk(
            document_id=c["document_id"],
            chunk_id=c["chunk_id"],
            score=c["score"],
            content_preview=c["content"][:200],
        )
        for c in chunks
    ]

    return AskResponse(
        answer=final_answer,
        refused=refused,
        refusal_reason=refusal_reason,
        confidence_score=confidence,
        sources=sources,
        kg_nodes=kg_nodes,
        kg_edges=kg_edges,
        meta={
            "support_score": support_score,
            "unsupported_claims": unsupported_claims,
            "candidate_answer": candidate_answer,
            "entities": entities,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


# ---------------------------------------------------------
# Ingestion pipeline
# ---------------------------------------------------------

def ingest_document(file: UploadFile) -> IngestResponse:
    """
    Full ingest for a single file:
    - Extract text
    - Chunk
    - Embed
    - Store chunks + embeddings
    - Extract KG triples per chunk and populate KG tables
    """
    text = extract_text_from_file(file)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from file.")

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="Document has no content.")

    # Create document
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO documents (title, original_filename) VALUES (%s, %s) RETURNING id;",
                    (file.filename, file.filename),
                )
                document_id = cur.fetchone()[0]

                # Insert chunks
                chunk_ids = []
                for idx, content in enumerate(chunks):
                    cur.execute(
                        """
                        INSERT INTO chunks (document_id, chunk_index, content)
                        VALUES (%s, %s, %s)
                        RETURNING id;
                        """,
                        (document_id, idx, content),
                    )
                    cid = cur.fetchone()[0]
                    chunk_ids.append(cid)

        # Embeddings (outside transaction to avoid long locks)
        embeddings = embed_texts(chunks)

        with conn:
            with conn.cursor() as cur:
                for cid, emb in zip(chunk_ids, embeddings):
                    cur.execute(
                        """
                        INSERT INTO chunk_embeddings (chunk_id, embedding)
                        VALUES (%s, %s)
                        ON CONFLICT (chunk_id) DO UPDATE SET embedding = EXCLUDED.embedding;
                        """,
                        (cid, Vector(emb)),
                    )

        # KG extraction per chunk (can be optimized in batches if needed)
        with conn:
            for cid, content in zip(chunk_ids, chunks):
                triples = extract_entities_and_relations(content)
                if triples:
                    insert_kg_triples(conn, document_id, cid, triples)

        return IngestResponse(
            document_id=document_id,
            filename=file.filename,
            num_chunks=len(chunks),
        )
    finally:
        release_conn(conn)


# ---------------------------------------------------------
# FastAPI routes
# ---------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(file: UploadFile = File(...)):
    """
    Upload a document for ingestion into the vector store & KG.

    UI calls this endpoint with multipart/form-data.
    """
    logger.info("Ingesting file: %s", file.filename)
    return ingest_document(file)


@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(request: AskRequest):
    """
    Main question answering endpoint.

    Body:
      { "query": "...", "top_k": 5 }

    Returns:
      - answer (or refusal message)
      - refused flag & reason
      - confidence_score
      - sources (document & chunk IDs)
      - KG nodes/edges used
      - meta (debug info like support_score, etc.)
    """
    logger.info("Received query: %s", request.query)
    return answer_question(request.query, request.top_k)
