# services/api/rag.py
"""
RAG API using:
- Gemini (google-generativeai) for SQL generation + answer composition
- Chroma (self-hosted) for vector store (embeddings added separately)
- sentence-transformers for query embeddings
- PostgreSQL (psycopg2) read-only access for SQL execution

Environment variables (examples):
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-1.5-flash
CHROMA_SERVER_HOST=localhost
CHROMA_SERVER_HTTP_PORT=8000
DATABASE_URL=postgresql://readonly:pwd@localhost:5432/argo
ALLOWED_TABLES=profiles,levels,measurements
EMBED_MODEL=all-MiniLM-L6-v2
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import re
import json
import chromadb
import google.generativeai as genai
import psycopg2
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

# --- .env loading ---
# This will load environment variables from a .env file in the same directory
from dotenv import load_dotenv
load_dotenv()
# --------------------


# -----------------------
# CONFIG / CLIENTS
# -----------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY env var not set")
genai.configure(api_key=GEMINI_API_KEY)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

CHROMA_HOST = os.getenv("CHROMA_SERVER_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_SERVER_HTTP_PORT", 8000))
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL env var not set")

ALLOWED_TABLES = set(os.getenv("ALLOWED_TABLES", "profiles,levels,measurements").split(","))

# create chroma client (HTTP) - updated to use HttpClient
chroma_client = chromadb.HttpClient(
    host=CHROMA_HOST,
    port=CHROMA_PORT
)
chroma_col = chroma_client.get_or_create_collection("floatchat")

# sentence-transformers embedding model (used for query embedding)
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
_embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# instantiate Gemini model object
_gemini = genai.GenerativeModel(GEMINI_MODEL)

# psycopg2 DB helper
def get_db_conn():
    return psycopg2.connect(DATABASE_URL, sslmode='prefer')

# -----------------------
# SQL safety / helpers
# -----------------------
DISALLOWED_SQL = re.compile(r'\b(insert|delete|update|drop|alter|create|truncate|copy|grant|revoke|vacuum|execute|pg_read_file)\b', re.I)

def clean_sql_output(output: str) -> str:
    """
    Extract the first SELECT statement from model output.
    """
    if not output:
        return output
    # Try to find the first SELECT ... ; or SELECT ... end-of-string
    m = re.search(r'(select\b.*?;)', output, re.I | re.S)
    if m:
        return m.group(1).strip()
    # fallback: extract from first 'select' to end
    m2 = re.search(r'(select\b.*)', output, re.I | re.S)
    if m2:
        text = m2.group(1).strip()
        if not text.endswith(';'):
            text = text + ';'
        return text
    return output.strip()

def find_tables_in_sql(sql: str) -> set:
    """
    Find candidate table names used in FROM / JOIN clauses.
    This is a heuristic and intentionally conservative.
    """
    tables = set()
    # from <table>
    for t in re.findall(r'\bfrom\s+([a-zA-Z_][a-zA-Z0-9_]*)', sql, re.I):
        tables.add(t)
    # join <table>
    for t in re.findall(r'\bjoin\s+([a-zA-Z_][a-zA-Z0-9_]*)', sql, re.I):
        tables.add(t)
    return tables

def validate_sql(sql: str) -> str:
    """
    Raise ValueError for disallowed SQL; otherwise ensure LIMIT and allowed tables
    """
    if DISALLOWED_SQL.search(sql):
        raise ValueError("Disallowed SQL command present.")
    if not re.match(r'^\s*select\b', sql, re.I):
        raise ValueError("Only SELECT queries allowed.")
    # enforce a LIMIT if not present
    if not re.search(r'\blimit\b', sql, re.I):
        sql = sql.rstrip(';') + " LIMIT 100;"
    found_tables = find_tables_in_sql(sql)
    disallowed = [t for t in found_tables if t not in ALLOWED_TABLES]
    if disallowed:
        raise ValueError(f"Query references disallowed table(s): {disallowed}")
    return sql

# -----------------------
# Schema fetch (cached)
# -----------------------
def fetch_schema():
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema='public'
            ORDER BY table_name, ordinal_position
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()
    except Exception as e:
        raise RuntimeError(f"Failed to fetch schema: {e}")
    schema = {}
    for table, col, dtype in rows:
        schema.setdefault(table, []).append({"column": col, "type": dtype})
    return schema

SCHEMA_CACHE = fetch_schema()

# -----------------------
# Prompt builders & LLM calls (Gemini)
# -----------------------
def build_sql_prompt(question: str, schema: Dict[str, Any], examples: List[Dict[str,str]] = None) -> str:
    schema_text = "\n".join([f"{t}: {', '.join([c['column'] for c in cols])}" for t, cols in schema.items() if t in ALLOWED_TABLES])
    examples_text = ""
    if examples:
        for ex in examples:
            examples_text += f"Q: {ex['q']}\nSQL: {ex['sql']}\n\n"
    prompt = f"""You are a SQL generator. Output exactly one PostgreSQL SELECT statement (no explanation) that answers the user's question. Use only the whitelisted tables/columns below and produce safe, read-only SQL. Always include a LIMIT if result might be large (max 100).

Schema (allowed tables only):
{schema_text}

{examples_text}
User question: \"\"\"{question}\"\"\"
Only output the SQL statement (no surrounding backticks, no commentary)."""
    return prompt

def call_llm_for_sql(prompt: str, temperature: float = 0.0) -> str:
    """
    Ask Gemini to generate SQL. Returns raw text.
    """
    # generate content (simple usage)
    resp = _gemini.generate_content(prompt)
    # resp may be object with .text
    text = getattr(resp, "text", None)
    if text is None:
        # fallback to string conversion
        text = str(resp)
    return text.strip()

def rag_compose_answer(question: str, sql_rows: List[Dict[str,Any]] = None, passages: List[Dict]=None) -> str:
    """
    Compose final answer using Gemini.
    We include short context: up to some characters from DB rows and top passages.
    """
    parts = []
    if sql_rows:
        # keep limited preview
        parts.append("DB_ROWS_PREVIEW:\n" + json.dumps(sql_rows, default=str)[:2000])
    if passages:
        for i,p in enumerate(passages, start=1):
            text_preview = p.get("text","")[:800]
            meta = p.get("meta",{})
            parts.append(f"[P{i}] {text_preview} (meta={json.dumps(meta)})")
    context = "\n\n".join(parts) if parts else "No extra context."
    compose_prompt = f"""You are an assistant that answers a user's question using the provided context (database rows and passages). Be concise, factual, and cite sources like [DB] for database rows and [P1], [P2] for passages.

Question:
{question}

Context:
{context}

Answer (short, include citations when relevant):"""
    resp = _gemini.generate_content(compose_prompt)
    return getattr(resp, "text", str(resp)).strip()

# -----------------------
# Vector search (Chroma) using local sentence-transformers for query embedding
# -----------------------
def vector_search(query: str, n_results: int = 5):
    q_emb = _embed_model.encode([query])[0].tolist()
    # query with embeddings
    results = chroma_col.query(query_embeddings=[q_emb], n_results=n_results, include=['documents','metadatas','distances','ids'])
    hits = []
    # results format: lists inside lists (one query)
    docs = results.get('documents', [[]])[0]
    metas = results.get('metadatas', [[]])[0]
    dists = results.get('distances', [[]])[0]
    ids = results.get('ids', [[]])[0]
    for doc, meta, dist, _id in zip(docs, metas, dists, ids):
        hits.append({"id": _id, "text": doc, "meta": meta, "score": float(dist)})
    return hits

# -----------------------
# Intent heuristic
# -----------------------
def decide_intent(question: str) -> str:
    sql_keywords = ['how many','count','average','avg','sum','total','between','where','group by','per','list','show me','which profiles','top']
    if any(k in question.lower() for k in sql_keywords):
        return "sql"
    return "vector"

# -----------------------
# FastAPI
# -----------------------
app = FastAPI(title="floatchat RAG API")

class AskRequest(BaseModel):
    question: str
    mode: str = "auto"  # auto, sql, vector, hybrid

@app.post("/ask")
def ask(req: AskRequest):
    q = req.question.strip()
    mode = (req.mode or "auto").lower()

    intent = mode if mode in ("sql","vector","hybrid") else decide_intent(q)

    sql_rows = None
    passages = None
    raw_sql = None

    if intent in ("sql", "hybrid"):
        prompt = build_sql_prompt(q, SCHEMA_CACHE, examples=[
            {"q":"How many profiles were collected in 2021 in the Bay of Bengal?","sql":"SELECT COUNT(*) FROM profiles WHERE region = 'Bay of Bengal' AND extract(year FROM date) = 2021;"},
            {"q":"List top 5 profiles by maximum temperature.","sql":"SELECT profile_id, MAX(temperature) AS max_temp FROM measurements GROUP BY profile_id ORDER BY max_temp DESC LIMIT 5;"}
        ])
        raw_out = call_llm_for_sql(prompt)
        raw_sql = clean_sql_output(raw_out)
        try:
            safe_sql = validate_sql(raw_sql)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"SQL validation error: {e}")
        # execute SQL (read-only user recommended)
        try:
            conn = get_db_conn()
            cur = conn.cursor()
            cur.execute(safe_sql)
            cols = [desc[0] for desc in cur.description] if cur.description else []
            rows = cur.fetchall()
            sql_rows = [dict(zip(cols, r)) for r in rows]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"DB execution error: {e}")
        finally:
            try:
                cur.close()
                conn.close()
            except:
                pass

    if intent in ("vector", "hybrid"):
        passages = vector_search(q, n_results=5)

    # Compose final answer
    answer = rag_compose_answer(q, sql_rows=sql_rows, passages=passages)

    return {
        "answer": answer,
        "intent": intent,
        "sql_raw": raw_sql,
        "rows": sql_rows,
        "passages": passages
    }