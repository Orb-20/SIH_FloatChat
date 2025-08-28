# scripts/build_embeddings_chroma.py
import os
import math
import time
from dotenv import load_dotenv
from io import StringIO

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer

import chromadb

load_dotenv()

# ---------- Config ----------
PG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
    "db": os.getenv("POSTGRES_DB", "argo"),
    "user": os.getenv("POSTGRES_USER", "argo"),
    "pw": os.getenv("POSTGRES_PASSWORD", "argo"),
}
PG_URL = f"postgresql+psycopg2://{PG['user']}:{PG['pw']}@{PG['host']}:{PG['port']}/{PG['db']}"

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "argo_profiles")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
BATCH_SIZE = int(os.getenv("EMBED_BATCH", "512"))

# ---------- Helpers ----------
def create_db_engine():
    return create_engine(PG_URL, future=True)

def create_chroma_client(host=CHROMA_HOST, port=CHROMA_PORT):
    """
    Try to create an HTTP client (if chroma server is running and HttpClient exists),
    otherwise fall back to in-process client chromadb.Client().
    """
    # Try HttpClient if available (some package versions expose this)
    try:
        if hasattr(chromadb, "HttpClient"):
            client = chromadb.HttpClient(host=host, port=port)
            print(f"Using chromadb.HttpClient -> {host}:{port}")
            return client
    except Exception:
        pass

    # Some versions used Settings(... chroma_api_impl="rest") but earlier error shows that isn't available.
    # Fall back to local client (in-process)
    try:
        client = chromadb.Client()
        print("Using chromadb.Client (in-process).")
        return client
    except Exception as e:
        raise RuntimeError(
            "Failed to create chromadb client (tried HttpClient and Client()). "
            "If you want to connect to a remote chroma server, make sure your chromadb package "
            "version supports HttpClient or the REST Settings API. Error: " + str(e)
        )

def get_or_create_collection(client, name):
    # try get_collection, else create_collection
    try:
        coll = client.get_collection(name)
        print("Found existing collection:", name)
        return coll
    except Exception:
        try:
            coll = client.create_collection(name=name)
            print("Created collection:", name)
            return coll
        except Exception as e:
            # Some versions offer get_or_create_collection
            if hasattr(client, "get_or_create_collection"):
                coll = client.get_or_create_collection(name=name)
                print("Get-or-created collection:", name)
                return coll
            raise RuntimeError("Could not get/create collection: " + str(e))

def sanitize_value(v):
    """Convert numpy / pandas types to plain python types and Timestamps to ISO strings."""
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        if np.isnan(v):
            return None
        return float(v)
    if isinstance(v, (np.bool_ , bool)):
        return bool(v)
    if isinstance(v, (pd.Timestamp,)):
        if pd.isna(v):
            return None
        # ISO string
        return v.isoformat()
    if isinstance(v, (bytes, bytearray)):
        try:
            return v.decode("utf-8", "ignore")
        except Exception:
            return str(v)
    # pandas NA
    if v is pd.NA:
        return None
    # fallback
    try:
        return int(v)
    except Exception:
        try:
            return float(v)
        except Exception:
            try:
                return str(v)
            except Exception:
                return None

def row_to_metadata(row_dict):
    """Make metadata dict JSON-serializable"""
    out = {}
    for k, v in row_dict.items():
        out[k] = sanitize_value(v)
    return out

# ---------- Core ----------
def fetch_profiles(limit=None):
    engine = create_db_engine()
    sql = "SELECT profile_id, platform_number, cycle_number, juld, latitude, longitude, data_mode, source_file FROM profiles ORDER BY profile_id"
    if limit:
        sql += f" LIMIT {int(limit)}"
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)
    return df

def build_text_for_row(row):
    # concise and consistent text
    juld = row.get("juld")
    juld_str = pd.to_datetime(juld).isoformat() if pd.notnull(juld) else "unknown"
    lat = sanitize_value(row.get("latitude"))
    lon = sanitize_value(row.get("longitude"))
    plat = sanitize_value(row.get("platform_number"))
    cyc = sanitize_value(row.get("cycle_number"))
    mode = sanitize_value(row.get("data_mode"))
    src = sanitize_value(row.get("source_file"))
    pid = sanitize_value(row.get("profile_id"))
    return f"profile {pid} | platform {plat} | cycle {cyc} | date {juld_str} | loc {lat},{lon} | mode {mode} | file {src}"

def compute_and_upsert_embeddings(limit=None):
    df = fetch_profiles(limit=limit)
    if df.empty:
        print("No profiles found.")
        return

    # ensure profile_id is string IDs
    ids = [str(int(x)) for x in df["profile_id"].tolist()]

    texts = [build_text_for_row(r) for _, r in df.iterrows()]
    metadatas = [row_to_metadata(r.to_dict()) for _, r in df.iterrows()]

    print("Loaded", len(texts), "profiles for embeddings.")

    # load model
    model = SentenceTransformer(EMBED_MODEL_NAME)
    print("Model loaded:", EMBED_MODEL_NAME)

    # Chromadb client + collection
    client = create_chroma_client()
    coll = get_or_create_collection(client, COLLECTION_NAME)

    # choose API: upsert preferred, fallback to add
    use_upsert = hasattr(coll, "upsert")
    use_add = hasattr(coll, "add")
    if not use_upsert and not use_add:
        raise RuntimeError("Chroma collection does not expose upsert or add methods. Unsupported chromadb version.")

    n = len(texts)
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch_texts = texts[start:end]
        batch_ids = ids[start:end]
        batch_meta = metadatas[start:end]

        print(f"Encoding batch {start}..{end-1} (size {len(batch_texts)})...")
        embeddings = model.encode(batch_texts, show_progress_bar=False, convert_to_numpy=True)

        # convert to python lists
        emb_list = [emb.tolist() for emb in embeddings]

        print(f"Upserting batch {start}..{end-1} into Chroma...")
        if use_upsert:
            # many chroma servers expect lists for embeddings, metadatas, documents
            try:
                coll.upsert(ids=batch_ids, embeddings=emb_list, metadatas=batch_meta, documents=batch_texts)
            except Exception as e:
                # fallback to add if upsert unsupported on this collection version
                if use_add:
                    coll.add(ids=batch_ids, embeddings=emb_list, metadatas=batch_meta, documents=batch_texts)
                else:
                    raise
        else:
            coll.add(ids=batch_ids, embeddings=emb_list, metadatas=batch_meta, documents=batch_texts)

        print(f"Batch {start}..{end-1} done.")
        time.sleep(0.05)  # tiny sleep to avoid hammering server

    # final size info (some clients / versions have count())
    try:
        size = coll.count()
        print("Done. Collection size:", size)
    except Exception:
        print("Done. (collection size not available on this chroma client version)")

def main():
    # If you want to only create embeddings for a subset while testing, set a small limit here
    compute_and_upsert_embeddings(limit=None)

if __name__ == "__main__":
    main()
