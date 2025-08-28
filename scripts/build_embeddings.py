import os
import argparse
import pandas as pd
import glob
import tqdm
import chromadb
from sentence_transformers import SentenceTransformer
import json
import math
import torch # Import torch to check for CUDA availability

def load_file(path, sample_size=None):
    """Load parquet or CSV file and optionally sample rows."""
    try:
        if path.endswith(".parquet"):
            df = pd.read_parquet(path)
        elif path.endswith(".csv"):
            df = pd.read_csv(path)
        else:
            print(f"⚠️ Skipping unsupported file type: {path}")
            return pd.DataFrame()

        if sample_size and sample_size > 0:
            if len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
        return df
    except Exception as e:
        print(f"❌ Error loading file {path}: {e}")
        return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser(description="Build and store embeddings in ChromaDB.")
    parser.add_argument("--data-path", required=True, help="Path to the directory containing data files (parquet or csv).")
    parser.add_argument("--collection", required=True, help="Name of the ChromaDB collection.")
    parser.add_argument("--host", default="localhost", help="ChromaDB host.")
    parser.add_argument("--port", type=int, default=8000, help="ChromaDB port.")
    parser.add_argument("--embed-model", default="all-MiniLM-L6-v2", help="Name of the sentence-transformer model to use.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for embedding generation.")
    parser.add_argument("--sample", type=int, default=0, help="Sample size for testing. 0 means use all data.")
    parser.add_argument("--resume-file", default="resume.json", help="File to store and resume progress.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    args = parser.parse_args()

    # --- FIX: Check for CUDA availability and set device automatically ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print("✅ CUDA is available. Using GPU for embeddings.")
    else:
        print("⚠️ CUDA not available. Using CPU for embeddings. This might be slow.")
        print("   If you have an NVIDIA GPU, please install a CUDA-enabled version of PyTorch.")

    # Connect to Chroma
    try:
        client = chromadb.HttpClient(host=args.host, port=args.port)
        # This is a heartbeat check
        client.heartbeat()
        collection = client.get_or_create_collection(name=args.collection)
        print(f"✅ Connected to Chroma at {args.host}:{args.port}, collection = {args.collection}")
    except Exception as e:
        print(f"❌ Could not connect to ChromaDB at {args.host}:{args.port}. Please ensure it's running.")
        print(f"   Error: {e}")
        return # Exit if we can't connect to the database

    # Load embedding model
    print(f"⚡ Loading embedding model: {args.embed_model} on {device.upper()}")
    model = SentenceTransformer(args.embed_model, device=device)

    # Load resume progress
    done_ids = set()
    if os.path.exists(args.resume_file):
        try:
            with open(args.resume_file, "r") as f:
                done_ids = set(json.load(f))
            print(f"🔄 Resuming, {len(done_ids)} docs already processed")
        except json.JSONDecodeError:
            print(f"⚠️ Could not read resume file at {args.resume_file}. Starting from scratch.")
            done_ids = set()


    # Collect all parquet/csv files
    files = glob.glob(os.path.join(args.data_path, "*.parquet")) + glob.glob(os.path.join(args.data_path, "*.csv"))
    if not files:
        print(f"❌ No .parquet or .csv files found in {args.data_path}")
        return

    total_new_docs = 0

    for f in files:
        if args.verbose:
            print(f"📂 Processing file: {f}")
        df = load_file(f, sample_size=args.sample if args.sample > 0 else None)
        if df.empty:
            continue

        # Create a unique ID for each document based on its index and filename
        df["doc_id"] = df.index.astype(str) + "_" + os.path.basename(f)

        # Filter out documents that have already been processed
        df = df[~df["doc_id"].isin(done_ids)]
        if df.empty:
            if args.verbose:
                print(f"✅ All docs from {f} already embedded, skipping.")
            continue

        docs_to_process_count = len(df)
        total_new_docs += docs_to_process_count
        print(f"⚡ Found {docs_to_process_count} new docs to embed from {f}")

        # Combine all columns into a single text document for embedding
        # This assumes all columns should be part of the document content
        documents_to_embed = df.astype(str).agg(" ".join, axis=1).tolist()
        ids_to_embed = df["doc_id"].tolist()

        # Batch embedding with a progress bar
        with tqdm.tqdm(total=docs_to_process_count, desc=f"Embedding {os.path.basename(f)}", unit="doc") as pbar:
            for i in range(0, docs_to_process_count, args.batch_size):
                batch_docs = documents_to_embed[i:i+args.batch_size]
                batch_ids = ids_to_embed[i:i+args.batch_size]

                # Create embeddings
                embeddings = model.encode(batch_docs, show_progress_bar=False, convert_to_tensor=False).tolist()

                # Add to ChromaDB
                try:
                    collection.add(ids=batch_ids, documents=batch_docs, embeddings=embeddings)
                except Exception as e:
                    print(f"\n❌ Error adding batch to ChromaDB: {e}")
                    continue # Continue to the next batch

                # Update resume file after each successful batch
                done_ids.update(batch_ids)
                with open(args.resume_file, "w") as f_resume:
                    json.dump(list(done_ids), f_resume)

                pbar.update(len(batch_ids))

    print(f"\n🎉 Finished embedding {total_new_docs} new docs.")
    print(f"📈 Total docs in collection '{args.collection}': {collection.count()}")

if __name__ == "__main__":
    main()


"""  
python scripts/build_embeddings.py --data-path data/interim --collection floatchat --batch-size 5000 --verbose   

"""