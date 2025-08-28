import os
import argparse
import pandas as pd
import glob
import tqdm
import chromadb
from sentence_transformers import SentenceTransformer
import json
import gc
import torch
import pyarrow.parquet as pq

def stream_data_chunks(path, chunk_size=100_000):
    """
    Generator function to yield data chunks from Parquet or CSV files.
    This avoids loading the entire file into memory.

    Args:
        path (str): Path to the data file.
        chunk_size (int): The number of rows for CSV chunking. Parquet is chunked by row group.

    Yields:
        pd.DataFrame: A chunk of the data.
    """
    try:
        if path.endswith(".parquet"):
            parquet_file = pq.ParquetFile(path)
            # Iterate over row groups, which is the natural way to chunk Parquet files.
            for row_group in range(parquet_file.num_row_groups):
                yield parquet_file.read_row_group(row_group).to_pandas()
        elif path.endswith(".csv"):
            # For CSVs, we can use the chunksize parameter directly.
            for chunk in pd.read_csv(path, chunksize=chunk_size, on_bad_lines='warn'):
                yield chunk
        else:
            print(f"⚠️ Skipping unsupported file type: {path}")
            return
    except Exception as e:
        print(f"❌ Error streaming data from file {path}: {e}")
        return

def main():
    """Main function to build and store embeddings in ChromaDB."""
    parser = argparse.ArgumentParser(
        description="Build and store embeddings in ChromaDB using a memory-efficient streaming approach."
    )
    parser.add_argument("--data-path", required=True, help="Path to the directory containing data files (parquet or csv).")
    parser.add_argument("--collection", required=True, help="Name of the ChromaDB collection.")
    parser.add_argument("--host", default="localhost", help="ChromaDB host.")
    parser.add_argument("--port", type=int, default=8000, help="ChromaDB port.")
    parser.add_argument("--embed-model", default="all-MiniLM-L6-v2", help="Name of the sentence-transformer model to use.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for embedding generation.")
    parser.add_argument("--file-chunk-size", type=int, default=100_000, help="Number of rows to read from a CSV file at a time.")
    parser.add_argument("--resume-file", default="resume_progress.json", help="File to store and resume progress.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    args = parser.parse_args()

    # --- Check for CUDA availability and set device ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print("✅ CUDA is available. Using GPU for embeddings.")
    else:
        print("⚠️ CUDA not available. Using CPU for embeddings. This might be slow.")

    # --- Connect to ChromaDB ---
    try:
        client = chromadb.HttpClient(host=args.host, port=args.port)
        client.heartbeat()
        collection = client.get_or_create_collection(name=args.collection)
        print(f"✅ Connected to Chroma at {args.host}:{args.port}, collection = {args.collection}")
    except Exception as e:
        print(f"❌ Could not connect to ChromaDB at {args.host}:{args.port}. Please ensure it's running.")
        print(f"   Error: {e}")
        return

    # --- Load Embedding Model ---
    print(f"⚡ Loading embedding model: {args.embed_model} on {device.upper()}")
    model = SentenceTransformer(args.embed_model, device=device)

    # --- Load Resume Progress ---
    done_ids = set()
    if os.path.exists(args.resume_file):
        try:
            with open(args.resume_file, "r") as f:
                done_ids = set(json.load(f))
            print(f"🔄 Resuming, {len(done_ids)} docs already processed.")
        except json.JSONDecodeError:
            print(f"⚠️ Could not read resume file at {args.resume_file}. Starting from scratch.")
            done_ids = set()

    # --- Collect all files ---
    files = glob.glob(os.path.join(args.data_path, "*.parquet")) + glob.glob(os.path.join(args.data_path, "*.csv"))
    if not files:
        print(f"❌ No .parquet or .csv files found in {args.data_path}")
        return

    total_new_docs_processed = 0

    # --- Process Files by Streaming Chunks ---
    for f in files:
        print(f"📂 Processing file: {f}")
        
        # Use the generator to stream chunks from the file
        for chunk_df in stream_data_chunks(f, chunk_size=args.file_chunk_size):
            # Create a unique ID for each document based on its original index and filename
            # Note: chunk_df.index will restart from 0 for each chunk, which is what we want.
            chunk_df["doc_id"] = chunk_df.index.astype(str) + "_" + os.path.basename(f)

            # Filter out documents that have already been processed
            initial_chunk_size = len(chunk_df)
            chunk_df = chunk_df[~chunk_df["doc_id"].isin(done_ids)]
            new_docs_in_chunk = len(chunk_df)

            if new_docs_in_chunk == 0:
                if args.verbose:
                    print(f"✅ No new docs in this chunk of {os.path.basename(f)}, skipping.")
                continue
            
            if new_docs_in_chunk < initial_chunk_size:
                 print(f"⚡ Found {new_docs_in_chunk} new docs to embed in this chunk of {os.path.basename(f)}.")

            # Combine all columns into a single text document for embedding
            documents_to_embed = chunk_df.astype(str).agg(" ".join, axis=1).tolist()
            ids_to_embed = chunk_df["doc_id"].tolist()

            # Batch embedding with a progress bar
            with tqdm.tqdm(total=new_docs_in_chunk, desc=f"Embedding {os.path.basename(f)} chunk", unit="doc") as pbar:
                for i in range(0, new_docs_in_chunk, args.batch_size):
                    batch_docs = documents_to_embed[i:i+args.batch_size]
                    batch_ids = ids_to_embed[i:i+args.batch_size]

                    # Create embeddings
                    embeddings = model.encode(batch_docs, show_progress_bar=False).tolist()

                    # Add to ChromaDB
                    try:
                        collection.add(ids=batch_ids, documents=batch_docs, embeddings=embeddings)
                        total_new_docs_processed += len(batch_ids)
                    except Exception as e:
                        print(f"\n❌ Error adding batch to ChromaDB: {e}")
                        continue  # Continue to the next batch

                    # Update resume file after each successful batch
                    done_ids.update(batch_ids)
                    with open(args.resume_file, "w") as f_resume:
                        json.dump(list(done_ids), f_resume)

                    pbar.update(len(batch_ids))

            # --- Explicit Memory Cleanup ---
            # Clean up the large objects from this chunk before loading the next one
            del chunk_df, documents_to_embed, ids_to_embed
            gc.collect()

    print(f"\n🎉 Finished embedding {total_new_docs_processed} new docs.")
    print(f"📈 Total docs in collection '{args.collection}': {collection.count()}")

if __name__ == "__main__":
    main()


"""  
python scripts/build_embeddings.py --data-path data/interim --collection floatchat --batch-size 5000 --verbose   

"""