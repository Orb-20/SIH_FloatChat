import chromadb

# 1. Start local Chroma client (no server needed)
client = chromadb.Client()

# 2. Create or get a collection
coll = client.get_or_create_collection(name="test_collection")

# 3. Add some test documents
coll.add(
    documents=[
        "Warm water at 50 dbar depth in the Pacific Ocean",
        "Cold water observed at 1000 dbar depth",
        "Surface temperature reading from the Atlantic Ocean"
    ],
    ids=["doc1", "doc2", "doc3"]
)

# 4. Query the collection with natural language
results = coll.query(
    query_texts=["warm water at 50 dbar"],
    n_results=2
)

# 5. Print results
print("Query Results:")
print(results)
