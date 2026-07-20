import json
import pickle
import faiss

from sentence_transformers import SentenceTransformer

query = "How long was the reign of the person who annexed  Aurangabad, Maharashtra in 1308?"

print("Loading model...")
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Loading index...")
index = faiss.read_index(
    "data/indexes/faiss.index"
)

with open(
    "data/indexes/faiss_docs.pkl",
    "rb"
) as f:
    documents = pickle.load(f)

query_embedding = model.encode(
    [query],
    convert_to_numpy=True
)

distances, indices = index.search(
    query_embedding,
    5
)

print("\nQUESTION:")
print(query)

print("\nTOP 5 RESULTS:\n")

for rank, idx in enumerate(indices[0], start=1):

    print("=" * 80)
    print(f"Rank {rank}")
    print(f"Doc ID: {documents[idx]['doc_id']}")
    print(documents[idx]["text"][:500])
    print()