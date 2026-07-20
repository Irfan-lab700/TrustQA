import pickle
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import re

QUERY =  "How long was the reign of the person who annexed  Aurangabad, Maharashtra in 1308?"

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

# -----------------------------
# Load BM25
# -----------------------------

with open("data/indexes/bm25.pkl", "rb") as f:
    bm25_data = pickle.load(f)

bm25 = bm25_data["bm25"]
documents = bm25_data["documents"]

# -----------------------------
# Load FAISS
# -----------------------------

index = faiss.read_index(
    "data/indexes/faiss.index"
)

with open(
    "data/indexes/faiss_docs.pkl",
    "rb"
) as f:
    faiss_docs = pickle.load(f)

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# BM25 Top 20
# -----------------------------

bm25_scores = bm25.get_scores(
    tokenize(QUERY)
)

bm25_ranked = sorted(
    range(len(bm25_scores)),
    key=lambda i: bm25_scores[i],
    reverse=True
)[:20]

# -----------------------------
# FAISS Top 20
# -----------------------------

query_embedding = model.encode(
    [QUERY],
    convert_to_numpy=True
)

_, faiss_indices = index.search(
    query_embedding,
    20
)

faiss_ranked = list(faiss_indices[0])

# -----------------------------
# RRF
# -----------------------------

rrf_scores = {}

K = 60

for rank, idx in enumerate(bm25_ranked):
    rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (K + rank + 1)

for rank, idx in enumerate(faiss_ranked):
    rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (K + rank + 1)

final_ranked = sorted(
    rrf_scores.items(),
    key=lambda x: x[1],
    reverse=True
)[:5]

# -----------------------------
# Output
# -----------------------------

print("\nQUESTION:")
print(QUERY)

print("\nRRF TOP RESULTS:\n")

for rank, (idx, score) in enumerate(final_ranked, start=1):

    print("=" * 80)
    print(f"Rank {rank}")
    print(f"RRF Score: {score:.4f}")
    print(f"Doc ID: {documents[idx]['doc_id']}")
    print(documents[idx]["text"][:500])
    print()