import json
import pickle
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

TOP_KS = [1, 5, 10]
RRF_K = 60

# ---------------------
# BM25
# ---------------------

with open(
    "data/indexes/bm25.pkl",
    "rb"
) as f:
    bm25_data = pickle.load(f)

bm25 = bm25_data["bm25"]
documents = bm25_data["documents"]

# ---------------------
# FAISS
# ---------------------

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

index = faiss.read_index(
    "data/indexes/faiss.index"
)

# ---------------------

def tokenize(text):
    return text.lower().split()

hits = {1: 0, 5: 0, 10: 0}
total = 0

with open(
    "data/raw/nq_500.json",
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        sample = json.loads(line)

        question = sample["question"]

        answers = [
            a.lower().strip()
            for a in sample["answers"]["text"]
        ]

        # BM25
        scores = bm25.get_scores(
            tokenize(question)
        )

        bm25_ranked = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:20]

        # FAISS
        emb = model.encode(
            [question],
            convert_to_numpy=True
        )

        _, faiss_indices = index.search(
            emb,
            20
        )

        faiss_ranked = list(
            faiss_indices[0]
        )

        # RRF
        rrf_scores = {}

        for rank, idx in enumerate(
            bm25_ranked
        ):
            rrf_scores[idx] = (
                rrf_scores.get(idx, 0)
                + 1 / (RRF_K + rank + 1)
            )

        for rank, idx in enumerate(
            faiss_ranked
        ):
            rrf_scores[idx] = (
                rrf_scores.get(idx, 0)
                + 1 / (RRF_K + rank + 1)
            )

        final_ranked = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        total += 1

        for k in TOP_KS:

            top_docs = [
                idx
                for idx, score
                in final_ranked[:k]
            ]

            retrieved_text = " ".join(
                documents[idx]["text"].lower()
                for idx in top_docs
            )

            found = any(
                ans in retrieved_text
                for ans in answers
            )

            if found:
                hits[k] += 1

print("\nRRF RESULTS\n")

for k in TOP_KS:

    recall = hits[k] / total

    print(
        f"Recall@{k}: {recall:.4f}"
    )

print(f"\nQuestions: {total}")