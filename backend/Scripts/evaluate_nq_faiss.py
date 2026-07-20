import json
import pickle
import faiss
from sentence_transformers import SentenceTransformer

TOP_KS = [1, 5, 10]

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

        query_emb = model.encode(
            [question],
            convert_to_numpy=True
        )

        _, indices = index.search(
            query_emb,
            10
        )

        retrieved_ids = indices[0]

        total += 1

        for k in TOP_KS:

            top_docs = retrieved_ids[:k]

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

print("\nFAISS RESULTS\n")

for k in TOP_KS:

    recall = hits[k] / total

    print(
        f"Recall@{k}: {recall:.4f}"
    )

print(f"\nQuestions: {total}")