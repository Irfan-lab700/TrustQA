import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

TOP_KS = [1, 5, 10]
K = 60

# Corpus

with open(
    "data/processed/corpus_final.jsonl",
    "r",
    encoding="utf-8"
) as f:

    corpus = [
        json.loads(line)
        for line in f
    ]

doc_map = {
    d["doc_id"]: d["text"].lower()
    for d in corpus
}

doc_ids = [
    d["doc_id"]
    for d in corpus
]

# BM25

with open(
    "data/indexes/bm25.pkl",
    "rb"
) as f:

    bm25_data = pickle.load(f)

bm25 = bm25_data["bm25"]

# FAISS


index = faiss.read_index(
    "data/indexes/faiss.index"
)

with open(
    "data/indexes/faiss_docs.pkl",
    "rb"
) as f:

    faiss_docs = pickle.load(f)


# Embedding model


model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# Answer extraction


def extract_answer(answer):

    if answer["answer_spans"]:
        return [
            span["text"].lower().strip()
            for span in answer["answer_spans"]
        ]

    elif answer["answer_value"]:
        return [
            str(answer["answer_value"]).lower().strip()
        ]

    return []

# Evaluation


hits = {
    1: 0,
    5: 0,
    10: 0
}

total = 0

with open(
    "data/raw/iirc_500.json",
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        article = json.loads(line)

        for q in article["questions"]:

            answers = extract_answer(
                q["answer"]
            )

            if not answers:
                continue

            question = q["question"]

            # BM25

            bm25_scores = bm25.get_scores(
                question.lower().split()
            )

            bm25_ranked = np.argsort(
                bm25_scores
            )[::-1][:20]

            # FAISS

            emb = model.encode(
                [question],
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            _, faiss_ranked = index.search(
                emb,
                20
            )

            faiss_ranked = faiss_ranked[0]

            # RRF

            rrf_scores = {}

            for rank, idx in enumerate(
                bm25_ranked
            ):

                docid = doc_ids[idx]

                rrf_scores[docid] = (
                    rrf_scores.get(docid, 0)
                    +
                    1 / (K + rank + 1)
                )

            for rank, idx in enumerate(
                faiss_ranked
            ):

                item = faiss_docs[idx]

                if isinstance(item, dict):
                    docid = item["doc_id"]
                else:
                    docid = item

                rrf_scores[docid] = (
                    rrf_scores.get(docid, 0)
                    +
                    1 / (K + rank + 1)
                )

            ranked_docs = sorted(
                rrf_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )

            total += 1

            for k in TOP_KS:

                top_docs = ranked_docs[:k]

                retrieved = " ".join(
                    doc_map[docid]
                    for docid, _
                    in top_docs
                    if docid in doc_map
                )

                if any(
                    ans in retrieved
                    for ans in answers
                ):
                    hits[k] += 1

#Results
print("\nRRF IIRC RESULTS\n")

for k in TOP_KS:

    print(
        f"Recall@{k}: {hits[k]/total:.4f}"
    )

print(f"\nQuestions: {total}")