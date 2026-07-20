import json
import pickle
from rank_bm25 import BM25Okapi

documents = []
tokenized_docs = []

with open("data/processed/corpus_final.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)

        text = doc["text"]

        documents.append(doc)
        tokenized_docs.append(text.lower().split())

print(f"Loaded {len(documents)} documents")

bm25 = BM25Okapi(tokenized_docs)

with open("data/indexes/bm25.pkl", "wb") as f:
    pickle.dump(
        {
            "bm25": bm25,
            "documents": documents
        },
        f
    )

print("BM25 index saved!")