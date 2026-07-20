import json
import pickle
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

print("Loading embedding model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

documents = []
texts = []

print("Loading corpus...")

with open("data/processed/corpus_final.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)

        documents.append(doc)
        texts.append(doc["text"])

print(f"Loaded {len(texts)} docs")

print("Generating embeddings...")
embeddings = model.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True
)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("Saving FAISS index...")

faiss.write_index(
    index,
    "data/indexes/faiss.index"
)

with open(
    "data/indexes/faiss_docs.pkl",
    "wb"
) as f:
    pickle.dump(documents, f)

print("Done!")
print("Vectors:", index.ntotal)