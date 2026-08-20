import json
import os
import pickle
import time
import faiss
import numpy as np

from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

# Config
TOP_K = 3
RRF_CANDIDATES = 20
RRF_K = 60

MODEL_PATH = "models/qwen/qwen2.5-3b-instruct-q4_k_m.gguf"
OUTPUT_FILE = "results/rrf_rag_iirc.json"
FAISS_INDEX = "data/indexes/faiss.index"
DOCS_FILE = "data/indexes/faiss_docs.pkl"
BM25_FILE = "data/indexes/bm25.pkl"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Load Qwen

print("Loading Qwen...")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=6,
    chat_format="chatml",
    verbose=False,
)

print("Qwen loaded")

# Load BM25

print("Loading BM25...")

with open(BM25_FILE, "rb") as f:
    bm25_data = pickle.load(f)

bm25 = bm25_data["bm25"]
bm25_documents = bm25_data["documents"]

print("BM25 loaded")

# Load Faiss

print("Loading FAISS...")

index = faiss.read_index(
    FAISS_INDEX
)

with open(DOCS_FILE, "rb") as f:
    faiss_documents = pickle.load(f)

print("FAISS loaded")
print("Documents:", len(faiss_documents))

# Doc lookup

doc_lookup = {}

for doc in faiss_documents:
    doc_lookup[doc["doc_id"]] = doc

# Embedding model

print("Loading embedding model...")

embedder = SentenceTransformer(
    EMBED_MODEL
)

print("Embedding model loaded")

# Load IIRC

print("Loading IIRC...")

qa_data = []

with open(
    "data/raw/iirc_500.json",
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        article = json.loads(line)

        for q in article["questions"]:

            answer_type = q["answer"]["type"]

            if answer_type not in [
                "span",
                "value"
            ]:
                continue

            if answer_type == "span":

                answers = [
                    span["text"]
                    for span in q["answer"]["answer_spans"]
                ]

            else:

                answers = [
                    str(q["answer"]["answer_value"])
                ]

            qa_data.append(
                {
                    "question": q["question"],
                    "answers": answers
                }
            )

print("Questions:", len(qa_data))

# Resume support 

results = []

if os.path.exists(OUTPUT_FILE):

    with open(
        OUTPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        results = json.load(f)

start = len(results)

print("Already completed:", start)

experiment_start = time.time()

# Main loop

for idx in range(start, len(qa_data)):

    sample = qa_data[idx]

    question = sample["question"]

    total_start = time.time()

    retrieval_start = time.time()

    # BM25 Top-20

    bm25_scores = bm25.get_scores(
        question.lower().split()
    )

    bm25_ranked = np.argsort(
        bm25_scores
    )[::-1][:RRF_CANDIDATES]
    
    # Faiss Top-20

    query_embedding = embedder.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    _, faiss_ranked = index.search(
        query_embedding,
        RRF_CANDIDATES
    )

    faiss_ranked = faiss_ranked[0]

    # RRF fusion

    rrf_scores = {}

    for rank, doc_idx in enumerate(bm25_ranked):

        doc_id = bm25_documents[
            doc_idx
        ]["doc_id"]

        rrf_scores[doc_id] = (
            rrf_scores.get(doc_id, 0)
            +
            1 / (RRF_K + rank + 1)
        )

    for rank, doc_idx in enumerate(faiss_ranked):

        doc_id = faiss_documents[
            doc_idx
        ]["doc_id"]

        rrf_scores[doc_id] = (
            rrf_scores.get(doc_id, 0)
            +
            1 / (RRF_K + rank + 1)
        )

    final_docs = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:TOP_K]

    contexts = []
    retrieved_docs = []

    for doc_id, score in final_docs:

        doc = doc_lookup[doc_id]

        contexts.append(
            doc["text"]
        )

        retrieved_docs.append(
            {
                "doc_id": doc_id,
                "rrf_score": float(score),
                "text": doc["text"]
            }
        )

    retrieval_time = (
        time.time()
        -
        retrieval_start
    )

    context = "\n\n".join(contexts)

    # Prompt

    prompt = f"""
You must answer ONLY from the provided context.

Rules:
- Copy the answer exactly from context.
- Do not explain.
- Do not write a sentence.
- Return only the answer phrase.
- If answer is absent return NOT_FOUND.

Context:
{context[:1500]}

Question:
{question}

Answer:
"""

    generation_start = time.time()

    try:

        response = llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            max_tokens=30
        )

        prediction = (
            response["choices"][0]
            ["message"]["content"]
            .strip()
        )

    except Exception as e:

        prediction = f"ERROR: {e}"

    generation_time = (
        time.time()
        -
        generation_start
    )

    total_time = (
        time.time()
        -
        total_start
    )

    results.append(
        {
            "question": question,
            "gold_answers": sample["answers"],
            "prediction": prediction,
            "retrieved_docs": retrieved_docs,
            "retrieval_time_sec": round(
                retrieval_time,
                4
            ),
            "generation_time_sec": round(
                generation_time,
                4
            ),
            "total_time_sec": round(
                total_time,
                4
            )
        }
    )

    if (idx + 1) % 10 == 0:

        print(
            f"Completed {idx+1}/{len(qa_data)}"
        )

    if (idx + 1) % 50 == 0:

        os.makedirs(
            "results",
            exist_ok=True
        )

        with open(
            OUTPUT_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                results,
                f,
                indent=2,
                ensure_ascii=False
            )

        print(
            "Checkpoint saved:",
            idx + 1
        )

# Save Final

os.makedirs(
    "results",
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False
    )

experiment_time = (
    time.time()
    -
    experiment_start
)

print("\nDONE")
print("Saved:", OUTPUT_FILE)

print(
    "Total experiment time (minutes):",
    round(experiment_time / 60, 2)
)