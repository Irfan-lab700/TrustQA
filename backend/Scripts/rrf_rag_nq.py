import json
import os
import pickle
import time

import faiss
import numpy as np

from llama_cpp import Llama
from sentence_transformers import SentenceTransformer


# -------------------------
# Config
# -------------------------

TOP_K = 3

BM25_TOP = 20
FAISS_TOP = 20

RRF_K = 60

MODEL_PATH = "models/qwen/qwen2.5-3b-instruct-q4_k_m.gguf"

FAISS_INDEX = "data/indexes/faiss.index"
DOCS_FILE = "data/indexes/faiss_docs.pkl"
BM25_FILE = "data/indexes/bm25.pkl"

OUTPUT_FILE = "results/rrf_rag_nq.json"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# -------------------------
# Load Qwen
# -------------------------

print("Loading Qwen...")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=6,
    chat_format="chatml",
    verbose=False,
)

print("Qwen loaded")


# -------------------------
# Load FAISS
# -------------------------

print("Loading FAISS...")

index = faiss.read_index(
    FAISS_INDEX
)

print(
    "FAISS vectors:",
    index.ntotal
)


# -------------------------
# Load Documents
# -------------------------

with open(
    DOCS_FILE,
    "rb"
) as f:

    documents = pickle.load(f)


print(
    "Documents:",
    len(documents)
)


# -------------------------
# Load BM25
# -------------------------

print("Loading BM25...")

with open(
    BM25_FILE,
    "rb"
) as f:

    bm25_data = pickle.load(f)


bm25 = bm25_data["bm25"]

print("BM25 loaded")


# -------------------------
# Load Embedding Model
# -------------------------

print("Loading embeddings...")

embedder = SentenceTransformer(
    EMBED_MODEL
)

print("Embedding loaded")


# -------------------------
# Load NQ
# -------------------------

print("Loading NQ...")

nq = []

with open(
    "data/raw/nq_500.json",
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        nq.append(
            json.loads(line)
        )


print(
    "Questions:",
    len(nq)
)


# -------------------------
# Resume
# -------------------------

results = []

if os.path.exists(
    OUTPUT_FILE
):

    with open(
        OUTPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        results = json.load(f)


start = len(results)

print(
    "Already completed:",
    start
)


# -------------------------
# RRF Loop
# -------------------------

for idx in range(
    start,
    len(nq)
):

    sample = nq[idx]

    question = sample["question"]

    total_start = time.time()


    # -------------------------
    # BM25 Retrieval
    # -------------------------

    retrieval_start = time.time()

    query_tokens = (
        question
        .lower()
        .split()
    )

    bm25_scores = bm25.get_scores(
        query_tokens
    )


    bm25_ranked = np.argsort(
        bm25_scores
    )[::-1][:BM25_TOP]


    # -------------------------
    # FAISS Retrieval
    # -------------------------

    query_embedding = embedder.encode(
        [question],
        normalize_embeddings=True
    )


    query_embedding = np.array(
        query_embedding
    ).astype("float32")


    faiss_scores, faiss_ranked = index.search(
        query_embedding,
        FAISS_TOP
    )


    faiss_ranked = faiss_ranked[0]


    # -------------------------
    # RRF Fusion
    # -------------------------

    rrf_scores = {}


    for rank, doc_idx in enumerate(
        bm25_ranked
    ):

        rrf_scores[doc_idx] = (
            rrf_scores.get(
                doc_idx,
                0
            )
            +
            1 /
            (RRF_K + rank + 1)
        )


    for rank, doc_idx in enumerate(
        faiss_ranked
    ):

        rrf_scores[doc_idx] = (
            rrf_scores.get(
                doc_idx,
                0
            )
            +
            1 /
            (RRF_K + rank + 1)
        )


    fused_docs = sorted(
        rrf_scores.items(),
        key=lambda x:x[1],
        reverse=True
    )[:TOP_K]


    retrieval_time = (
        time.time()
        -
        retrieval_start
    )


    contexts = []

    retrieved_docs = []


    for doc_idx, score in fused_docs:

        doc = documents[doc_idx]


        contexts.append(
            doc["text"]
        )


        retrieved_docs.append(
            {
                "doc_id": doc["doc_id"],
                "rrf_score": float(score),
                "text": doc["text"]
            }
        )


    context = "\n\n".join(
        contexts
    )


    # -------------------------
    # Generation
    # -------------------------

    prompt = f"""

You must answer ONLY from the provided context.

Rules:
- Copy the answer exactly from the context whenever possible.
- Do not explain.
- Do not write a full sentence.
- Return only the answer phrase.
- If the answer is not present in the context, return: NOT_FOUND

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
                    "role":"user",
                    "content":prompt
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

        prediction = (
            f"ERROR: {e}"
        )


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

            "gold_answers":
                sample["answers"],

            "prediction":
                prediction,

            "retrieved_docs":
                retrieved_docs,

            "retrieval_time_sec":
                round(
                    retrieval_time,
                    4
                ),

            "generation_time_sec":
                round(
                    generation_time,
                    4
                ),

            "total_time_sec":
                round(
                    total_time,
                    4
                )
        }
    )


    if (idx+1)%10 == 0:

        print(
            f"Completed {idx+1}/{len(nq)}"
        )


    if (idx+1)%50 == 0:

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
            idx+1
        )


# -------------------------
# Final Save
# -------------------------

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


print("\nDONE")
print(
    "Saved:",
    OUTPUT_FILE
)