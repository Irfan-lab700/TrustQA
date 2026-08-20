
import json
import os
import pickle
import time
import math

import faiss
import numpy as np

from llama_cpp import Llama
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "models/qwen/qwen2.5-3b-instruct-q4_k_m.gguf"
NQ_FILE = "data/raw/nq_500.json"

FAISS_INDEX = "data/indexes/faiss.index"
DOCS_FILE = "data/indexes/faiss_docs.pkl"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

OUTPUT_FILE = "results/trustqa_test1.json"

TEST_QUESTIONS = 500

TOP_K = 3
CONTEXT_LIMIT = 1500

ENTROPY_THRESHOLD = 0.55

TIER1_MAX_TOKENS = 12
TIER1_LOGPROBS = 5

TIER2_MAX_TOKENS = 30

TEMPERATURE = 0

CHECKPOINT_EVERY = 10


# ============================================================
# LOAD QWEN
# ============================================================

print("Loading Qwen model...")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=6,
    logits_all=True,
    chat_format="chatml",
    verbose=False
)

print("Qwen model loaded")


# ============================================================
# LOAD FAISS
# ============================================================

print("\nLoading FAISS...")

index = faiss.read_index(FAISS_INDEX)

print("FAISS vectors:", index.ntotal)

with open(DOCS_FILE, "rb") as f:
    documents = pickle.load(f)

print("Documents:", len(documents))


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

embedder = SentenceTransformer(EMBED_MODEL)

print("Embedding model loaded")


# ============================================================
# LOAD NQ
# ============================================================

print("\nLoading NQ...")

questions = []

with open(NQ_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            questions.append(json.loads(line))

print("Loaded questions:", len(questions))


TEST_QUESTIONS = min(TEST_QUESTIONS, len(questions))


# ============================================================
# ENTROPY
# ============================================================

def calculate_entropy(top_logprobs):
    """
    Calculates the same approximate entropy used in
    the Tier-1 experiments.

    For every generated token:
        1. Convert returned log-probabilities to probabilities.
        2. Normalize the returned top-k probabilities.
        3. Calculate entropy.
    Finally, average entropy across generated tokens.
    """

    if not top_logprobs:
        return None

    total_entropy = 0.0
    token_count = 0

    for token_info in top_logprobs:

        if not token_info:
            continue

        probabilities = []

        for logprob in token_info.values():

            try:
                probability = math.exp(logprob)
            except Exception:
                continue

            probabilities.append(probability)

        if not probabilities:
            continue

        total_probability = sum(probabilities)

        if total_probability <= 0:
            continue

        probabilities = [
            p / total_probability
            for p in probabilities
        ]

        entropy = 0.0

        for probability in probabilities:

            if probability > 0:
                entropy -= (
                    probability
                    * math.log(probability)
                )

        total_entropy += entropy
        token_count += 1

    if token_count == 0:
        return None

    return total_entropy / token_count


# ============================================================
# TIER 1
# ============================================================

def run_tier1(question):

    # Keep the exact same prompt used in the
    # previous Tier-1 baseline experiments.
    prompt = f"""
<|im_start|>user
Answer only in 1-5 words:
{question}
<|im_end|>
<|im_start|>assistant
"""

    start = time.time()

    output = llm(
        prompt,
        max_tokens=TIER1_MAX_TOKENS,
        temperature=TEMPERATURE,
        logprobs=TIER1_LOGPROBS
    )

    latency = time.time() - start

    prediction = (
        output["choices"][0]["text"]
        .strip()
    )

    entropy = None

    log_data = output["choices"][0].get("logprobs")

    if log_data is not None:

        top_logprobs = log_data.get("top_logprobs")

        if top_logprobs:
            entropy = calculate_entropy(top_logprobs)

    return prediction, entropy, latency


# ============================================================
# FAISS RETRIEVAL
# ============================================================

def retrieve_faiss(question):

    start = time.time()

    query_embedding = embedder.encode(
        [question],
        normalize_embeddings=True
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
        TOP_K
    )

    retrieval_time = time.time() - start

    contexts = []
    retrieved_docs = []

    for score, doc_idx in zip(
        scores[0],
        indices[0]
    ):

        if doc_idx < 0:
            continue

        doc = documents[doc_idx]

        contexts.append(doc["text"])

        retrieved_docs.append(
            {
                "doc_id": doc["doc_id"],
                "score": float(score),
                "text": doc["text"]
            }
        )

    context = "\n\n".join(contexts)

    return (
        
        context[:CONTEXT_LIMIT],
        retrieved_docs,
        retrieval_time
    )


# ============================================================
# TIER 2
# ============================================================

def run_tier2(question, context):

    # Keep this prompt aligned with the FAISS-RAG baseline.
    prompt = f"""
You must answer ONLY from the provided context.

Rules:
- Copy the answer exactly from the context whenever possible.
- Do not explain.
- Do not write a full sentence.
- Return only the answer phrase.
- If the answer is not present in the context, return: NOT_FOUND

Context:
{context}

Question:
{question}

Answer:
"""

    start = time.time()

    response = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=TIER2_MAX_TOKENS
    )

    latency = time.time() - start

    prediction = (
        response["choices"][0]
        ["message"]["content"]
        .strip()
    )

    return prediction, latency


# ============================================================
# LOAD EXISTING CHECKPOINT
# ============================================================

results = []

if os.path.exists(OUTPUT_FILE):

    try:

        with open(
            OUTPUT_FILE,
            "r",
            encoding="utf-8"
        ) as f:
            results = json.load(f)

        if not isinstance(results, list):
            results = []

    except Exception:

        print(
            "\nWarning: Existing output file could not "
            "be loaded. Starting from zero."
        )

        results = []


start_index = len(results)

if start_index > TEST_QUESTIONS:
    start_index = 0
    results = []

print("\nAlready completed:", start_index)
print("Questions to run:", TEST_QUESTIONS)


# ============================================================
# EXPERIMENT
# ============================================================

experiment_start = time.time()

for idx in range(
    start_index,
    TEST_QUESTIONS
):

    item = questions[idx]

    question = item["question"]

    # Preserve the original NQ answer structure exactly.
    gold_answers = item.get(
        "answers",
        {}
    )

    question_start = time.time()

    print("\n")
    print("=" * 60)
    print(
        f"Question {idx + 1}/{TEST_QUESTIONS}"
    )
    print("Q:", question)

    print("Gold:", gold_answers)


    # ========================================================
    # TIER 1
    # ========================================================

    try:

        (
            tier1_prediction,
            entropy,
            tier1_latency
        ) = run_tier1(question)

    except Exception as e:

        tier1_prediction = f"ERROR: {e}"
        entropy = None
        tier1_latency = time.time() - question_start

        print("\nTier-1 ERROR:", e)


    print(
        "\nTier-1 Prediction:",
        tier1_prediction
    )

    if entropy is not None:

        print(
            "Entropy:",
            round(entropy, 5)
        )

    else:

        print("Entropy: N/A")


    # ========================================================
    # TIER 2 DECISION
    # ========================================================

    tier2_triggered = False

    tier2_prediction = None

    retrieved_docs = []

    retrieval_time = 0.0

    tier2_generation_time = 0.0

    final_prediction = tier1_prediction

    decision = "TIER_1"


    if (
        entropy is not None
        and entropy > ENTROPY_THRESHOLD
    ):

        tier2_triggered = True

        decision = "TIER_2"

        print(
            "\n>>> HIGH ENTROPY"
        )

        print(
            f">>> {entropy:.5f} > "
            f"{ENTROPY_THRESHOLD}"
        )

        print(
            ">>> Triggering FAISS Tier-2..."
        )


        # ----------------------------------------------------
        # RETRIEVAL
        # ----------------------------------------------------

        try:

            (
                context,
                retrieved_docs,
                retrieval_time
            ) = retrieve_faiss(question)

            print(
                "Retrieved documents:",
                len(retrieved_docs)
            )

            print(
                "Retrieval latency:",
                f"{retrieval_time:.3f} sec"
            )


        except Exception as e:

            print(
                "\nRetrieval ERROR:",
                e
            )

            context = ""


        # ----------------------------------------------------
        # TIER 2 GENERATION
        # ----------------------------------------------------

        if context:

            try:

                (
                    tier2_prediction,
                    tier2_generation_time
                ) = run_tier2(
                    question,
                    context
                )

            except Exception as e:

                tier2_prediction = (
                    f"ERROR: {e}"
                )

                tier2_generation_time = 0.0

                print(
                    "\nTier-2 ERROR:",
                    e
                )

        else:

            tier2_prediction = "NOT_FOUND"


        final_prediction = tier2_prediction

        print(
            "\nTier-2 Prediction:",
            tier2_prediction
        )

        print(
            "Tier-2 Generation:",
            f"{tier2_generation_time:.2f} sec"
        )


    else:

        print(
            "\n>>> LOW / ACCEPTABLE ENTROPY"
        )

        if entropy is not None:

            print(
                f">>> {entropy:.5f} <= "
                f"{ENTROPY_THRESHOLD}"
            )

        else:

            print(
                ">>> Entropy unavailable; "
                "accepting Tier-1"
            )

        print(
            ">>> Accepting Tier-1"
        )


    # ========================================================
    # TOTAL LATENCY
    # ========================================================

    total_time = (
        time.time()
        - question_start
    )


    print(
        "\nFinal Prediction:",
        final_prediction
    )

    print(
        "Decision:",
        decision
    )

    print(
        "Total latency:",
        f"{total_time:.2f} sec"
    )


    # ========================================================
    # RAW RESULT
    # ========================================================

    result = {
        "question": question,

        "gold_answers": gold_answers,

        "tier1_prediction": tier1_prediction,

        "tier1_entropy": (
            round(entropy, 6)
            if entropy is not None
            else None
        ),

        "tier2_triggered": tier2_triggered,

        "tier2_prediction": tier2_prediction,

        "final_prediction": final_prediction,

        "decision": decision,

        "retrieved_docs": retrieved_docs,

        "tier1_latency_sec": round(
            tier1_latency,
            4
        ),

        "retrieval_time_sec": round(
            retrieval_time,
            4
        ),

        "tier2_generation_time_sec": round(
            tier2_generation_time,
            4
        ),

        "total_time_sec": round(
            total_time,
            4
        )
    }


    results.append(result)


    # ========================================================
    # CHECKPOINT
    # ========================================================

    if (
        (idx + 1) % CHECKPOINT_EVERY == 0
        or
        (idx + 1) == TEST_QUESTIONS
    ):

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
            "\nCheckpoint saved:",
            idx + 1
        )


# ============================================================
# FINAL RAW SAVE
# ============================================================

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
    - experiment_start
)


# ============================================================
# EXPERIMENT SUMMARY
# ============================================================

tier2_count = sum(
    1
    for result in results
    if result.get("tier2_triggered") is True
)

tier1_count = len(results) - tier2_count

print("\n")
print("=" * 60)
print("TRUSTQA RAW EXPERIMENT COMPLETE")
print("=" * 60)

print(
    "Total questions:",
    len(results)
)

print(
    "Tier-1 accepted:",
    tier1_count
)

print(
    "Tier-2 triggered:",
    tier2_count
)

if results:

    print(
        "Tier-2 trigger rate:",
        round(
            tier2_count / len(results) * 100,
            2
        ),
        "%"
    )

print(
    "Experiment time:",
    round(
        experiment_time / 60,
        2
    ),
    "minutes"
)

print(
    "Saved:",
    OUTPUT_FILE
)

print("=" * 60)