import json
import os
import pickle
import time

import numpy as np

from llama_cpp import Llama


TOP_K = 3

MODEL_PATH = "models/qwen/qwen2.5-3b-instruct-q4_k_m.gguf"

OUTPUT_FILE = "results/bm25_rag_iirc.json"


print("Loading Qwen...")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=6,
    chat_format="chatml",
    verbose=False,
)

print("Qwen loaded")


print("Loading BM25...")

with open(
    "data/indexes/bm25.pkl",
    "rb"
) as f:

    data = pickle.load(f)

bm25 = data["bm25"]
documents = data["documents"]

print(
    "Documents:",
    len(documents)
)


print("Loading IIRC...")

qa_data = []

with open(
    "data/raw/iirc_500.json",
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        sample = json.loads(line)

        for q in sample["questions"]:

            answer_type = q["answer"]["type"]

            if answer_type not in [
                "span",
                "value"
            ]:
                continue

            question = q["question"]

            if answer_type == "span":

                answers = []

                for span in q["answer"]["answer_spans"]:

                    answers.append(
                        span["text"]
                    )

            else:

                answers = [
                    q["answer"]["answer_value"]
                ]

            qa_data.append(
                {
                    "question": question,
                    "answers": answers
                }
            )


print(
    "Questions:",
    len(qa_data)
)


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


experiment_start = time.time()


for idx in range(
    start,
    len(qa_data)
):

    sample = qa_data[idx]

    question = sample["question"]


    total_start = time.time()


    retrieval_start = time.time()


    query_tokens = (
        question
        .lower()
        .split()
    )

    scores = bm25.get_scores(
        query_tokens
    )

    top_indices = (
        np.argsort(scores)[::-1]
        [:TOP_K]
    )


    retrieval_time = (
        time.time()
        -
        retrieval_start
    )


    contexts = []

    retrieved_docs = []


    for doc_idx in top_indices:

        doc = documents[doc_idx]

        contexts.append(
            doc["text"]
        )

        retrieved_docs.append(
            {
                "doc_id":
                    doc["doc_id"],

                "score":
                    float(
                        scores[doc_idx]
                    ),

                "text":
                    doc["text"]
            }
        )


    context = "\n\n".join(
        contexts
    )


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

        response = (
            llm.create_chat_completion(

                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                temperature=0,

                max_tokens=30
            )
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
            "question":
                question,

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
print(
    "Saved:",
    OUTPUT_FILE
)

print(
    "Total experiment time (minutes):",
    round(
        experiment_time / 60,
        2
    )
)