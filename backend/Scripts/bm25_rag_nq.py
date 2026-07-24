import json
import os
import pickle
import numpy as np

from llama_cpp import Llama

TOP_K = 3
MODEL_PATH = "models/qwen/qwen2.5-3b-instruct-q4_k_m.gguf"
OUTPUT_FILE = "results/bm25_rag_nq.json"


print("Loading Qwen...")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=6,
    chat_format="chatml",
    verbose=False,
)

print("Model loaded")


print("Loading BM25...")

with open("data/indexes/bm25.pkl", "rb") as f:
    data = pickle.load(f)

bm25 = data["bm25"]
documents = data["documents"]

print(f"Documents: {len(documents)}")


print("Loading NQ...")

nq = []

with open("data/raw/nq_500.json", "r", encoding="utf-8") as f:
    for line in f:
        nq.append(json.loads(line))

print(f"Questions: {len(nq)}")


results = []

if os.path.exists(OUTPUT_FILE):

    with open(
        OUTPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        results = json.load(f)


completed = len(results)

print(f"Already completed: {completed}")


for idx, sample in enumerate(
    nq[completed:],
    start=completed
):

    question = sample["question"]

    query_tokens = question.lower().split()

    scores = bm25.get_scores(query_tokens)

    top_indices = np.argsort(scores)[::-1][:TOP_K]


    contexts = []

    for i in top_indices:
        contexts.append(
            documents[i]["text"]
        )


    context = "\n\n".join(contexts)


    prompt = f""" You must answer ONLY from the provided context. Rules: - Copy the answer exactly from the context whenever possible. - Do not explain. - Do not write a full sentence. - Return only the answer phrase. - If multiple words form the answer, return only those words. - If the answer is not present in the context, return: NOT_FOUND Context: {context[:1500]} Question: {question} Answer: """ 


    try:

        response = llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0,
            max_tokens=30,
        )

        prediction = (
            response["choices"][0]
            ["message"]["content"]
            .strip()
        )

    except Exception as e:

        prediction = f"ERROR: {e}"


    results.append(
        {
            "question": question,
            "gold_answers": sample["answers"],
            "prediction": prediction,
        }
    )


    current = idx + 1

    if current % 10 == 0:
        print(f"Completed {current}/{len(nq)}")


    if current % 50 == 0:

        os.makedirs("results", exist_ok=True)

        with open(
            OUTPUT_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                results,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"Checkpoint saved: {current}")


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False,
    )


print("\nDone")
print(f"Saved: {OUTPUT_FILE}")