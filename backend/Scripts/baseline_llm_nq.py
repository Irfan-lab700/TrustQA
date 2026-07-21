import json
import os
from llama_cpp import Llama

#config 
MODEL_PATH = "models/qwen/qwen2.5-3b-instruct-q4_k_m.gguf"
NQ_FILE = "data/raw/nq_500.json"
OUTPUT_FILE = "results/baseline_llm_nq_new.json"

#load model
print("Loading Qwen model...\n")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=6,      # adjust if needed
    chat_format="chatml",
    verbose=False
)

print("Model loaded.\n")


#load questions 
questions = []

with open(NQ_FILE, "r", encoding="utf-8") as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        item = json.loads(line)

        questions.append(
            {
                "question": item["question"],
                "answers": item.get("answers", [])
            }
        )

print(f"Loaded {len(questions)} questions.\n")

#answer generate 
results = []

for i, sample in enumerate(questions):

    question = sample["question"]

    try:

        response = llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content":
                    f"""Give only the final answer.

Do not write a sentence.
Do not explain.
Do not repeat the question.
Answer in 1-5 words whenever possible.

Examples:
Q: Who played James Bond in Licence to Kill?
A: Timothy Dalton

Q: When was the Battle of Little Bighorn fought?
A: June 25–26, 1876

Question: {question}"""
                }
            ],
            max_tokens=30,
            temperature=0
        )

        prediction = (
            response["choices"][0]
            ["message"]["content"]
            .strip()
        )

    except Exception as e:

        prediction = f"ERROR: {str(e)}"

    results.append(
        {
            "question": question,
            "gold_answers": sample["answers"],
            "prediction": prediction
        }
    )

    # Progress
    if (i + 1) % 10 == 0:
        print(f"Completed {i+1}/{len(questions)}")

    # Autosave every 50
    if (i + 1) % 50 == 0:

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
                ensure_ascii=False
            )

        print(f"Saved checkpoint at {i+1}")

#final save 
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
        ensure_ascii=False
    )

print("\nDone.")
print(f"Saved to: {OUTPUT_FILE}")