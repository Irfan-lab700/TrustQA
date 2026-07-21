import json
import os
from llama_cpp import Llama

MODEL_PATH = "models/qwen/qwen2.5-3b-instruct-q4_k_m.gguf"
OUTPUT_FILE = "results/baseline_llm_iirc.json"

print("Loading model...")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=6,
    chat_format="chatml",
    verbose=False,
)

print("Model loaded")

questions = []

with open("data/raw/iirc_500.json", "r", encoding="utf-8") as f:

    for line in f:

        item = json.loads(line)

        for q in item["questions"]:

            ans = q["answer"]

            if ans["type"] in ["binary", "none"]:
                continue

            gold_answers = []

            if ans["type"] == "value":
                gold_answers.append(
                    str(ans["answer_value"])
                )

            elif ans["answer_spans"]:

                gold_answers.extend(
                    [
                        x["text"]
                        for x in ans["answer_spans"]
                    ]
                )

            if not gold_answers:
                continue

            questions.append(
                {
                    "question": q["question"],
                    "gold_answers": gold_answers,
                }
            )

print(f"Questions loaded: {len(questions)}")

results = []

for i, sample in enumerate(questions):

    question = sample["question"]

    try:

        response = llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": f"""
Answer the question using only a short answer.

Do not explain.
Do not write a full sentence.
Do not repeat the question.
If unsure, give your best short answer.

Question: {question}
"""
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
            "gold_answers": sample["gold_answers"],
            "prediction": prediction,
        }
    )

    if (i + 1) % 10 == 0:
        print(f"Completed {i+1}/{len(questions)}")

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
                ensure_ascii=False,
            )

        print(f"Saved checkpoint {i+1}")

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

print("Done")