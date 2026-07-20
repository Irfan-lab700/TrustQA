import json
import pickle

TOP_KS = [1, 5, 10]

with open(
    "data/indexes/bm25.pkl",
    "rb"
) as f:

    bm25_data = pickle.load(f)

bm25 = bm25_data["bm25"]
documents = bm25_data["documents"]

def tokenize(text):
    return text.lower().split()

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

hits = {1:0, 5:0, 10:0}
total = 0

with open(
    "data/raw/iirc_500.json",
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        article = json.loads(line)

        for q in article["questions"]:

            question = q["question"]

            answers = extract_answer(
                q["answer"]
            )

            if not answers:
                continue

            scores = bm25.get_scores(
                tokenize(question)
            )

            ranked = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )

            total += 1

            for k in TOP_KS:

                top_docs = ranked[:k]

                retrieved = " ".join(
                    documents[idx]["text"].lower()
                    for idx in top_docs
                )

                if any(
                    ans in retrieved
                    for ans in answers
                ):
                    hits[k] += 1

print("\nBM25 IIRC RESULTS\n")

for k in TOP_KS:

    print(
        f"Recall@{k}: {hits[k]/total:.4f}"
    )

print(f"\nQuestions: {total}")