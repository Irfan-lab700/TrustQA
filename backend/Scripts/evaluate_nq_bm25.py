import json
import pickle

TOP_KS = [1, 5, 10]

# Load BM25

with open("data/indexes/bm25.pkl", "rb") as f:
    bm25_data = pickle.load(f)

bm25 = bm25_data["bm25"]
documents = bm25_data["documents"]

# same tokenizer used before
def tokenize(text):
    return text.lower().split()

# Metrics

hits = {1: 0, 5: 0, 10: 0}
total = 0

# Evaluate

with open(
    "data/raw/nq_500.json",
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        sample = json.loads(line)

        question = sample["question"]

        answers = [
            a.lower().strip()
            for a in sample["answers"]["text"]
        ]

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

            retrieved_text = " ".join(
                documents[idx]["text"].lower()
                for idx in top_docs
            )

            found = any(
                answer in retrieved_text
                for answer in answers
            )

            if found:
                hits[k] += 1

# Results

print("\nBM25 RESULTS\n")

for k in TOP_KS:

    recall = hits[k] / total

    print(
        f"Recall@{k}: {recall:.4f}"
    )

print(f"\nQuestions: {total}")