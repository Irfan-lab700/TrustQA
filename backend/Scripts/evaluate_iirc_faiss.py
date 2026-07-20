import json
import pickle
import faiss

from sentence_transformers import SentenceTransformer

TOP_KS = [1,5,10]

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

index = faiss.read_index(
    "data/indexes/faiss.index"
)

with open(
    "data/indexes/faiss_docs.pkl",
    "rb"
) as f:

    documents = pickle.load(f)

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

hits = {1:0,5:0,10:0}
total = 0

with open(
    "data/raw/iirc_500.json",
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        article = json.loads(line)

        for q in article["questions"]:

            answers = extract_answer(
                q["answer"]
            )

            if not answers:
                continue

            emb = model.encode(
                [q["question"]],
                convert_to_numpy=True
            )

            _, indices = index.search(
                emb,
                10
            )

            total += 1

            for k in TOP_KS:

                retrieved = " ".join(
                    documents[idx]["text"].lower()
                    for idx in indices[0][:k]
                )

                if any(
                    ans in retrieved
                    for ans in answers
                ):
                    hits[k] += 1

print("\nFAISS IIRC RESULTS\n")

for k in TOP_KS:

    print(
        f"Recall@{k}: {hits[k]/total:.4f}"
    )

print(f"\nQuestions: {total}")