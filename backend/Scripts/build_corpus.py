from datasets import load_dataset
import json

corpus = []

# NQ
nq = load_dataset("llukas22/nq-simplified")

nq_500 = nq["train"].shuffle(seed=42).select(range(500))

for idx, sample in enumerate(nq_500):
    corpus.append({
        "doc_id": f"nq_{idx}",
        "text": sample["context"]
    })

# IIRC
iirc = load_dataset("voidful/IIRC")

iirc_500 = iirc["train"].shuffle(seed=42).select(range(500))

for idx, sample in enumerate(iirc_500):

    corpus.append({
        "doc_id": f"iirc_main_{idx}",
        "text": sample["text"]
    })

    for q in sample["questions"]:
        for ctx in q["context"]:
            corpus.append({
                "doc_id": f"iirc_ctx_{idx}",
                "text": ctx["text"]
            })

with open("data/processed/corpus.jsonl", "w", encoding="utf-8") as f:
    for doc in corpus:
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")

print(f"Corpus Size: {len(corpus)}")