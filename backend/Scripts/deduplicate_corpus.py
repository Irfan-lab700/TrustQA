import json

seen = set()
unique_docs = []

with open("data/processed/corpus.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)

        text = doc["text"].strip()

        if text not in seen:
            seen.add(text)
            unique_docs.append(doc)

with open("data/processed/corpus_dedup.jsonl", "w", encoding="utf-8") as f:
    for doc in unique_docs:
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")

print("Original:", len(seen) + (3505-len(seen)))  # optional
print("Unique:", len(unique_docs))