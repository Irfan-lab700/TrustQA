import json

INPUT_FILE = "data/processed/corpus_dedup.jsonl"
OUTPUT_FILE = "data/processed/corpus_final.jsonl"

CHUNK_SIZE = 250  # words

total_docs = 0
final_docs = 0
chunked_docs = 0

with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

    for line in fin:
        total_docs += 1

        doc = json.loads(line)
        words = doc["text"].split()

        # Normal passage
        if len(words) <= 500:
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
            final_docs += 1

        # Long passage -> chunk
        else:
            chunked_docs += 1

            for idx in range(0, len(words), CHUNK_SIZE):

                chunk_words = words[idx:idx + CHUNK_SIZE]

                chunk_doc = {
                    "doc_id": f"{doc['doc_id']}_chunk_{idx//CHUNK_SIZE}",
                    "text": " ".join(chunk_words)
                }

                fout.write(
                    json.dumps(chunk_doc, ensure_ascii=False) + "\n"
                )

                final_docs += 1

print(f"Original Docs: {total_docs}")
print(f"Long Docs Chunked: {chunked_docs}")
print(f"Final Docs: {final_docs}")