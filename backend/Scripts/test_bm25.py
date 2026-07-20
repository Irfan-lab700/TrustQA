import pickle

query = "How long was the reign of the person who annexed  Aurangabad, Maharashtra in 1308?"

with open("data/indexes/bm25.pkl", "rb") as f:
    data = pickle.load(f)

bm25 = data["bm25"]
documents = data["documents"]

query_tokens = query.lower().split()

scores = bm25.get_scores(query_tokens)

top_indices = sorted(
    range(len(scores)),
    key=lambda i: scores[i],
    reverse=True
)[:5]

print("\nQUESTION:")
print(query)

print("\nTOP 5 RESULTS:\n")

for rank, idx in enumerate(top_indices, start=1):

    print("=" * 80)
    print(f"Rank {rank}")
    print(f"Doc ID: {documents[idx]['doc_id']}")
    print(documents[idx]["text"][:500])
    print()