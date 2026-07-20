from datasets import load_dataset

nq = load_dataset("llukas22/nq-simplified")
iirc = load_dataset("voidful/IIRC")

nq_500 = nq["train"].shuffle(seed=42).select(range(500))
iirc_500 = iirc["train"].shuffle(seed=42).select(range(500))

nq_500.to_json("data/raw/nq_500.json")
iirc_500.to_json("data/raw/iirc_500.json")

print("Saved successfully")