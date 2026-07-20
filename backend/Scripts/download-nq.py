from datasets import load_dataset

print("Downloading NQ...")

nq = load_dataset("llukas22/nq-simplified")

print(nq)

print("\nSample Question:")
print(nq["train"][0]["question"])