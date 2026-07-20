from datasets import load_dataset
print("Downloading IIRC...")
iirc = load_dataset("voidful/IIRC")

print(iirc)
print(iirc["train"][0])