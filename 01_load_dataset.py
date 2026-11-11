
from datasets import load_dataset

print("Loading ScienceQA-IMG dataset...")

dataset = load_dataset("lmms-lab/ScienceQA-IMG")

print("\nDataset information:")
print(dataset)

print("\nSplit sizes:")
for split in dataset:
    print(f"{split}: {len(dataset[split])} records")

print("\nAvailable columns:")
print(dataset["train"].column_names)

print("\nFirst training record:")
print(dataset["train"][0])
