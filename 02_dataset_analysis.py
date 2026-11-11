
from datasets import load_dataset
import pandas as pd
from collections import Counter

# Load cached dataset
dataset = load_dataset("lmms-lab/ScienceQA-IMG")

# Convert splits into DataFrames (exclude image objects)
splits = {}

for split in dataset:
    splits[split] = dataset[split].remove_columns(["image"]).to_pandas()

# 1. Dataset overview
print("\n--- DATASET OVERVIEW ---")

for name, df in splits.items():
    print(f"{name}: {len(df)} records")

# 2. Subject distribution
print("\n--- SUBJECT DISTRIBUTION ---")
print(splits["train"]["subject"].value_counts())

# 3. Grade distribution
print("\n--- GRADE DISTRIBUTION ---")
print(splits["train"]["grade"].value_counts())

# 4. Number of answer options
print("\n--- ANSWER OPTION DISTRIBUTION ---")
option_counts = Counter(
    len(choices) for choices in splits["train"]["choices"]
)
print(option_counts)

# 5. Missing values
print("\n--- MISSING VALUES ---")
print(splits["train"].isnull().sum())

# 6. Blank hints
blank_hints = splits["train"]["hint"].fillna("").str.strip().eq("").sum()
print("\nBlank hints:", blank_hints)

# 7. Answer distribution
print("\n--- ANSWER DISTRIBUTION ---")
print(splits["train"]["answer"].value_counts().sort_index())

# 8. Check invalid answer indices
invalid_answers = sum(
    answer < 0 or answer >= len(choices)
    for answer, choices in zip(
        splits["train"]["answer"],
        splits["train"]["choices"]
    )
)

print("\nInvalid answer indices:", invalid_answers)

# 9. Save statistics
splits["train"].describe(include="all").to_csv(
    "data/processed/train_summary.csv"
)

print("\nAnalysis completed.")
