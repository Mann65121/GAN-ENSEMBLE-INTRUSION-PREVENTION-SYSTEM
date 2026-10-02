import pandas as pd
import os

print("Loading UNSW dataset...")

train = pd.read_csv("data/raw/UNSW/UNSW_NB15_training-set.csv")
test  = pd.read_csv("data/raw/UNSW/UNSW_NB15_testing-set.csv")

df = pd.concat([train, test], axis=0).reset_index(drop=True)

# Binary label already present (0/1)
df["label"] = df["label"].apply(lambda x: 0 if x == 0 else 1)

# ❌ REMOVE ONLY attack_cat
if "attack_cat" in df.columns:
    df.drop(columns=["attack_cat"], inplace=True)

os.makedirs("data/indirect_leakage", exist_ok=True)
df.to_csv("data/indirect_leakage/unsw_indirect.csv", index=False)

print("✅ Dataset ready (attack_cat removed, ct_* kept)")
print("Columns:", df.columns.tolist())
