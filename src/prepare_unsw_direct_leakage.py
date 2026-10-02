import pandas as pd
import os

print("Loading UNSW dataset...")

train = pd.read_csv("data/raw/UNSW/UNSW_NB15_training-set.csv")
test  = pd.read_csv("data/raw/UNSW/UNSW_NB15_testing-set.csv")

df = pd.concat([train, test], axis=0).reset_index(drop=True)

# Binary label
df["label"] = df["label"].apply(lambda x: 0 if x == 0 else 1)

os.makedirs("data/leakage", exist_ok=True)
df.to_csv("data/leakage/UNSW_leakage.csv", index=False)

print("✅ Leakage dataset ready")
print("Shape:", df.shape)
