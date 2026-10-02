import pandas as pd

train = pd.read_csv("data/raw/UNSW/UNSW_NB15_training-set.csv")
test  = pd.read_csv("data/raw/UNSW/UNSW_NB15_testing-set.csv")

df = pd.concat([train, test], axis=0).reset_index(drop=True)

# binary label
df["label"] = df["label"].apply(lambda x: 0 if x == 0 else 1)

df.to_csv("data/raw/UNSW_FULL.csv", index=False)

print("✅ Dataset combined:", df.shape)
