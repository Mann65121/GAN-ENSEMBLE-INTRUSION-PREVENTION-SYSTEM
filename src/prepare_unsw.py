import pandas as pd
import os

train_path = "data/raw/UNSW/UNSW_NB15_training-set.csv"
test_path  = "data/raw/UNSW/UNSW_NB15_testing-set.csv"

df_train = pd.read_csv(train_path)
df_test  = pd.read_csv(test_path)

df = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)

# Ensure binary label
# attack_cat exists but we use 'label'
df["label"] = df["label"].apply(lambda x: 0 if x == 0 else 1)

out_path = "data/raw/UNSW_NB15.csv"
df.to_csv(out_path, index=False)

print("✅ UNSW_NB15.csv created")
print("Shape:", df.shape)
