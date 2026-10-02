import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

print("Preprocessing (INDIRECT LEAKAGE)...")

df = pd.read_csv("data/indirect_leakage/unsw_indirect.csv")
df.fillna(0, inplace=True)

# Encode categorical
for col in df.select_dtypes(include="object").columns:
    if col != "label":
        df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop("label", axis=1)   # ct_* included
y = df["label"]

# Scale
X = StandardScaler().fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

os.makedirs("data/indirect_processed", exist_ok=True)

pd.DataFrame(X_train).to_csv("data/indirect_processed/X_train.csv", index=False)
pd.DataFrame(X_test).to_csv("data/indirect_processed/X_test.csv", index=False)
y_train.to_csv("data/indirect_processed/y_train.csv", index=False)
y_test.to_csv("data/indirect_processed/y_test.csv", index=False)

print("✅ Preprocessing done")
