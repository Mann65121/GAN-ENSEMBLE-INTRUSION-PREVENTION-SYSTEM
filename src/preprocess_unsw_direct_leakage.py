import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

print("Preprocessing (WITH DIRECT LEAKAGE)...")

df = pd.read_csv("data/leakage/UNSW_leakage.csv")
df.fillna(0, inplace=True)

# Encode categorical
for col in df.select_dtypes(include="object").columns:
    if col != "label":
        df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop("label", axis=1)
y = df["label"]

# Scaling
X = StandardScaler().fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

os.makedirs("data/leakage_processed", exist_ok=True)

pd.DataFrame(X_train).to_csv("data/leakage_processed/X_train.csv", index=False)
pd.DataFrame(X_test).to_csv("data/leakage_processed/X_test.csv", index=False)
y_train.to_csv("data/leakage_processed/y_train.csv", index=False)
y_test.to_csv("data/leakage_processed/y_test.csv", index=False)

print("✅ Preprocessing done (leakage ON)")
