import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

df = pd.read_csv("data/raw/UNSW_FULL.csv")
df.fillna(0, inplace=True)

# 🔥 KEEP ALL FEATURES (INCLUDING attack_cat)

for col in df.select_dtypes(include="object").columns:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop("label", axis=1)
y = df["label"]

X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

os.makedirs("data/leakage_binary", exist_ok=True)

pd.DataFrame(X_train).to_csv("data/leakage_binary/X_train.csv", index=False)
pd.DataFrame(X_test).to_csv("data/leakage_binary/X_test.csv", index=False)
y_train.to_csv("data/leakage_binary/y_train.csv", index=False)
y_test.to_csv("data/leakage_binary/y_test.csv", index=False)

print("✅ Leakage Binary preprocessing done")
