import pandas as pd
import sys, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATASET = sys.argv[1]

df = pd.read_csv(f"data/raw/{DATASET}.csv")
df.fillna(0, inplace=True)

# 🚨 REMOVE LEAKAGE FEATURES (UNSW ONLY)
if DATASET == "UNSW_NB15":
    leakage_cols = [
        "attack_cat", "ct_state_ttl", "ct_dst_src_ltm",
        "ct_src_dport_ltm", "ct_dst_sport_ltm",
        "ct_dst_ltm", "ct_src_ltm", "ct_srv_dst"
    ]
    for col in leakage_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

# Encode categoricals
for col in df.select_dtypes(include="object").columns:
    if col != "label":
        df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop("label", axis=1)
y = df["label"]

X = StandardScaler().fit_transform(X)

# 🔥 STRONGER SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

out = f"data/processed/{DATASET}"
os.makedirs(out, exist_ok=True)

pd.DataFrame(X_train).to_csv(f"{out}/X_train.csv", index=False)
pd.DataFrame(X_test).to_csv(f"{out}/X_test.csv", index=False)
y_train.to_csv(f"{out}/y_train.csv", index=False)
y_test.to_csv(f"{out}/y_test.csv", index=False)

print(f"✅ Preprocessing done (leakage‑safe) for {DATASET}")
