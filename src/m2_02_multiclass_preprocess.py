import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

print("Step‑2: Multiclass preprocessing started")

# 1️⃣ Load dataset
df = pd.read_csv("data/raw/UNSW_NB15.csv")
df.fillna(0, inplace=True)

# 2️⃣ Target = attack category (multiclass)
y = df["attack_cat"]

# 3️⃣ Drop leakage + non‑feature columns
drop_cols = [
    "attack_cat",   # target
    "label",        # binary label (Month‑1)
    "ct_state_ttl", "ct_dst_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm",
    "ct_dst_ltm", "ct_src_ltm", "ct_srv_dst"
]

X = df.drop(columns=[c for c in drop_cols if c in df.columns])

# 4️⃣ Encode categorical features
for col in X.select_dtypes(include="object").columns:
    X[col] = LabelEncoder().fit_transform(X[col])

# 5️⃣ Encode multiclass labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Attack classes:", list(label_encoder.classes_))

# 6️⃣ Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7️⃣ Train / Test split (STRATIFIED)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.30,
    stratify=y_encoded,
    random_state=42
)

# 8️⃣ Save processed files
os.makedirs("data/month2", exist_ok=True)

pd.DataFrame(X_train).to_csv("data/month2/X_train_mc.csv", index=False)
pd.DataFrame(X_test).to_csv("data/month2/X_test_mc.csv", index=False)
pd.Series(y_train).to_csv("data/month2/y_train_mc.csv", index=False)
pd.Series(y_test).to_csv("data/month2/y_test_mc.csv", index=False)

print("✅ Multiclass preprocessing DONE")
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
