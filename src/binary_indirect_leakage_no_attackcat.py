import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

print("🚀 Preparing Binary Dataset (Indirect Leakage, NO attack_cat)...")

# -----------------------------
# STEP 1: LOAD + PREPARE
# -----------------------------
train = pd.read_csv("data/raw/UNSW/UNSW_NB15_training-set.csv")
test  = pd.read_csv("data/raw/UNSW/UNSW_NB15_testing-set.csv")

df = pd.concat([train, test], axis=0).reset_index(drop=True)

# Binary label
df["label"] = df["label"].apply(lambda x: 0 if x == 0 else 1)

# ❌ REMOVE ONLY attack_cat (direct leakage)
if "attack_cat" in df.columns:
    df.drop(columns=["attack_cat"], inplace=True)

# ✅ KEEP ct_* + normal features

# -----------------------------
# STEP 2: PREPROCESS
# -----------------------------
df.fillna(0, inplace=True)

# Encode categorical
for col in df.select_dtypes(include="object").columns:
    if col != "label":
        df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop("label", axis=1)
y = df["label"]

# Scale
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

print("Train shape:", X_train.shape)

# -----------------------------
# STEP 3: TRAIN MODEL
# -----------------------------
print("\n🚀 Training Binary IDS (Indirect Leakage)...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=18,
    min_samples_leaf=2,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# STEP 4: EVALUATION
# -----------------------------
pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print("\n🔥 FINAL RESULT (INDIRECT LEAKAGE - NO attack_cat)")
print("Accuracy:", acc)
print("\nClassification Report:\n")
print(classification_report(y_test, pred))

# Save model
os.makedirs("models", exist_ok=True)
pickle.dump(model, open("models/binary_indirect_no_attackcat.pkl", "wb"))

print("✅ Model saved")
