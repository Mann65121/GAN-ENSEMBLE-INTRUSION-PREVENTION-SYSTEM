import pandas as pd
import sys, pickle, os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

DATASET = sys.argv[1]
base = f"data/processed/{DATASET}"

X = pd.read_csv(f"{base}/X_final.csv")
y = pd.read_csv(f"{base}/y_final.csv").values.ravel()

rf = RandomForestClassifier(
    n_estimators=120,
    max_depth=12,              # 🔥 prevents memorization
    min_samples_leaf=5,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

gb = GradientBoostingClassifier(
    n_estimators=80,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

lr = LogisticRegression(
    max_iter=1000,
    C=0.5,                     # 🔥 regularization
    n_jobs=-1
)

model = VotingClassifier(
    estimators=[("rf", rf), ("gb", gb), ("lr", lr)],
    voting="soft",
    n_jobs=-1
)

model.fit(X, y)

os.makedirs("models", exist_ok=True)
pickle.dump(model, open(f"models/ensemble_{DATASET}.pkl", "wb"))

print(f"✅ Regularized ensemble trained for {DATASET}")
