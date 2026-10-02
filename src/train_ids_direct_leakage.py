import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

print("🚀 Training Binary IDS (DIRECT LEAKAGE)...")

X_train = pd.read_csv("data/leakage_processed/X_train.csv")
y_train = pd.read_csv("data/leakage_processed/y_train.csv").values.ravel()

X_test = pd.read_csv("data/leakage_processed/X_test.csv")
y_test = pd.read_csv("data/leakage_processed/y_test.csv").values.ravel()

# Models
et = ExtraTreesClassifier(
    n_estimators=200,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method="hist",
    n_jobs=-1,
    random_state=42
)

lgbm = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    n_jobs=-1,
    random_state=42
)

# Stacking
model = StackingClassifier(
    estimators=[
        ("et", et),
        ("xgb", xgb),
        ("lgbm", lgbm)
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    n_jobs=-1
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print("\n🔥 FINAL RESULT (DIRECT LEAKAGE)")
print("Accuracy:", acc)
print(classification_report(y_test, pred))

pickle.dump(model, open("models/direct_leakage_ids.pkl","wb"))
