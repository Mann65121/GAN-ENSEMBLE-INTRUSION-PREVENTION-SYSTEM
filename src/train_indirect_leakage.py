import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

print("🚀 Training IDS (INDIRECT LEAKAGE)...")

X_train = pd.read_csv("data/indirect_processed/X_train.csv")
y_train = pd.read_csv("data/indirect_processed/y_train.csv").values.ravel()

X_test = pd.read_csv("data/indirect_processed/X_test.csv")
y_test = pd.read_csv("data/indirect_processed/y_test.csv").values.ravel()

# Models (balanced, no SMOTE)
et = ExtraTreesClassifier(
    n_estimators=250,
    max_depth=30,
    class_weight="balanced",
    n_jobs=1,
    random_state=42
)

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=9,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method="hist",
    n_jobs=1,
    random_state=42
)

lgbm = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    class_weight="balanced",
    n_jobs=1,
    random_state=42
)

model = VotingClassifier(
    estimators=[
        ("et", et),
        ("xgb", xgb),
        ("lgbm", lgbm)
    ],
    voting="soft",
    n_jobs=1
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print("\n🔥 FINAL RESULT (INDIRECT LEAKAGE)")
print("Accuracy:", acc)
print(classification_report(y_test, pred))
