import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


print("Loading dataset...")

X_train = pd.read_csv("data/month2/X_train_mc.csv")
y_train = pd.read_csv("data/month2/y_train_mc.csv").values.ravel()

X_test = pd.read_csv("data/month2/X_test_mc.csv")
y_test = pd.read_csv("data/month2/y_test_mc.csv").values.ravel()


print("Building stacked ensemble...")

base_models = [

("et", ExtraTreesClassifier(
    n_estimators=200,
    class_weight="balanced",
    n_jobs=1,
    random_state=42
)),

("xgb", XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    n_jobs=1,
    random_state=42
)),

("lgbm", LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    n_jobs=1,
    random_state=42
))

]


meta_model = LogisticRegression(max_iter=1000)


stack_model = StackingClassifier(

    estimators=base_models,
    final_estimator=meta_model,
    n_jobs=1

)


print("Training stacked IDS model...")

stack_model.fit(X_train, y_train)


print("Generating predictions...")

pred = stack_model.predict(X_test)

acc = accuracy_score(y_test, pred)

report = classification_report(y_test, pred)

print("\n🔥 Final Accuracy:", acc)

print("\nClassification Report:\n")
print(report)


# ---------------------------------
# Save results to file
# ---------------------------------

print("Saving results...")

with open("results_final_88.txt", "w") as f:
    f.write("Final IDS Result\n")
    f.write("----------------------------\n")
    f.write(f"Accuracy: {acc}\n\n")
    f.write(report)

print("Results saved to results_final_88.txt")
