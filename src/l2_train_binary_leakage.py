import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

X_train = pd.read_csv("data/leakage_binary/X_train.csv")
y_train = pd.read_csv("data/leakage_binary/y_train.csv").values.ravel()

X_test = pd.read_csv("data/leakage_binary/X_test.csv")
y_test = pd.read_csv("data/leakage_binary/y_test.csv").values.ravel()

model = RandomForestClassifier(
    n_estimators=120,
    max_depth=12,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("🔥 Binary Leakage Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

pickle.dump(model, open("models/binary_leakage.pkl","wb"))
