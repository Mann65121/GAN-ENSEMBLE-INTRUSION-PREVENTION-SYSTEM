import pandas as pd
import sys, pickle, os
from sklearn.metrics import classification_report, f1_score

DATASET = sys.argv[1]
base = f"data/processed/{DATASET}"

X_test = pd.read_csv(f"{base}/X_test.csv")
y_test = pd.read_csv(f"{base}/y_test.csv")

model = pickle.load(open(f"models/ensemble_{DATASET}.pkl","rb"))
pred = model.predict(X_test)

report = classification_report(y_test, pred, digits=4)
print(report)

os.makedirs("results", exist_ok=True)
with open(f"results/{DATASET}_report.txt","w") as f:
    f.write(report)

print("Attack F1 =", f1_score(y_test, pred, pos_label=1))
