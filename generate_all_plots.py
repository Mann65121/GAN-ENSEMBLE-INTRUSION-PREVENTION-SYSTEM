import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# -----------------------------
# FUNCTION TO GENERATE ALL PLOTS
# -----------------------------
def generate_plots(dataset_name, model_path, X_test_path, y_test_path):

    print(f"\n🚀 Processing {dataset_name}...")

    # Load model
    model = pickle.load(open(model_path, "rb"))

    # Load data
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # -----------------------------
    # ROC CURVE
    # -----------------------------
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{dataset_name} ROC Curve")
    plt.legend()
    plt.savefig(f"{dataset_name}_roc.png")
    plt.close()

    # -----------------------------
    # PRECISION-RECALL CURVE
    # -----------------------------
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{dataset_name} PR Curve")
    plt.savefig(f"{dataset_name}_pr.png")
    plt.close()

    # -----------------------------
    # CONFUSION MATRIX
    # -----------------------------
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"{dataset_name} Confusion Matrix")
    plt.savefig(f"{dataset_name}_cm.png")
    plt.close()

    print(f"✅ Plots generated for {dataset_name}")


# -----------------------------
# RUN FOR NSL-KDD
# -----------------------------
generate_plots(
    dataset_name="NSL_KDD",
    model_path="models/gan_binary_nsl_fixed.pkl",
    X_test_path="data/processed/NSL_KDD/X_test.csv",
    y_test_path="data/processed/NSL_KDD/y_test.csv"
)

# -----------------------------
# RUN FOR UNSW
# -----------------------------
generate_plots(
    dataset_name="UNSW",
    model_path="models/binary_indirect_no_attackcat.pkl",  # change if needed
    X_test_path="data/indirect_processed/X_test.csv",
    y_test_path="data/indirect_processed/y_test.csv"
)


# -----------------------------
# ACCURACY BAR GRAPH (MANUAL)
# -----------------------------
models = ["NSL (No GAN)", "NSL (GAN)", "UNSW Clean", "UNSW Leakage"]
accuracy = [98, 99.1, 87, 95]   # update if needed

plt.figure()
plt.bar(models, accuracy)
plt.ylabel("Accuracy (%)")
plt.title("Model Comparison")
plt.xticks(rotation=20)
plt.savefig("accuracy_comparison.png")
plt.close()

print("\n🎯 ALL PNG FILES GENERATED SUCCESSFULLY!")
