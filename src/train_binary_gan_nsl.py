import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("🚀 NSL-KDD Binary IDS (WITH GAN - FIXED)")

# -----------------------
# LOAD DATA
# -----------------------
X_train = pd.read_csv("data/processed/NSL_KDD/X_train.csv").values
y_train = pd.read_csv("data/processed/NSL_KDD/y_train.csv").values.ravel()

X_test = pd.read_csv("data/processed/NSL_KDD/X_test.csv")
y_test = pd.read_csv("data/processed/NSL_KDD/y_test.csv").values.ravel()

# -----------------------
# GAN MODEL
# -----------------------
latent_dim = 32
feature_dim = X_train.shape[1]

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )

    def forward(self, z):
        return self.net(z)

G = Generator()
G.load_state_dict(torch.load("models/gan_NSL_KDD.pth"))
G.eval()

# -----------------------
# GENERATE SYNTHETIC DATA (FIXED)
# -----------------------
attack_data = X_train[y_train == 1]
normal_data = X_train[y_train == 0]

attack_count = len(attack_data)
normal_count = len(normal_data)

print("Attack:", attack_count, "Normal:", normal_count)

if attack_count < normal_count:
    need = normal_count - attack_count
    print("Generating synthetic attacks:", need)

    z = torch.randn(need, latent_dim)
    synthetic = G(z).detach().numpy()

    # 🔥 ADD NOISE (important)
    noise = np.random.normal(0, 0.05, synthetic.shape)
    synthetic = synthetic + noise

    # 🔥 LIMIT SYNTHETIC DATA (avoid perfect learning)
    synthetic = synthetic[:int(0.7 * need)]

    X_final = np.vstack([X_train, synthetic])
    y_final = np.hstack([y_train, np.ones(len(synthetic))])

else:
    print("⚠️ GAN not needed")
    X_final = X_train
    y_final = y_train

print("Final dataset shape:", X_final.shape)

# -----------------------
# MODEL (REGULARIZED)
# -----------------------
model = RandomForestClassifier(
    n_estimators=120,
    max_depth=10,          # 🔥 reduced depth
    min_samples_leaf=5,    # 🔥 prevents memorization
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

# Train
model.fit(X_final, y_final)

# -----------------------
# EVALUATION
# -----------------------
pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print("\n🔥 FINAL GAN RESULT (FIXED)")
print("Accuracy:", acc)
print("\nClassification Report:\n")
print(classification_report(y_test, pred))

# Save model
pickle.dump(model, open("models/gan_binary_nsl_fixed.pkl", "wb"))

print("✅ Model saved")
