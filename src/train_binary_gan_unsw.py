import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("🚀 UNSW Binary IDS (WITH GAN)")

# -----------------------
# LOAD DATA
# -----------------------
X_train = pd.read_csv("data/processed/UNSW_NB15/X_train.csv").values
y_train = pd.read_csv("data/processed/UNSW_NB15/y_train.csv").values.ravel()

X_test = pd.read_csv("data/processed/UNSW_NB15/X_test.csv")
y_test = pd.read_csv("data/processed/UNSW_NB15/y_test.csv").values.ravel()

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
G.load_state_dict(torch.load("models/gan_UNSW_NB15.pth"))
G.eval()

# -----------------------
# GENERATE SYNTHETIC ATTACKS
# -----------------------
attack_data = X_train[y_train == 1]
normal_data = X_train[y_train == 0]

need = len(normal_data) - len(attack_data)

print("Generating synthetic attacks:", need)

z = torch.randn(need, latent_dim)
synthetic = G(z).detach().numpy()

# Combine
X_final = np.vstack([X_train, synthetic])
y_final = np.hstack([y_train, np.ones(need)])

print("Final dataset size:", X_final.shape)

# -----------------------
# TRAIN MODEL
# -----------------------
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=15,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

model.fit(X_final, y_final)

# -----------------------
# EVALUATION
# -----------------------
pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print("\n🔥 UNSW GAN RESULT")
print("Accuracy:", acc)
print(classification_report(y_test, pred))

pickle.dump(model, open("models/gan_binary_unsw.pkl", "wb"))
