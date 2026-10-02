import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import sys
from imblearn.over_sampling import SMOTE

DATASET = sys.argv[1]
base = f"data/processed/{DATASET}"

X = pd.read_csv(f"{base}/X_train.csv").values
y = pd.read_csv(f"{base}/y_train.csv").values.ravel()

normal = sum(y == 0)
attack = sum(y == 1)

latent = 32
features = X.shape[1]

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent, 128),
            nn.ReLU(),
            nn.Linear(128, features)
        )
    def forward(self, z):
        return self.net(z)

# Case handling
if attack >= normal:
    print(f"⚠️ {DATASET}: Attack samples >= Normal samples")
    print("➡ Skipping GAN generation, applying SMOTE only")

    X_final, y_final = SMOTE().fit_resample(X, y)

else:
    print(f"✅ {DATASET}: Normal > Attack, using GAN + SMOTE")

    need = normal - attack

    G = Generator()
    G.load_state_dict(torch.load(f"models/gan_{DATASET}.pth"))
    G.eval()

    z = torch.randn(need, latent)
    synthetic = G(z).detach().numpy()

    X_gan = np.vstack([X, synthetic])
    y_gan = np.hstack([y, np.ones(need)])

    X_final, y_final = SMOTE().fit_resample(X_gan, y_gan)

pd.DataFrame(X_final).to_csv(f"{base}/X_final.csv", index=False)
pd.Series(y_final).to_csv(f"{base}/y_final.csv", index=False)

print(f"✅ Balanced dataset ready for {DATASET}")
print("Final class distribution:", np.bincount(y_final.astype(int)))
