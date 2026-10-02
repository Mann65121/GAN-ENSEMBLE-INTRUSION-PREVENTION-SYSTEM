import torch, torch.nn as nn
import pandas as pd
import sys, os
from torch.utils.data import DataLoader, TensorDataset

DATASET = sys.argv[1]
base = f"data/processed/{DATASET}"

X = pd.read_csv(f"{base}/X_train.csv").values
y = pd.read_csv(f"{base}/y_train.csv").values.ravel()

X_attack = torch.tensor(X[y == 1], dtype=torch.float32)

latent = 32
features = X_attack.shape[1]

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent,128),
            nn.ReLU(),
            nn.Linear(128,features)
        )
    def forward(self,z): return self.net(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(features,128),
            nn.ReLU(),
            nn.Linear(128,1),
            nn.Sigmoid()
        )
    def forward(self,x): return self.net(x)

G, D = Generator(), Discriminator()
optG = torch.optim.Adam(G.parameters(), 0.0002)
optD = torch.optim.Adam(D.parameters(), 0.0002)
loss = nn.BCELoss()

loader = DataLoader(TensorDataset(X_attack), batch_size=64, shuffle=True)

for epoch in range(30):
    for (real,) in loader:
        bs = real.size(0)
        z = torch.randn(bs, latent)
        fake = G(z)

        dloss = loss(D(real), torch.ones(bs,1)) + loss(D(fake.detach()), torch.zeros(bs,1))
        optD.zero_grad(); dloss.backward(); optD.step()

        gloss = loss(D(fake), torch.ones(bs,1))
        optG.zero_grad(); gloss.backward(); optG.step()

    print(f"{DATASET} | Epoch {epoch+1}/30")

os.makedirs("models", exist_ok=True)
torch.save(G.state_dict(), f"models/gan_{DATASET}.pth")
print("✅ GAN trained for", DATASET)
