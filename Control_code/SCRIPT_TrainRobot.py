"""
Train a twin-tower MLP on raw binaural sonar to predict:
1) closest obstacle distance (from camera profile min distance)
2) left/right/center side based on average profile distance difference (with threshold).
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader

from Library import Utils
from Library import DataProcessor

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

seed = 123
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, flush=True)

# -----------------------------
# Settings
# -----------------------------
az_steps = 15
max_extent = 25
use_raw_sonar = True
batch_size_train = 64
batch_size_val = 128
drop_p = 0.1
epochs = 300
patience = 60
side_loss_weight = 0.3
same_thresh_mm = 0
plots_dir = Path("Plots")
plots_dir.mkdir(exist_ok=True)
models_dir = Path("Models")
models_dir.mkdir(exist_ok=True)

# -----------------------------
# Load and collate data
# -----------------------------
processor1 = DataProcessor.DataProcessor("session6")
processor2 = DataProcessor.DataProcessor("session7")
processor3 = DataProcessor.DataProcessor("session8")

collated1 = processor1.collate_data(az_min=-max_extent, az_max=max_extent, az_steps=az_steps)
collated2 = processor2.collate_data(az_min=-max_extent, az_max=max_extent, az_steps=az_steps)
collated3 = processor3.collate_data(az_min=-max_extent, az_max=max_extent, az_steps=az_steps)

collated_results = [collated1, collated2, collated3]
profiles = DataProcessor.collect(collated_results, "profiles")
sonar_flat = DataProcessor.collect(collated_results, "sonar_data")

# -----------------------------
# Ground-truth targets from camera data
# -----------------------------
closest_visual_distance = Utils.get_extrema_values(profiles, "min") / 1000.0  # meters

mid = profiles.shape[1] // 2
left_avg = np.mean(profiles[:, :mid], axis=1)
right_avg = np.mean(profiles[:, mid + 1 :], axis=1)
diff_mm = right_avg - left_avg
avg_side = np.sign(diff_mm)
avg_side[np.abs(diff_mm) <= same_thresh_mm] = 0.0

# side classes: -1 -> 0, 0 -> 1, +1 -> 2
side_class = (avg_side + 1).astype(np.int64)

# -----------------------------
# Build feature matrix
# -----------------------------
if use_raw_sonar:
    X = sonar_flat.astype(np.float32)
else:
    raise ValueError("No sonar features selected.")

y_reg_raw = closest_visual_distance.astype(np.float32)
y_reg = np.log1p(y_reg_raw)

valid = np.isfinite(X).all(axis=1)
valid &= np.isfinite(y_reg)
valid &= np.isfinite(side_class)

X = X[valid]
y_reg = y_reg[valid]
y_reg_raw = y_reg_raw[valid]
side_class = side_class[valid]

# -----------------------------
# Train/val split
# -----------------------------
X_train, X_val, y_reg_train, y_reg_val, y_reg_raw_train, y_reg_raw_val, side_train, side_val = train_test_split(
    X,
    y_reg,
    y_reg_raw,
    side_class,
    test_size=0.2,
    random_state=seed,
    shuffle=True,
)

# -----------------------------
# Standardize features + normalize target
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

y_reg_mean = float(y_reg_train.mean())
y_reg_std = float(y_reg_train.std() + 1e-8)
y_reg_train = (y_reg_train - y_reg_mean) / y_reg_std
y_reg_val = (y_reg_val - y_reg_mean) / y_reg_std


class SonarDataset(Dataset):
    def __init__(self, X_arr, y_reg_arr, y_side_arr):
        self.X = torch.tensor(X_arr, dtype=torch.float32)
        self.y_reg = torch.tensor(y_reg_arr, dtype=torch.float32)
        self.y_side = torch.tensor(y_side_arr, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_reg[idx], self.y_side[idx]


train_loader = DataLoader(
    SonarDataset(X_train, y_reg_train, side_train),
    batch_size=batch_size_train,
    shuffle=True,
)
val_loader = DataLoader(
    SonarDataset(X_val, y_reg_val, side_val),
    batch_size=batch_size_val,
    shuffle=False,
)

# -----------------------------
# Model
# -----------------------------
input_len = X_train.shape[1]


class TwinTowerNet(nn.Module):
    def __init__(self, in_len, drop):
        super().__init__()
        if in_len % 2 != 0:
            raise ValueError("Expected even feature length for left/right split.")
        self.half = in_len // 2
        self.tower = nn.Sequential(
            nn.Linear(self.half, 128),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(64 * 2, 128),
            nn.ReLU(),
            nn.Dropout(drop),
        )
        self.reg_head = nn.Linear(128, 1)
        self.side_head = nn.Linear(128, 3)

    def forward(self, x):
        left = x[:, : self.half]
        right = x[:, self.half :]
        z_left = self.tower(left)
        z_right = self.tower(right)
        z = torch.cat([z_left, z_right], dim=1)
        z = self.fuse(z)
        return self.reg_head(z).squeeze(1), self.side_head(z)


model = TwinTowerNet(input_len, drop_p).to(device)

side_counts = np.bincount(side_train, minlength=3).astype(np.float32)
side_weights = side_counts.sum() / np.maximum(side_counts, 1.0)
side_weights = torch.tensor(side_weights, dtype=torch.float32, device=device)

loss_reg = nn.SmoothL1Loss()
loss_side = nn.CrossEntropyLoss(weight=side_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -----------------------------
# Train
# -----------------------------
best_state = None
best_val = np.inf
patience_left = patience
train_hist = []
val_hist = []
val_dist_hist = []

for epoch in range(epochs):
    model.train()
    running = 0.0
    for xb, yb_reg, yb_side in train_loader:
        xb = xb.to(device)
        yb_reg = yb_reg.to(device)
        yb_side = yb_side.to(device)

        pred_reg, pred_side = model(xb)
        l_reg = loss_reg(pred_reg, yb_reg)
        l_side = loss_side(pred_side, yb_side)
        loss = l_reg + side_loss_weight * l_side

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running += loss.item() * xb.size(0)

    train_loss = running / len(train_loader.dataset)
    train_hist.append(train_loss)

    model.eval()
    running = 0.0
    running_dist = 0.0
    with torch.no_grad():
        for xb, yb_reg, yb_side in val_loader:
            xb = xb.to(device)
            yb_reg = yb_reg.to(device)
            yb_side = yb_side.to(device)
            pred_reg, pred_side = model(xb)
            l_reg = loss_reg(pred_reg, yb_reg)
            l_side = loss_side(pred_side, yb_side)
            loss = l_reg + side_loss_weight * l_side
            running += loss.item() * xb.size(0)
            running_dist += l_reg.item() * xb.size(0)

    val_loss = running / len(val_loader.dataset)
    val_dist = running_dist / len(val_loader.dataset)
    val_hist.append(val_loss)
    val_dist_hist.append(val_dist)

    if val_dist < best_val:
        best_val = val_dist
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        patience_left = patience
    else:
        patience_left -= 1
        if patience_left <= 0:
            print(f"Early stop at epoch {epoch + 1}")
            break

if best_state is not None:
    model.load_state_dict(best_state)

# -----------------------------
# Evaluate
# -----------------------------
model.eval()
with torch.no_grad():
    xb = torch.tensor(X_val, dtype=torch.float32, device=device)
    pred_reg_n, pred_side = model(xb)
    pred_reg_n = pred_reg_n.cpu().numpy()
    pred_side = pred_side.cpu().numpy()

pred_reg_log = pred_reg_n * y_reg_std + y_reg_mean
pred_dist = np.expm1(pred_reg_log)
true_dist = y_reg_raw_val
dist_mae = np.mean(np.abs(pred_dist - true_dist))
side_pred = np.argmax(pred_side, axis=1)
side_acc = np.mean(side_pred == side_val)

print(f"Val distance MAE: {dist_mae:.3f} m")
print(f"Val side accuracy: {side_acc:.3f}")

# -----------------------------
# Plots
# -----------------------------
plt.figure(figsize=(6, 4))
plt.plot(train_hist, label="train")
plt.plot(val_hist, label="val")
plt.plot(val_dist_hist, label="val_dist")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(plots_dir / "distance_side_twin_loss.png", dpi=150)

plt.figure(figsize=(5, 5))
plt.scatter(true_dist, pred_dist, s=8, alpha=0.5)
lims = [min(true_dist.min(), pred_dist.min()), max(true_dist.max(), pred_dist.max())]
plt.plot(lims, lims, "k--", lw=1)
if len(true_dist) > 1:
    rho = np.corrcoef(true_dist, pred_dist)[0, 1]
    plt.title(f"Pred vs True (r={rho:.2f})")
plt.xlabel("True closest distance (m)")
plt.ylabel("Pred closest distance (m)")
plt.tight_layout()
plt.savefig(plots_dir / "distance_side_twin_distance_scatter.png", dpi=150)

cm = confusion_matrix(side_val, side_pred, labels=[0, 1, 2])
plt.figure(figsize=(4, 4))
plt.imshow(cm, cmap="Blues")
plt.colorbar(label="Count")
plt.xticks([0, 1, 2], ["-1", "0", "+1"])
plt.yticks([0, 1, 2], ["-1", "0", "+1"])
plt.xlabel("Pred side")
plt.ylabel("True side")
plt.tight_layout()
plt.savefig(plots_dir / "distance_side_twin_confusion.png", dpi=150)

# -----------------------------
# Save model + preprocessing
# -----------------------------
checkpoint = {
    "model_state": model.state_dict(),
    "input_len": X_train.shape[1],
    "scaler_mean": scaler.mean_.astype(np.float32),
    "scaler_scale": scaler.scale_.astype(np.float32),
    "y_reg_mean": y_reg_mean,
    "y_reg_std": y_reg_std,
    "same_thresh_mm": same_thresh_mm,
}
torch.save(checkpoint, models_dir / "distance_side_twin.pt")
