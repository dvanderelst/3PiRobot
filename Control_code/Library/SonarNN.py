import numpy as np
import torch
import torch.nn as nn


class TwinTowerNet(nn.Module):
    def __init__(self, in_len, drop_p=0.0):
        super().__init__()
        if in_len % 2 != 0:
            raise ValueError("Expected even feature length for left/right split.")
        self.half = in_len // 2
        self.tower = nn.Sequential(
            nn.Linear(self.half, 128),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(64 * 2, 128),
            nn.ReLU(),
            nn.Dropout(drop_p),
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


def load_distance_side_model(checkpoint_path, device=None):
    if device is None:
        device = torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = TwinTowerNet(ckpt["input_len"], drop_p=0.0).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    meta = {
        "scaler_mean": ckpt["scaler_mean"],
        "scaler_scale": ckpt["scaler_scale"],
        "y_reg_mean": ckpt["y_reg_mean"],
        "y_reg_std": ckpt["y_reg_std"],
        "same_thresh_mm": ckpt.get("same_thresh_mm", None),
    }
    return model, meta


def processNN(left_sonar, right_sonar, model, meta, device=None):
    if device is None:
        device = torch.device("cpu")
    left = np.asarray(left_sonar, dtype=np.float32).ravel()
    right = np.asarray(right_sonar, dtype=np.float32).ravel()
    x = np.concatenate([left, right], axis=0)[None, :]
    if x.shape[1] != len(meta["scaler_mean"]):
        raise ValueError("Input length does not match model scaler length.")
    x = (x - meta["scaler_mean"]) / meta["scaler_scale"]
    xb = torch.tensor(x, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred_reg_n, pred_side = model(xb)
    pred_reg_n = pred_reg_n.cpu().numpy()[0]
    pred_side = pred_side.cpu().numpy()[0]
    pred_reg_log = pred_reg_n * meta["y_reg_std"] + meta["y_reg_mean"]
    pred_dist = float(np.expm1(pred_reg_log))
    side_idx = int(np.argmax(pred_side))
    side_sign = side_idx - 1
    return pred_dist, side_sign
