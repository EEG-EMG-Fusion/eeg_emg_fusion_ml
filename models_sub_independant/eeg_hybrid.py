"""
EEG Hybrid Experiment — Raw signal + Handcrafted features
----------------------------------------------------------
NeBULA Dataset | Muskaan Garg | H00416442

Tests whether adding handcrafted EEG features (RMS, mean, std,
waveform length, alpha power, beta power) alongside the raw signal
CNN can improve performance beyond the raw-only models.

Three models are evaluated:
  1. LR baseline      — 90 handcrafted features only, no deep learning
  2. EEGNet + features — EEGNet raw branch + feature MLP branch
  3. CNN-LSTM + features — EEG CNN-LSTM raw branch + feature MLP branch


"""

import os
import json
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy import signal as sp_signal
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score,
    confusion_matrix, classification_report
)

# ── Config ──────────────────────────────────────────────────────────────────

DATA_DIR    = "../preprocessed"
RESULTS_DIR = "./results/eeg_hybrid"

SEED        = 42
BATCH_SIZE  = 64
EPOCHS      = 160
LR          = 1e-4
DROPOUT     = 0.30
PATIENCE    = 30
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05

N_CHANNELS  = 15
N_CLASSES   = 3
FS          = 200  # sampling rate

WINDOW_STARTS    = np.array([0,40,80,120,160,200,240,280,320,360,400], dtype=np.int64)
KEEP_WIN_STARTS  = {40, 80, 120}


# ── Seeds & device ───────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available():    return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


# ── Feature extraction ───────────────────────────────────────────────────────

def compute_eeg_features(X_win: np.ndarray, fs: int = FS) -> np.ndarray:
    """
    Compute 6 handcrafted features per channel per window.

    Input:  X_win  (N, 15, 80)
    Output: feats  (N, 90)   — 6 features × 15 channels

    Features:
      - RMS          — overall activation energy
      - Mean         — average amplitude
      - Std          — amplitude variability
      - Waveform length  — signal complexity
      - Alpha power  (8–12 Hz) — motor-related cortical rhythms
      - Beta power   (13–30 Hz) — motor preparation marker
    """
    N, C, T = X_win.shape

    rms  = np.sqrt(np.mean(X_win**2, axis=2))           # (N, C)
    mean = np.mean(X_win, axis=2)                         # (N, C)
    std  = np.std(X_win, axis=2)                          # (N, C)
    wl   = np.sum(np.abs(np.diff(X_win, axis=2)), axis=2) # (N, C)

    sos_alpha = sp_signal.butter(4, [8, 12],  btype='bandpass', fs=fs, output='sos')
    sos_beta  = sp_signal.butter(4, [13, 30], btype='bandpass', fs=fs, output='sos')

    alpha_pow = np.zeros((N, C), dtype=np.float32)
    beta_pow  = np.zeros((N, C), dtype=np.float32)

    for i in range(N):
        a = sp_signal.sosfiltfilt(sos_alpha, X_win[i], axis=1)
        b = sp_signal.sosfiltfilt(sos_beta,  X_win[i], axis=1)
        alpha_pow[i] = np.sqrt(np.mean(a**2, axis=1))
        beta_pow[i]  = np.sqrt(np.mean(b**2, axis=1))

    return np.concatenate([rms, mean, std, wl, alpha_pow, beta_pow], axis=1).astype(np.float32)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_and_prepare():
    X = np.load(os.path.join(DATA_DIR, "X_eeg_win.npy")).astype(np.float32)
    y = np.load(os.path.join(DATA_DIR, "y_win.npy")).astype(np.int64)
    s = np.load(os.path.join(DATA_DIR, "subject_ids_win.npy")).astype(np.int64)

    # Convert 1-indexed labels to 0-indexed
    if y.min() == 1:
        y = y - 1

    # Filter to onset-centred windows only
    pos = np.arange(len(X)) % len(WINDOW_STARTS)
    win_starts = WINDOW_STARTS[pos]
    mask = np.isin(win_starts, list(KEEP_WIN_STARTS))
    X = X[mask]; y = y[mask]; s = s[mask]

    print(f"  Loaded: {X.shape}  labels: {np.bincount(y)}")

    print("  Computing handcrafted features...")
    F = compute_eeg_features(X)
    print(f"  Feature shape: {F.shape}")

    return X, F, y, s


def subject_split(s):
    """Same 25/5/7 split as all other models — seed=42."""
    subjects = np.unique(s)
    np.random.shuffle(subjects)
    train = set(subjects[:25])
    val   = set(subjects[25:30])
    test  = set(subjects[30:])
    return train, val, test


# ── Dataset ──────────────────────────────────────────────────────────────────

class EEGHybridDataset(Dataset):
    """Holds both raw signal and handcrafted features."""
    def __init__(self, X_raw: np.ndarray, X_feat: np.ndarray, y: np.ndarray):
        self.X_raw  = torch.tensor(X_raw,  dtype=torch.float32)
        self.X_feat = torch.tensor(X_feat, dtype=torch.float32)
        self.y      = torch.tensor(y,      dtype=torch.long)

    def __len__(self):  return len(self.y)

    def __getitem__(self, idx):
        return self.X_raw[idx], self.X_feat[idx], self.y[idx]


# ── Models ───────────────────────────────────────────────────────────────────

class FeatureBranch(nn.Module):
    """Small MLP that processes handcrafted EEG features."""
    def __init__(self, in_dim: int, out_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, out_dim),
            nn.ELU(),
            nn.Dropout(DROPOUT),
        )
    def forward(self, x): return self.mlp(x)


class AttentionPool(nn.Module):
    """Soft attention pooling over LSTM timesteps."""
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)
    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=1)
        return torch.sum(w * x, dim=1)


class EEGNetHybrid(nn.Module):
    """
    EEGNet raw signal branch + handcrafted feature branch.

    Raw branch:  EEGNet (F1=16, D=2, F2=32) → 160-dim
    Feat branch: MLP(90 → 32)
    Combined:    concat → FC(192, 64) → FC(64, 3)

    Input:  x_raw  (B, 1, 15, 80)
            x_feat (B, 90)
    Output: (B, 3)
    """
    def __init__(self, n_channels=15, win_size=80, feat_dim=90,
                 F1=16, D=2, F2=32):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, 32), padding=(0, 16), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1*D, kernel_size=(n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1*D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(DROPOUT),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1*D, F2, kernel_size=(1, 8), padding=(0, 4), groups=F1*D, bias=False),
            nn.Conv2d(F2, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(DROPOUT),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, win_size)
            raw_dim = self.block2(self.block1(dummy)).view(1, -1).shape[1]

        self.feat_branch = FeatureBranch(feat_dim, 32)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(raw_dim + 32, 64),
            nn.ELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, N_CLASSES),
        )

    def forward(self, x_raw, x_feat):
        # raw branch — x_raw comes in as (B, 1, 15, 80) from dataset
        z_raw  = self.block2(self.block1(x_raw)).view(x_raw.size(0), -1)
        z_feat = self.feat_branch(x_feat)
        return self.classifier(torch.cat([z_raw, z_feat], dim=1))


class EEGCnnLstmHybrid(nn.Module):
    """
    EEG CNN-LSTM raw signal branch + handcrafted feature branch.

    Raw branch:  CNN → BiLSTM → attention → 128-dim
    Feat branch: MLP(90 → 32)
    Combined:    concat → FC(160, 64) → FC(64, 3)

    Input:  x_raw  (B, 15, 80)
            x_feat (B, 90)
    Output: (B, 3)
    """
    def __init__(self, feat_dim=90):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 7), padding=(0, 3), bias=False),
            nn.BatchNorm2d(32), nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=(N_CHANNELS, 1), bias=False),
            nn.BatchNorm2d(32), nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 2)),
            nn.Dropout(DROPOUT),
            nn.Conv2d(32, 64, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(64), nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 2)),
            nn.Dropout(DROPOUT),
        )

        self.lstm = nn.LSTM(input_size=64, hidden_size=64,
                            batch_first=True, bidirectional=True)
        self.attn = AttentionPool(128)
        self.feat_branch = FeatureBranch(feat_dim, 32)

        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, N_CLASSES),
        )

    def forward(self, x_raw, x_feat):
        x = x_raw.unsqueeze(1)       # (B, 1, 15, 80)
        x = self.cnn(x)              # (B, 64, 1, 20)
        x = x.squeeze(2).transpose(1, 2)  # (B, 20, 64)
        x, _ = self.lstm(x)          # (B, 20, 128)
        z_raw = self.attn(x)         # (B, 128)
        z_feat = self.feat_branch(x_feat)  # (B, 32)
        return self.classifier(torch.cat([z_raw, z_feat], dim=1))


# ── Training loop ─────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    preds, trues = [], []

    for x_raw, x_feat, yb in loader:
        x_raw  = x_raw.to(device)
        x_feat = x_feat.to(device)
        yb     = yb.to(device)

        if is_train:
            optimizer.zero_grad()

        out  = model(x_raw, x_feat)
        loss = criterion(out, yb)

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * len(yb)
        preds.extend(out.argmax(1).detach().cpu().tolist())
        trues.extend(yb.detach().cpu().tolist())

    n = len(loader.dataset)
    return (total_loss / n,
            accuracy_score(trues, preds),
            f1_score(trues, preds, average="macro", zero_division=0),
            trues, preds)


def train_model(model, train_loader, val_loader, test_loader,
                model_name, device, results_dir):
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8)

    best_val_f1   = -1.0
    best_val_loss = float("inf")
    best_state    = None
    best_epoch    = 0
    no_improve    = 0
    history       = []

    print(f"\n{'='*65}")
    print(f"  Training: {model_name}")
    print(f"{'='*65}")

    for epoch in range(1, EPOCHS + 1):
        tl, ta, tf, _, _ = run_epoch(model, train_loader, criterion, optimizer, device)
        vl, va, vf, _, _ = run_epoch(model, val_loader,   criterion, None,      device)
        scheduler.step(vf)

        history.append({"epoch": epoch,
                        "train_loss": tl, "train_acc": ta, "train_f1": tf,
                        "val_loss":   vl, "val_acc":   va, "val_f1":   vf})

        print(f"  Epoch {epoch:03d} | "
              f"train loss={tl:.4f} acc={ta:.3f} f1={tf:.3f} | "
              f"val loss={vl:.4f} acc={va:.3f} f1={vf:.3f}")

        improved = (vf > best_val_f1) or (vf == best_val_f1 and vl < best_val_loss)
        if improved:
            best_val_f1   = vf
            best_val_loss = vl
            best_state    = copy.deepcopy(model.state_dict())
            best_epoch    = epoch
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
                break

    model.load_state_dict(best_state)
    _, test_acc, test_f1, y_true, y_pred = run_epoch(
        model, test_loader, criterion, None, device)

    cm     = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    print(f"\n{'='*65}")
    print(f"  {model_name} — TEST RESULTS")
    print(f"  Best epoch : {best_epoch}")
    print(f"  Accuracy   : {test_acc*100:.1f}%")
    print(f"  F1 macro   : {test_f1:.4f}")
    print(f"  Per class  : "
          f"Task1={report['0']['f1-score']:.3f}  "
          f"Task2={report['1']['f1-score']:.3f}  "
          f"Task3={report['2']['f1-score']:.3f}")
    print("  Confusion matrix:")
    print(f"    {cm[0].tolist()}  ← Task 1")
    print(f"    {cm[1].tolist()}  ← Task 2")
    print(f"    {cm[2].tolist()}  ← Task 3")
    print(f"{'='*65}")

    # Save
    prefix = os.path.join(results_dir, model_name.lower().replace(" ", "_"))
    torch.save(model.state_dict(), prefix + ".pt")
    np.save(prefix + "_history.npy", np.array(history, dtype=object))
    with open(prefix + "_summary.json", "w") as f:
        json.dump({
            "model": model_name, "best_epoch": best_epoch,
            "test_accuracy": float(test_acc), "test_f1_macro": float(test_f1),
            "confusion_matrix": cm.tolist(), "classification_report": report,
        }, f, indent=2)

    return test_acc, test_f1


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = get_device()

    print("=" * 65)
    print("  EEG Hybrid Experiment — Raw + Handcrafted Features")
    print("  Muskaan Garg | H00416442 | Heriot-Watt University")
    print("=" * 65)
    print(f"  Device: {device}")

    X, F, y, s = load_and_prepare()
    feat_dim = F.shape[1]  # 90

    train_s, val_s, test_s = subject_split(s)
    print(f"  Test subjects: {sorted([int(x) for x in test_s])}")

    train_m = np.array([sid in train_s for sid in s])
    val_m   = np.array([sid in val_s   for sid in s])
    test_m  = np.array([sid in test_s  for sid in s])

    # ── Part 1: Logistic regression baseline ─────────────────────────────────
    print("\n" + "="*65)
    print("  PART 1 — Logistic Regression Baseline (features only)")
    print("="*65)

    scaler = StandardScaler()
    F_train = scaler.fit_transform(F[train_m])
    F_val   = scaler.transform(F[val_m])
    F_test  = scaler.transform(F[test_m])

    lr_clf = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
    lr_clf.fit(F_train, y[train_m])
    lr_pred = lr_clf.predict(F_test)
    lr_f1   = f1_score(y[test_m], lr_pred, average="macro", zero_division=0)
    lr_acc  = accuracy_score(y[test_m], lr_pred)
    print(f"  Accuracy : {lr_acc*100:.1f}%")
    print(f"  F1 macro : {lr_f1:.4f}")
    print(classification_report(y[test_m], lr_pred,
          target_names=["Task1","Task2","Task3"], zero_division=0))

    # ── Build dataloaders for neural models ─────────────────────────────────
    # EEGNet expects (B, 1, 15, 80) so we unsqueeze in dataset
    class EEGNetDataset(Dataset):
        def __init__(self, X_raw, X_feat, y):
            self.X_raw  = torch.tensor(X_raw, dtype=torch.float32).unsqueeze(1)
            self.X_feat = torch.tensor(X_feat, dtype=torch.float32)
            self.y      = torch.tensor(y, dtype=torch.long)
        def __len__(self): return len(self.y)
        def __getitem__(self, i): return self.X_raw[i], self.X_feat[i], self.y[i]

    # CNN-LSTM expects (B, 15, 80)
    def make_loaders_cnnlstm(X_sub, F_sub, y_sub, shuffle):
        ds = EEGHybridDataset(X_sub, F_sub, y_sub)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    def make_loaders_eegnet(X_sub, F_sub, y_sub, shuffle):
        ds = EEGNetDataset(X_sub, F_sub, y_sub)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    # ── Part 2: EEGNet + features ────────────────────────────────────────────
    print("\n" + "="*65)
    print("  PART 2 — EEGNet + Feature Branch")
    print("="*65)

    model_eegnet = EEGNetHybrid(feat_dim=feat_dim).to(device)
    n_params = sum(p.numel() for p in model_eegnet.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    tr_eegnet = make_loaders_eegnet(X[train_m], F[train_m], y[train_m], True)
    vl_eegnet = make_loaders_eegnet(X[val_m],   F[val_m],   y[val_m],   False)
    te_eegnet = make_loaders_eegnet(X[test_m],  F[test_m],  y[test_m],  False)

    train_model(model_eegnet, tr_eegnet, vl_eegnet, te_eegnet,
                "EEGNet_Hybrid", device, RESULTS_DIR)

    # ── Part 3: EEG CNN-LSTM + features ──────────────────────────────────────
    print("\n" + "="*65)
    print("  PART 3 — EEG CNN-LSTM + Feature Branch")
    print("="*65)

    model_cnnlstm = EEGCnnLstmHybrid(feat_dim=feat_dim).to(device)
    n_params = sum(p.numel() for p in model_cnnlstm.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    tr_cnn = make_loaders_cnnlstm(X[train_m], F[train_m], y[train_m], True)
    vl_cnn = make_loaders_cnnlstm(X[val_m],   F[val_m],   y[val_m],   False)
    te_cnn = make_loaders_cnnlstm(X[test_m],  F[test_m],  y[test_m],  False)

    train_model(model_cnnlstm, tr_cnn, vl_cnn, te_cnn,
                "EEGCNNLSTM_Hybrid", device, RESULTS_DIR)

    # ── Final comparison ─────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  COMPARISON SUMMARY")
    print("  Baseline (EEGNet)           : 33.9%  F1=0.305")
    print("  Baseline (EEG CNN-LSTM)     : 35.2%  F1=0.337")
    print(f"  LR + features               : {lr_acc*100:.1f}%  F1={lr_f1:.3f}")
    print("  EEGNet + features           : see above")
    print("  EEG CNN-LSTM + features     : see above")
    print("  Chance                      : 33.3%  F1=0.333")
    print("="*65)


if __name__ == "__main__":
    main()