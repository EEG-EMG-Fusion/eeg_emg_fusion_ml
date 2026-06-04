"""
NeBULA Dataset - EEG CNN-LSTM (Unified Architecture) — Cleaned Data
--------------------------------------------------------------------
Muskaan Garg | H00416442 | Heriot-Watt University

EEG classification using the same Conv1d + BiLSTM + attention architecture
as the EMG CNN-LSTM, applied to ICA-cleaned EEG data.

Architecture is identical to EMG CNN-LSTM for fair modality comparison:
  Raw branch:  Conv1d(k=9) → Conv1d(k=7) → BiLSTM(64) → Attention → (B, 128)
  Feat branch: MLP(75 → 64 → 32)
  Classifier:  FC(160 → 64) → FC(64 → 3)

Features (5 × 15 channels = 75):
  RMS, std, waveform length, alpha power (8–12Hz), beta power (13–30Hz)
  Mean dropped — confirmed useless after z-scoring.

"""

import os
import json
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy import signal as sp_signal
from sklearn.metrics import (
    accuracy_score, f1_score,
    confusion_matrix, classification_report
)

# ── Config ───────────────────────────────────────────────────────────────────

DATA_DIR    = "../preprocessed"
RESULTS_DIR = "./results/eeg_cnn_lstm_clean"

SEED            = 42
BATCH_SIZE      = 64
EPOCHS          = 160
LR              = 1e-4
DROPOUT         = 0.30
PATIENCE        = 30
WEIGHT_DECAY    = 1e-4
LABEL_SMOOTHING = 0.05

N_CHANNELS = 15
N_CLASSES  = 3
FS         = 200   # Hz

# Same window recovery as all other models
WINDOW_STARTS    = np.array([0,40,80,120,160,200,240,280,320,360,400], dtype=np.int64)
KEEP_WIN_STARTS  = {40, 80, 120}   # onset windows for EEG


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available():         return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


# ── Feature extraction ────────────────────────────────────────────────────────

def compute_eeg_features(X, fs=FS):
    """
    5 features × 15 channels = 75 per window.

    Features:
      RMS, std, waveform length, alpha power (8–12Hz), beta power (13–30Hz)

    Mean excluded — useless after z-scoring (LR ablation: 33.3% = chance).
    Khan features excluded — hurt LR performance below chance on clean data.

    Input:  (N, 15, 80)
    Output: (N, 75)
    """
    rms = np.sqrt(np.mean(X**2, axis=2))                     # (N, 15)
    std = np.std(X, axis=2)                                   # (N, 15)
    wl  = np.sum(np.abs(np.diff(X, axis=2)), axis=2)         # (N, 15)

    sos_alpha = sp_signal.butter(4, [8, 12],  btype='bandpass', fs=fs, output='sos')
    sos_beta  = sp_signal.butter(4, [13, 30], btype='bandpass', fs=fs, output='sos')

    N, C = X.shape[:2]
    alpha = np.zeros((N, C), dtype=np.float32)
    beta  = np.zeros((N, C), dtype=np.float32)
    for i in range(N):
        alpha[i] = np.sqrt(np.mean(sp_signal.sosfiltfilt(sos_alpha, X[i], axis=1)**2, axis=1))
        beta[i]  = np.sqrt(np.mean(sp_signal.sosfiltfilt(sos_beta,  X[i], axis=1)**2, axis=1))

    return np.concatenate([rms, std, wl, alpha, beta], axis=1).astype(np.float32)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    X = np.load(os.path.join(DATA_DIR, "X_eeg_win_clean.npy")).astype(np.float32)
    y = np.load(os.path.join(DATA_DIR, "y_win_clean.npy")).astype(np.int64)
    s = np.load(os.path.join(DATA_DIR, "subject_ids_win_clean.npy")).astype(np.int64)

    if y.min() == 1:
        y = y - 1

    # filter to onset windows
    pos        = np.arange(len(X)) % len(WINDOW_STARTS)
    win_starts = WINDOW_STARTS[pos]
    mask       = np.isin(win_starts, list(KEEP_WIN_STARTS))
    X = X[mask]; y = y[mask]; s = s[mask]

    print(f"  Loaded: {X.shape}  labels: {np.bincount(y)}")

    print("  Computing features...")
    F = compute_eeg_features(X)
    print(f"  Feature shape: {F.shape}")

    return X, F, y, s


def subject_split(s):
    """Same 25/5/7 split — seed=42."""
    subjects = np.unique(s)
    np.random.shuffle(subjects)
    return set(subjects[:25]), set(subjects[25:30]), set(subjects[30:])


# ── Dataset ───────────────────────────────────────────────────────────────────

class EEGDataset(Dataset):
    def __init__(self, X_raw, X_feat, y):
        self.X_raw  = torch.tensor(X_raw,  dtype=torch.float32)
        self.X_feat = torch.tensor(X_feat, dtype=torch.float32)
        self.y      = torch.tensor(y,      dtype=torch.long)

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        return self.X_raw[i], self.X_feat[i], self.y[i]


def make_loaders(X, F, y, s):
    train_s, val_s, test_s = subject_split(s)

    def get_loader(subset, shuffle):
        mask = np.array([sid in subset for sid in s])
        ds   = EEGDataset(X[mask], F[mask], y[mask])
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    return (
        get_loader(train_s, True),
        get_loader(val_s,   False),
        get_loader(test_s,  False),
        train_s, val_s, test_s,
    )


# ── Model — identical architecture to EMG CNN-LSTM ────────────────────────────

class AttentionPool(nn.Module):
    """Soft attention over BiLSTM timesteps — same as EMG CNN-LSTM."""
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=1)
        return torch.sum(w * x, dim=1)


class RawSignalBranch(nn.Module):
    """
    Conv1d + BiLSTM + Attention — identical structure to EMG CNN-LSTM.
    Input:  (B, 15, 80)
    Output: (B, 128)
    """
    def __init__(self):
        super().__init__()

        # same kernel sizes as EMG: 9, 7
        self.cnn = nn.Sequential(
            nn.Conv1d(N_CHANNELS, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.MaxPool1d(2),            # 80 → 40

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.MaxPool1d(2),            # 40 → 20

            nn.Dropout(DROPOUT),
        )

        # same BiLSTM as EMG: hidden=64, bidirectional → output 128
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            batch_first=True,
            bidirectional=True,
        )

        self.attn = AttentionPool(128)

    def forward(self, x):
        x = self.cnn(x)          # (B, 64, 20)
        x = x.transpose(1, 2)    # (B, 20, 64)
        x, _ = self.lstm(x)      # (B, 20, 128)
        return self.attn(x)      # (B, 128)


class FeatureBranch(nn.Module):
    """
    Handcrafted feature MLP — same structure as EMG CNN-LSTM.
    Input:  (B, feat_dim)
    Output: (B, 32)
    """
    def __init__(self, in_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.mlp(x)


class EEGCnnLstmClean(nn.Module):
    """
    EEG CNN-LSTM with unified Conv1d architecture matching EMG CNN-LSTM.

    Raw branch:  Conv1d(15→32,k=9) → Conv1d(32→64,k=7) → BiLSTM(64) → Attn → (B,128)
    Feat branch: MLP(75→64→32) → (B, 32)
    Classifier:  FC(160→64) → FC(64→3)

    Input:  x_raw  (B, 15, 80)
            x_feat (B, 75)
    Output: (B, 3)
    """
    def __init__(self, feat_dim=75):
        super().__init__()
        self.raw_branch  = RawSignalBranch()
        self.feat_branch = FeatureBranch(feat_dim)

        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, N_CLASSES),
        )

    def forward(self, x_raw, x_feat):
        z_raw  = self.raw_branch(x_raw)       # (B, 128)
        z_feat = self.feat_branch(x_feat)     # (B, 32)
        return self.classifier(torch.cat([z_raw, z_feat], dim=1))


# ── Training ──────────────────────────────────────────────────────────────────

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


def train():
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = get_device()

    X, F, y, s = load_data()
    feat_dim    = F.shape[1]   # 75

    train_loader, val_loader, test_loader, train_s, val_s, test_s = make_loaders(X, F, y, s)

    model   = EEGCnnLstmClean(feat_dim=feat_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print()
    print('-' * 65)
    print('  EEG CNN-LSTM (Unified Arch) — Cleaned Data')
    print('-' * 65)
    print(f'  Device         : {device}')
    print(f'  Input shape    : {X.shape}')
    print(f'  Feature shape  : {F.shape}')
    print(f'  Train subjects : {sorted([int(x) for x in train_s])}')
    print(f'  Val subjects   : {sorted([int(x) for x in val_s])}')
    print(f'  Test subjects  : {sorted([int(x) for x in test_s])}')
    print(f'  Parameters     : {n_params:,}')
    print()

    def split_stats(loader, name):
        ys = loader.dataset.y.numpy()
        print(f'  {name:5s}: {len(ys):5d} windows  '
              f'(T1={(ys==0).sum()}, T2={(ys==1).sum()}, T3={(ys==2).sum()})')

    split_stats(train_loader, 'train')
    split_stats(val_loader,   'val')
    split_stats(test_loader,  'test')
    print()

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8)

    best_val_f1   = -1.0
    best_val_loss = float('inf')
    best_state    = None
    best_epoch    = 0
    no_improve    = 0
    history       = []

    for epoch in range(1, EPOCHS + 1):
        tl, ta, tf, _, _ = run_epoch(model, train_loader, criterion, optimizer, device)
        vl, va, vf, _, _ = run_epoch(model, val_loader,   criterion, None,      device)
        scheduler.step(vf)

        history.append({
            'epoch': epoch,
            'train_loss': tl, 'train_acc': ta, 'train_f1': tf,
            'val_loss':   vl, 'val_acc':   va, 'val_f1':   vf,
        })

        print(f'  Epoch {epoch:03d} | '
              f'train loss={tl:.4f} acc={ta:.3f} f1={tf:.3f} | '
              f'val loss={vl:.4f} acc={va:.3f} f1={vf:.3f}')

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
                print(f'\n  Early stopping at epoch {epoch}. Best epoch: {best_epoch}')
                break

    model.load_state_dict(best_state)
    _, test_acc, test_f1, y_true, y_pred = run_epoch(
        model, test_loader, criterion, None, device)

    cm     = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    print('\n' + '=' * 65)
    print('  TEST RESULTS')
    print(f'  Best epoch : {best_epoch}')
    print(f'  Accuracy   : {test_acc*100:.1f}%')
    print(f'  F1 macro   : {test_f1:.4f}')
    print(f'  Per class  : '
          f'Task1={report["0"]["f1-score"]:.3f}  '
          f'Task2={report["1"]["f1-score"]:.3f}  '
          f'Task3={report["2"]["f1-score"]:.3f}')
    print('  Confusion matrix:')
    print(f'    {cm[0].tolist()}  ← Task 1')
    print(f'    {cm[1].tolist()}  ← Task 2')
    print(f'    {cm[2].tolist()}  ← Task 3')
    print('=' * 65)

    # save
    prefix = os.path.join(RESULTS_DIR, 'eeg_cnn_lstm_clean')
    torch.save(model.state_dict(), prefix + '.pt')
    np.save(prefix + '_history.npy', np.array(history, dtype=object))
    with open(prefix + '_summary.json', 'w') as f:
        json.dump({
            'model': 'EEGCnnLstmClean',
            'architecture': 'Conv1d+BiLSTM+Attention (unified with EMG)',
            'data': 'ICA-cleaned NeBULA derivatives',
            'features': '5 features x 15 channels = 75 (RMS, std, WL, alpha, beta)',
            'best_epoch': best_epoch,
            'best_val_f1': float(best_val_f1),
            'test_accuracy': float(test_acc),
            'test_f1_macro': float(test_f1),
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'train_subjects': sorted([int(x) for x in train_s]),
            'val_subjects':   sorted([int(x) for x in val_s]),
            'test_subjects':  sorted([int(x) for x in test_s]),
        }, f, indent=2)

    print(f'\n  Model   → {prefix}.pt')
    print(f'  Summary → {prefix}_summary.json')


if __name__ == '__main__':
    train()