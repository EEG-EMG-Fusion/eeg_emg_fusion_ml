"""
EEG Feature Baseline
-------------------------------------------------------
Muskaan Garg | H00416442 | Heriot-Watt University

Logistic regression baseline using 11 features × 15 channels = 165 features
derived from Khan et al. (2020) on the NeBULA ICA-cleaned EEG data.

Features (Khan et al., 2020, IEEE Reviews in Biomedical Engineering):
  Time-domain:
    1.  RMS               — signal energy
    2.  Standard deviation — amplitude variability
    3.  Maximum peak value — max absolute amplitude
    4.  Kurtosis           — signal peakedness
    5.  Waveform length    — signal path length
    6.  Hjorth Mobility    — signal dynamics
    7.  Hjorth Complexity  — signal complexity

  Frequency-domain (band power):
    8.  Delta  0.5–4  Hz  — slow cortical potentials (cited for reaching in [114])
    9.  Theta  4–8    Hz  — motor oscillations
    10. Alpha  8–12   Hz  — ERD marker (mu rhythm)
    11. Beta   13–30  Hz  — motor preparation ERD

Compares:
  - Original pipeline (X_eeg_win.npy)         + old 6-feature set
  - Cleaned pipeline  (X_eeg_win_clean.npy)   + old 6-feature set
  - Cleaned pipeline  (X_eeg_win_clean.npy)   + new 11-feature set (Khan 2020)
"""

import os
import numpy as np
from scipy import signal as sp_signal
from scipy.stats import kurtosis, skew
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, classification_report

# ── Config ───────────────────────────────────────────────────────────────────

DATA_DIR  = './preprocessed'
SEED      = 42
FS        = 200   # Hz after resampling

WINDOW_STARTS   = np.array([0,40,80,120,160,200,240,280,320,360,400], dtype=np.int64)
KEEP_EEG_STARTS = {40, 80, 120}   # onset-centred windows for EEG


# ── Feature extraction ────────────────────────────────────────────────────────

def hjorth_params(x):
    """
    Compute Hjorth Mobility and Complexity per channel.
    x: (N, C, T)
    Returns mobility (N, C), complexity (N, C)
    """
    dx  = np.diff(x, axis=2)           # first derivative
    ddx = np.diff(dx, axis=2)          # second derivative

    var_x   = np.var(x,   axis=2) + 1e-10
    var_dx  = np.var(dx,  axis=2) + 1e-10
    var_ddx = np.var(ddx, axis=2) + 1e-10

    mobility    = np.sqrt(var_dx  / var_x)
    mob_dx      = np.sqrt(var_ddx / var_dx)
    complexity  = mob_dx / mobility

    return mobility, complexity


def band_power(x, fs, low, high):
    """
    Bandpass filter and compute RMS power per channel.
    x: (N, C, T)
    Returns (N, C)
    """
    N, C, T = x.shape
    sos = sp_signal.butter(4, [low, high], btype='bandpass', fs=fs, output='sos')
    out = np.zeros((N, C), dtype=np.float32)
    for i in range(N):
        filtered = sp_signal.sosfiltfilt(sos, x[i], axis=1)
        out[i] = np.sqrt(np.mean(filtered**2, axis=1))
    return out


def compute_old_features(X, fs=FS):
    """
    Original 6-feature set: RMS, mean, std, WL, alpha, beta.
    Input:  (N, 15, 80)
    Output: (N, 90)
    """
    rms  = np.sqrt(np.mean(X**2, axis=2))
    mean = np.mean(X, axis=2)
    std  = np.std(X, axis=2)
    wl   = np.sum(np.abs(np.diff(X, axis=2)), axis=2)
    alph = band_power(X, fs, 8, 12)
    beta = band_power(X, fs, 13, 30)
    return np.concatenate([rms, mean, std, wl, alph, beta], axis=1).astype(np.float32)


def compute_khan_features(X, fs=FS):
    """
    Khan et al. (2020) feature set — 11 features × 15 channels = 165.

    Time-domain (7):
      RMS, std, max peak, kurtosis, waveform length,
      Hjorth Mobility, Hjorth Complexity

    Frequency-domain (4):
      Delta (0.5–4Hz), Theta (4–8Hz), Alpha (8–12Hz), Beta (13–30Hz)

    Input:  (N, 15, 80)
    Output: (N, 165)
    """
    N, C, T = X.shape

    # ── Time-domain ──────────────────────────────────────────────────────────

    # 1. RMS
    rms = np.sqrt(np.mean(X**2, axis=2))           # (N, C)

    # 2. Standard deviation
    std = np.std(X, axis=2)                          # (N, C)

    # 3. Maximum peak value (max absolute amplitude)
    peak = np.max(np.abs(X), axis=2)                 # (N, C)

    # 4. Kurtosis — signal peakedness
    # scipy kurtosis operates on last axis by default
    kurt = np.zeros((N, C), dtype=np.float32)
    for i in range(N):
        kurt[i] = kurtosis(X[i], axis=1, fisher=True)

    # 5. Waveform length
    wl = np.sum(np.abs(np.diff(X, axis=2)), axis=2)  # (N, C)

    # 6 & 7. Hjorth Mobility and Complexity
    mob, comp = hjorth_params(X)                      # (N, C) each

    # ── Frequency-domain ─────────────────────────────────────────────────────

    # 8. Delta 0.5–4 Hz — slow cortical potentials (cited for reaching in Khan et al. 2020 [114])
    delta = band_power(X, fs, 0.5, 4)                # (N, C)

    # 9. Theta 4–8 Hz — motor oscillations
    theta = band_power(X, fs, 4, 8)                  # (N, C)

    # 10. Alpha 8–12 Hz — ERD marker
    alpha = band_power(X, fs, 8, 12)                 # (N, C)

    # 11. Beta 13–30 Hz — motor preparation ERD
    beta = band_power(X, fs, 13, 30)                 # (N, C)

    feats = np.concatenate(
        [rms, std, peak, kurt, wl, mob, comp,
         delta, theta, alpha, beta],
        axis=1
    ).astype(np.float32)

    return feats   # (N, 165)


# ── Data loading and splitting ────────────────────────────────────────────────

def load_and_filter(x_file, y_file, s_file, t_file):
    """Load arrays and filter to onset windows."""
    X = np.load(os.path.join(DATA_DIR, x_file)).astype(np.float32)
    y = np.load(os.path.join(DATA_DIR, y_file)).astype(np.int64)
    s = np.load(os.path.join(DATA_DIR, s_file)).astype(np.int64)
    t = np.load(os.path.join(DATA_DIR, t_file)).astype(np.int64)

    if y.min() == 1:
        y = y - 1

    pos        = np.arange(len(X)) % len(WINDOW_STARTS)
    win_starts = WINDOW_STARTS[pos]
    mask       = np.isin(win_starts, list(KEEP_EEG_STARTS))

    return X[mask], y[mask], s[mask], t[mask]


def subject_split(s):
    """Same 25/5/7 split as all models — seed=42."""
    np.random.seed(SEED)
    subjects = np.unique(s)
    np.random.shuffle(subjects)
    train = set(subjects[:25])
    val   = set(subjects[25:30])
    test  = set(subjects[30:])
    return train, val, test


def run_lr(F_tr, F_te, y_tr, y_te, label):
    sc = StandardScaler()
    lr = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
    lr.fit(sc.fit_transform(F_tr), y_tr)
    pred = lr.predict(sc.transform(F_te))
    f1   = f1_score(y_te, pred, average='macro', zero_division=0)
    acc  = accuracy_score(y_te, pred)
    print(f'\n  {label}')
    print(f'  Accuracy : {acc*100:.1f}%  |  F1 macro : {f1:.4f}')
    print(classification_report(y_te, pred,
          target_names=['Task1','Task2','Task3'], zero_division=0))
    return f1, acc


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print('=' * 65)
    print('  EEG Feature Baseline — Khan et al. (2020)')
    print('  Muskaan Garg | H00416442 | Heriot-Watt University')
    print('=' * 65)

    # ── Load original pipeline data ───────────────────────────────────────────
    print('\n  Loading original pipeline data...')
    X_orig, y_orig, s_orig, _ = load_and_filter(
        'X_eeg_win.npy', 'y_win.npy',
        'subject_ids_win.npy', 'trial_ids_win.npy'
    )
    print(f'  Original: {X_orig.shape}  labels: {np.bincount(y_orig)}')

    # ── Load cleaned pipeline data ────────────────────────────────────────────
    print('  Loading cleaned pipeline data...')
    X_clean, y_clean, s_clean, _ = load_and_filter(
        'X_eeg_win_clean.npy', 'y_win_clean.npy',
        'subject_ids_win_clean.npy', 'trial_ids_win_clean.npy'
    )
    print(f'  Cleaned:  {X_clean.shape}  labels: {np.bincount(y_clean)}')

    # ── Compute features ──────────────────────────────────────────────────────
    print('\n  Computing features...')
    F_orig_old   = compute_old_features(X_orig)
    F_clean_old  = compute_old_features(X_clean)
    F_clean_khan = compute_khan_features(X_clean)
    print(f'  Original + old features : {F_orig_old.shape}')
    print(f'  Cleaned  + old features : {F_clean_old.shape}')
    print(f'  Cleaned  + Khan features: {F_clean_khan.shape}')

    # ── Splits ────────────────────────────────────────────────────────────────
    train_s, _, test_s = subject_split(s_orig)
    tr_o = np.array([sid in train_s for sid in s_orig])
    te_o = np.array([sid in test_s  for sid in s_orig])

    train_s2, _, test_s2 = subject_split(s_clean)
    tr_c = np.array([sid in train_s2 for sid in s_clean])
    te_c = np.array([sid in test_s2  for sid in s_clean])

    print(f'\n  Original test subjects : {sorted([int(x) for x in test_s])}')
    print(f'  Cleaned  test subjects : {sorted([int(x) for x in test_s2])}')

    # ── Run comparisons ───────────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('  RESULTS')
    print('=' * 65)

    run_lr(F_orig_old[tr_o],   F_orig_old[te_o],
           y_orig[tr_o],       y_orig[te_o],
           '1. Original pipeline + old features (6×15=90)')

    run_lr(F_clean_old[tr_c],  F_clean_old[te_c],
           y_clean[tr_c],      y_clean[te_c],
           '2. Cleaned pipeline  + old features (6×15=90)')

    run_lr(F_clean_khan[tr_c], F_clean_khan[te_c],
           y_clean[tr_c],      y_clean[te_c],
           '3. Cleaned pipeline  + Khan features (11×15=165)')

    print('=' * 65)
    print('  Chance baseline: 33.3%  F1=0.333')
    print('  Original deep learning baseline (EEGNet): 33.9%  F1=0.305')
    print('  Original deep learning baseline (EEG CNN-LSTM): 35.2%  F1=0.337')
    print('=' * 65)


if __name__ == '__main__':
    main()