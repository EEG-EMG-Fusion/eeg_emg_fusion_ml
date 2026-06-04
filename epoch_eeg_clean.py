"""
NeBULA Cleaned EEG — Epoch & Windowing Pipeline
-------------------------------------------------
Muskaan Garg | H00416442 | Heriot-Watt University

Loads ICA-cleaned EEG from the NeBULA Derivatives folder
(sub-XX_EEG-clean.mat) and produces windowed arrays in the
same format as epoch.py — but using the authors' cleaned data
instead of raw BrainVision files.

Their pipeline already applied:
  - Bandpass filter 0.5–50 Hz
  - Common average referencing
  - Automated artifact removal (EEGLAB Clean Rawdata)
  - Channel interpolation
  - Baseline correction (−100ms pre-event)

This script adds:
  - Motor channel selection (same 15 channels as epoch.py)
  - Resampling 1000Hz → 200Hz
  - Z-score normalisation per channel
  - Epoch extraction around G events (−500ms to +2000ms)
  - Sliding window (size=80, step=40)

OUTPUT (saved to ./preprocessed/):
  Per-subject:
    sub-XX/sub-XX_free_eeg_clean.npy     shape (n_trials, 15, 500)
    sub-XX/sub-XX_free_labels_clean.npy  shape (n_trials,)

  Combined windowed:
    X_eeg_win_clean.npy        shape (n_windows, 15, 80)
    y_win_clean.npy            shape (n_windows,)
    subject_ids_win_clean.npy  shape (n_windows,)
    trial_ids_win_clean.npy    shape (n_windows,)

Run from project root:
    python epoch_clean.py
"""

import os
import glob
import numpy as np
import h5py

# ── Constants ────────────────────────────────────────────────────────────────

# Same 15 motor channels as epoch.py
MOTOR_CH = ['C3', 'C4', 'Cz', 'FC3', 'FC4', 'CP3', 'CP4',
            'C1', 'C2', 'C5', 'C6', 'FC1', 'FC2', 'CP1', 'CP2']

# Exclude same subjects as original pipeline
# Also exclude sub-28 which was not in original preprocessed data
# Handle both 2-digit and 3-digit filename formats
VALID_SUBS = {str(s).zfill(2) for s in
              [1,2,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,
               22,23,24,25,26,27,29,30,31,32,33,34,35,36,37,38,39,40]}

EEG_FS    = 1000   # NeBULA recordings at 1000 Hz
TARGET_FS = 200    # resample target — same as epoch.py
DECIMATE  = EEG_FS // TARGET_FS  # factor of 5

# Epoch window around G event — identical to epoch.py
PRE_SAMPLES  = int(0.5 * TARGET_FS)   # 100 samples = 500ms before onset
POST_SAMPLES = int(2.0 * TARGET_FS)   # 400 samples = 2000ms after onset
EPOCH_LEN    = PRE_SAMPLES + POST_SAMPLES  # 500 samples total

# Sliding window — identical to epoch.py
WIN_SIZE = 80   # 400ms at 200Hz
WIN_STEP = 40   # 200ms, 50% overlap

FREE_COND_IDX = 0   # 0=free, 1=low, 2=high — free condition only

CLEAN_DIR  = './data/Cleaned_EEG'
OUTPUT_DIR = './preprocessed'


# ── Helpers ──────────────────────────────────────────────────────────────────

def read_str(f, ref):
    """Read a MATLAB character array stored as an h5py reference."""
    try:
        obj = f[ref]
        val = obj[()]
        if hasattr(val, 'dtype') and val.dtype.kind == 'u':
            return ''.join(chr(c) for c in val.flatten())
        return str(val)
    except Exception:
        return ''


def get_channel_names(f, eeg_obj):
    """Return list of channel name strings from chanlocs.labels."""
    labels = eeg_obj['chanlocs']['labels']
    return [read_str(f, labels[i, 0]) for i in range(labels.shape[0])]


def get_motor_indices(ch_names):
    """
    Return indices of the 15 motor channels in the channel list.
    Prints a warning for any missing channels.
    """
    indices   = []
    available = []
    for ch in MOTOR_CH:
        if ch in ch_names:
            indices.append(ch_names.index(ch))
            available.append(ch)
        else:
            print(f'    WARNING: {ch} not found — may have been removed by ICA')
    return indices, available


def get_g_events(f, eeg_obj):
    """
    Extract G events from the EEGLAB event structure.
    Returns list of (latency_at_1000hz, task_label) tuples.
    Event types are stored as 'G 1', 'G 2', 'G 3' (with space).
    """
    types   = eeg_obj['event']['type']
    latency = eeg_obj['event']['latency']

    events = []
    for i in range(types.shape[0]):
        t = read_str(f, types[i, 0])
        if t.startswith('G'):
            try:
                label = int(t.split()[-1])
                lat   = f[latency[i, 0]][()].flatten()[0]
                events.append((int(lat), label))
            except Exception:
                continue
    return events


# ── Per-subject loading ───────────────────────────────────────────────────────

def load_subject(mat_path):
    """
    Load cleaned EEG for one subject (free condition only).

    Returns:
        data   — (15, n_samples) float32 — motor channels, 1000Hz
        events — list of (latency_1000hz, label) tuples
    """
    with h5py.File(mat_path, 'r') as f:
        eeg_refs = f['EEG_clean']
        ref      = eeg_refs[FREE_COND_IDX, 0]
        obj      = f[ref]

        ch_names = get_channel_names(f, obj)
        motor_idx, available = get_motor_indices(ch_names)

        # data stored as (n_samples, n_channels) — select motor channels and transpose
        data = obj['data'][:]                      # (n_samples, 127)
        data = data[:, motor_idx].T.astype(np.float32)  # (n_motor, n_samples)

        events = get_g_events(f, obj)

    return data, events


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(data):
    """
    Resample from 1000Hz to 200Hz and z-score per channel.
    Bandpass and CAR were already applied by the NeBULA pipeline.
    Decimation by 5 is safe since their bandpass ceiling is 50Hz.
    """
    # Resample: 1000Hz → 200Hz via integer decimation
    data = data[:, ::DECIMATE]   # (n_ch, n_samples // 5)

    # Z-score per channel across the full recording
    data = (data - data.mean(axis=1, keepdims=True)) / \
           (data.std(axis=1, keepdims=True) + 1e-8)

    return data


# ── Epoch extraction ──────────────────────────────────────────────────────────

def extract_epochs(data, events):
    """
    Cut fixed windows around each G event.
    Latency is converted from 1000Hz to 200Hz before slicing.

    Returns:
        eeg_epochs — (n_trials, 15, 500)
        labels     — (n_trials,)
    """
    eeg_epochs = []
    labels     = []
    skipped    = 0

    for lat_1000, label in events:
        # convert onset sample from 1000Hz to 200Hz
        onset_200 = int(lat_1000 / DECIMATE)
        start     = onset_200 - PRE_SAMPLES
        end       = onset_200 + POST_SAMPLES

        if start < 0 or end > data.shape[1]:
            skipped += 1
            continue

        eeg_epochs.append(data[:, start:end])   # (15, 500)
        labels.append(label)

    if skipped:
        print(f'    [{skipped} trials skipped — window out of bounds]')

    return np.array(eeg_epochs), np.array(labels)


# ── Windowing ─────────────────────────────────────────────────────────────────

def apply_windowing(epochs, labels):
    """
    Slide a window across each epoch.
    Identical to epoch.py — size=80, step=40, gives 11 windows per trial.

    Returns:
        windows   — (n_windows, 15, 80)
        win_labels — (n_windows,)
        trial_ids  — (n_windows,)
    """
    windows, win_labels, trial_ids = [], [], []

    for trial_idx, (epoch, label) in enumerate(zip(epochs, labels)):
        n_samples = epoch.shape[1]
        for s in range(0, n_samples - WIN_SIZE + 1, WIN_STEP):
            windows.append(epoch[:, s : s + WIN_SIZE])
            win_labels.append(label)
            trial_ids.append(trial_idx)

    return (np.array(windows),
            np.array(win_labels),
            np.array(trial_ids))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('=' * 65)
    print('  NeBULA Cleaned EEG — Epoch & Windowing Pipeline')
    print('  Muskaan Garg | H00416442 | Heriot-Watt University')
    print('=' * 65)
    print(f'  Input dir  : {CLEAN_DIR}')
    print(f'  Output dir : {OUTPUT_DIR}')
    print(f'  Condition  : free only (index {FREE_COND_IDX})')
    print(f'  Epoch      : -{PRE_SAMPLES} to +{POST_SAMPLES} samples ({EPOCH_LEN} total)')
    print(f'  Window     : size={WIN_SIZE}, step={WIN_STEP}')
    print()

    mat_files = sorted(glob.glob(os.path.join(CLEAN_DIR, 'sub-*_EEG-clean.mat')))

    if not mat_files:
        print(f'ERROR: No .mat files found in {CLEAN_DIR}')
        print('       Check the path and filename pattern.')
        return

    print(f'  Found {len(mat_files)} subject files\n')

    all_eeg_win = []
    all_y_win   = []
    all_sid_win = []
    all_tid_win = []
    loaded      = 0

    for mat_path in mat_files:
        basename = os.path.basename(mat_path)
        sid      = basename.split('_')[0].replace('sub-', '')

        # normalise to 2-digit for comparison
        sid_norm = sid.lstrip('0').zfill(2) if sid.lstrip('0') else '0'
        sid_norm = sid_norm.zfill(2)
        if sid_norm not in VALID_SUBS:
            print(f'  Subject {sid}: SKIPPED (not in original subject list)')
            continue

        print(f'  ── Subject {sid} ─────────────────────────────────────────')

        try:
            data, events = load_subject(mat_path)
        except Exception as e:
            print(f'    ERROR loading: {e}')
            continue

        print(f'    Loaded: {data.shape} at {EEG_FS}Hz  |  {len(events)} G events')

        # preprocess
        data = preprocess(data)
        print(f'    After resample + z-score: {data.shape} at {TARGET_FS}Hz')

        # extract epochs
        eeg_ep, labels = extract_epochs(data, events)
        if len(labels) == 0:
            print('    No valid epochs — skipping subject')
            continue

        print(f'    Epochs: {eeg_ep.shape}  '
              f'(Task1={(labels==1).sum()}, '
              f'Task2={(labels==2).sum()}, '
              f'Task3={(labels==3).sum()})')

        # save per-subject trial arrays
        sub_dir = os.path.join(OUTPUT_DIR, f'sub-{sid}')
        os.makedirs(sub_dir, exist_ok=True)
        np.save(os.path.join(sub_dir, f'sub-{sid}_free_eeg_clean.npy'),    eeg_ep)
        np.save(os.path.join(sub_dir, f'sub-{sid}_free_labels_clean.npy'), labels)

        # apply windowing
        eeg_win, y_win, t_ids = apply_windowing(eeg_ep, labels)

        sid_int = int(sid.lstrip('0') or '0')
        n_win   = len(y_win)

        all_eeg_win.append(eeg_win)
        all_y_win.append(y_win)
        all_sid_win.append(np.full(n_win, sid_int, dtype=np.int32))
        all_tid_win.append(t_ids)

        print(f'    Windows: {eeg_win.shape}  '
              f'(Task1={(y_win==1).sum()}, '
              f'Task2={(y_win==2).sum()}, '
              f'Task3={(y_win==3).sum()})')
        loaded += 1

    if not all_eeg_win:
        print('\nERROR: No subjects loaded successfully.')
        return

    # combine and save
    X_eeg = np.concatenate(all_eeg_win, axis=0)
    y     = np.concatenate(all_y_win,   axis=0)
    sids  = np.concatenate(all_sid_win, axis=0)
    tids  = np.concatenate(all_tid_win, axis=0)

    np.save(os.path.join(OUTPUT_DIR, 'X_eeg_win_clean.npy'),        X_eeg)
    np.save(os.path.join(OUTPUT_DIR, 'y_win_clean.npy'),             y)
    np.save(os.path.join(OUTPUT_DIR, 'subject_ids_win_clean.npy'),   sids)
    np.save(os.path.join(OUTPUT_DIR, 'trial_ids_win_clean.npy'),     tids)

    print()
    print('=' * 65)
    print(f'  Subjects processed : {loaded}')
    print(f'  EEG windows shape  : {X_eeg.shape}')
    print(f'  Labels             : Task1={(y==1).sum()}, '
          f'Task2={(y==2).sum()}, Task3={(y==3).sum()}')
    print(f'  Unique subjects    : {np.unique(sids).tolist()}')
    print(f'  Saved to           : {OUTPUT_DIR}/')
    print('=' * 65)


if __name__ == '__main__':
    main()