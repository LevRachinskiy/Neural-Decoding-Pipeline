import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, hilbert
from collections import defaultdict

MAT_PATH = "data/KeywordReading_Overt_R01.mat"
LAB_PATH = "data/KeywordReading_Overt_R01_trials.lab"
OUT_DIR = "runs/figs"


# ---------- loaders ----------
def load_ecog_matrix(mat_path):
    m = loadmat(mat_path)
    if "signal" not in m:
        raise KeyError("Key 'signal' not found in MAT file. Available: " + ", ".join(m.keys()))
    X = m["signal"]
    if X.ndim != 2:
        raise ValueError(f"'signal' must be 2D; got shape {X.shape}")
    # Ensure (samples, channels)
    if X.shape[0] < X.shape[1]:
        # likely (channels, samples) -> transpose if channels <= 512
        if X.shape[0] <= 512 and X.shape[1] > X.shape[0]:
            X = X.T
    fs = 1000
    params = m.get("parameters", None)
    try:
        for key in ["samplerate", "fs", "Fs", "sample_rate"]:
            if isinstance(params, np.ndarray) and params.size == 1:
                p0 = params.item()
                if hasattr(p0, key):
                    val = getattr(p0, key)
                    fs = int(np.squeeze(val))
                    break
    except Exception:
        pass
    return X, fs


def load_lab(path, fs=None, assume_seconds=True):
    evts = []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) >= 3:
                start = float(parts[0])
                label = parts[2]
            else:
                start = float(parts[0])
                label = parts[1]
            if not assume_seconds and fs is not None:
                start = start / fs
            evts.append((start, label))
    return evts


# ---------- signal utils ----------
def bandpass(x, fs, low=70, high=150, order=4):
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")
    return filtfilt(b, a, x, axis=0)


def zscore(x, axis=0, eps=1e-8):
    mu = np.mean(x, axis=axis, keepdims=True)
    sd = np.std(x, axis=axis, keepdims=True)
    return (x - mu) / (sd + eps)


# ---------- plots ----------
def plot_highgamma_channel(data, fs, labels, ch_idx=42, outdir=OUT_DIR):
    x = data[:, ch_idx]
    x = bandpass(x, fs)
    env = np.abs(hilbert(x))
    z = zscore(env, axis=0).squeeze()
    t = np.arange(len(z)) / fs

    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(13, 4))
    plt.plot(t, z, linewidth=1.0, label=f"Ch {ch_idx} (70–150Hz env, z)")
    ymax = np.percentile(z, 99)
    for start, lab in labels:
        plt.axvline(start, linestyle="--")
        plt.text(start + 0.02, ymax, lab, fontsize=8, rotation=90, va="top")
    plt.title("High-Gamma Envelope with Labeled Onsets")
    plt.xlabel("Time (s)")
    plt.ylabel("Z-scored Envelope")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(outdir, f"highgamma_ch{ch_idx}.png")
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"Saved {out}")


def plot_event_locked(data, fs, labels, ch_idx=42, pre=0.8, post=1.2, min_trials=3, outdir=OUT_DIR):
    x = data[:, ch_idx]
    x = bandpass(x, fs)
    env = np.abs(hilbert(x))
    z = zscore(env, axis=0).squeeze()
    by_lab = defaultdict(list)
    n_pre, n_post = int(pre * fs), int(post * fs)
    for start, lab in labels:
        idx = int(start * fs)
        a = max(0, idx - n_pre)
        b = min(len(z), idx + n_post)
        if b - a != n_pre + n_post:
            continue
        by_lab[lab].append(z[a:b])
    t = np.linspace(-pre, post, n_pre + n_post, endpoint=False)

    os.makedirs(outdir, exist_ok=True)
    for lab, segs in by_lab.items():
        if len(segs) < min_trials:
            continue
        arr = np.vstack(segs)
        mean = arr.mean(axis=0)
        sem = arr.std(axis=0) / np.sqrt(arr.shape[0])
        plt.figure(figsize=(10, 4))
        plt.plot(t, mean, label=f"{lab} (n={arr.shape[0]})")
        plt.fill_between(t, mean - sem, mean + sem, alpha=0.25)
        plt.axvline(0, linestyle="--")
        plt.title(f"Event-Locked High-Gamma (Ch {ch_idx}) — {lab}")
        plt.xlabel("Time from label onset (s)")
        plt.ylabel("Z-scored Envelope")
        plt.legend()
        plt.tight_layout()
        out = os.path.join(outdir, f"event_locked_{lab}_ch{ch_idx}.png")
        plt.savefig(out, dpi=180)
        plt.close()
        print(f"Saved {out}")


def plot_event_heatmap(data, fs, labels, target_label="Enter", pre=0.5, post=0.8, max_ch=64, outdir=OUT_DIR):
    D = bandpass(data, fs)
    env = np.abs(hilbert(D, axis=0))
    Z = zscore(env, axis=0)

    evts = [(int(s * fs), lab) for s, lab in labels if lab == target_label]
    n_pre, n_post = int(pre * fs), int(post * fs)
    segs = []
    for idx, _ in evts:
        a = max(0, idx - n_pre)
        b = min(len(Z), idx + n_post)
        if b - a != n_pre + n_post:
            continue
        segs.append(Z[a:b, :])

    if not segs:
        print(f"No complete epochs for label {target_label}. Skipping heatmap.")
        return

    arr = np.stack(segs, axis=0)  # (trials, time, ch)
    mean = arr.mean(axis=0).T  # (ch, time)
    if mean.shape[0] > max_ch:
        mean = mean[:max_ch, :]
    t = np.linspace(-pre, post, mean.shape[1], endpoint=False)

    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.imshow(mean, aspect="auto", extent=[t[0], t[-1], 0, mean.shape[0]])
    plt.colorbar(label="Z-scored envelope")
    plt.axvline(0, linestyle="--")
    plt.title(f"Event-Locked High-Gamma Heatmap — {target_label} (channels x time)")
    plt.xlabel("Time from onset (s)")
    plt.ylabel("Channel")
    plt.tight_layout()
    out = os.path.join(outdir, f"heatmap_{target_label}.png")
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"Saved {out}")


# ---------- runner ----------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data, fs = load_ecog_matrix(MAT_PATH)

    # Are .lab onsets seconds or samples? If you know it's seconds, keep assume_seconds=True.
    labels = load_lab(LAB_PATH, fs=fs, assume_seconds=True)

    # Choose channels to try (speech-responsive varies by subject)
    channels = [42, 30, 56]
    for ch in channels:
        plot_highgamma_channel(data, fs, labels, ch_idx=ch, outdir=OUT_DIR)
        plot_event_locked(data, fs, labels, ch_idx=ch, outdir=OUT_DIR)

    for lab in ["Enter", "Down", "Up", "Left", "Right", "Back"]:
        plot_event_heatmap(data, fs, labels, target_label=lab, outdir=OUT_DIR)

    # Save a little manifest
    manifest = {
        "mat": MAT_PATH,
        "lab": LAB_PATH,
        "fs": fs,
        "channels_plotted": channels,
        "labels_heatmap": ["Enter", "Down", "Up", "Left", "Right", "Back"],
        "out_dir": OUT_DIR,
    }
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {os.path.join(OUT_DIR, 'manifest.json')}")


if __name__ == "__main__":
    main()
