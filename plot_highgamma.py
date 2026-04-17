import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, hilbert
import os
import sys

MAT_PATH = "data/KeywordReading_Overt_R01.mat"
LAB_PATH = "data/KeywordReading_Overt_R01_trials.lab"
OUT = "runs/figs/highgamma_ch{ch}.png"
LOW, HIGH = 70, 150
ORDER = 4


def load_ecog_matrix(mat_path):
    """
    Returns: ecog (samples, channels), fs
    - uses 'signal' key from your .mat
    - transposes if necessary so first dim = samples
    - fs defaults to 1000 if not found
    """

    m = loadmat(mat_path)
    if "signal" not in m:
        raise KeyError("Key 'signal' not found in MAT file. Available: " + ", ".join(m.keys()))

    X = m["signal"]
    # Ensure shape = (samples, channels)
    if X.ndim != 2:
        raise ValueError(f"'signal' must be 2D; got shape {X.shape}")
    # Heuristic: if channels > samples, assume it's (channels, samples) and transpose
    if X.shape[0] < X.shape[1]:
        # likely (channels, samples) -> transpose
        if X.shape[0] <= 512 and X.shape[1] > X.shape[0]:
            X = X.T
    else:
        # likely already (samples, channels); do nothing
        pass

    # Sampling rate (default 1000 Hz)
    fs = 1000
    # Try to read from parameters if present and numeric
    params = m.get("parameters", None)
    if params is not None:
        # parameters is often a MATLAB struct; try common access patterns
        try:
            # Common layouts to try, ignore failures
            for key in ["samplerate", "fs", "Fs", "sample_rate"]:
                if isinstance(params, np.ndarray) and params.size == 1:
                    p0 = params.item()
                    if hasattr(p0, key):
                        val = getattr(p0, key)
                        fs = int(np.squeeze(val))
                        break
                elif isinstance(params, dict) and key in params:
                    fs = int(np.squeeze(params[key]))
                    break
        except Exception:
            pass

    return X, fs


def bandpass(x, fs, low=LOW, high=HIGH, order=ORDER):
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")
    return filtfilt(b, a, x, axis=0)


def load_lab(path):
    # Supports: start end label   OR   start label
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
            evts.append((start, label))
    return evts


def main(ch_idx=42):
    data, fs = load_ecog_matrix(MAT_PATH)
    x = data[:, ch_idx]
    x = bandpass(x, fs=fs)
    env = np.abs(hilbert(x))
    z = (env - env.mean()) / (env.std() + 1e-8)
    t = np.arange(len(z)) / fs

    labels = load_lab(LAB_PATH)

    plt.figure(figsize=(13, 4))
    plt.plot(t, z, linewidth=1.0, label=f"Ch {ch_idx} (70–150Hz envelope, z)")
    ymax = np.percentile(z, 99)
    for start, lab in labels:
        plt.axvline(start, linestyle="--")
        plt.text(start + 0.02, ymax, lab, fontsize=8, rotation=90, va="top")
    plt.title("High-Gamma Envelope with Labeled Onsets")
    plt.xlabel("Time (s)")
    plt.ylabel("Z-scored Envelope")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT.format(ch=ch_idx)), exist_ok=True)
    out = OUT.format(ch=ch_idx)
    plt.savefig(out, dpi=180)
    print(f"Saved {out}")


if __name__ == "__main__":
    ch = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    main(ch)
