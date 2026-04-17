import json
import os
import time

import numpy as np
import redis
import scipy.io as sio

from utils import env, serialize_array
from utils_mat import find_first_key, safe_get_nested


def _ensure_file(mat_path: str) -> None:
    if not mat_path:
        raise FileNotFoundError(
            "ECOG_MAT_PATH is not set. Update .env to point to data/<file>.mat or provide an absolute path."
        )
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(
            f"ECOG_MAT_PATH points to '{mat_path}', but the file was not found. "
            "Copy the .mat into data/ and set ECOG_MAT_PATH=data/<file>.mat or supply an absolute path."
        )


def _fetch_sampling_rate(data: dict) -> tuple[float, list[str]]:
    sampling_paths = [
        ["parameters", "SamplingRate", "NumericValue"],
        ["parameters", "fs"],
        ["parameters", "Fs"],
        ["fs"],
        ["Fs"],
        ["sampling_rate"],
    ]

    for path in sampling_paths:
        value = safe_get_nested(data, path)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric > 0:
            return numeric, path
    raise ValueError(
        "Could not determine a positive sampling rate. Checked parameters.SamplingRate.NumericValue, "
        "parameters.fs, parameters.Fs, fs, Fs, sampling_rate."
    )


def load_ecog_from_mat(mat_path: str):
    _ensure_file(mat_path)
    data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    signal_key_candidates = ["signal", "Signal", "ecog", "data", "X"]
    signal_key = find_first_key(data, signal_key_candidates)
    if not signal_key:
        raise KeyError(
            "No signal array found in .mat file. Provide one of: signal, Signal, ecog, data, X."
        )

    signal_raw = data[signal_key]
    signal = np.array(signal_raw)
    signal = np.squeeze(signal)
    if signal.ndim == 1:
        signal = signal[:, np.newaxis]
    if signal.ndim != 2:
        raise ValueError(f"Signal array must be 2D after squeeze; got shape {signal.shape} from key '{signal_key}'.")

    orientation = "samples_first"
    if signal.shape[0] < signal.shape[1]:
        signal = signal.T
        orientation = "channels_first_transposed"

    if signal.size == 0:
        raise ValueError(f"Signal array from key '{signal_key}' is empty.")

    total_channels = signal.shape[1]
    if total_channels >= 4:
        ecog = signal[:, :-3]
        aux_channels = 3
    else:
        ecog = signal
        aux_channels = 0

    ecog = ecog.astype(np.float32, copy=False)
    fs, fs_path = _fetch_sampling_rate(data)

    metadata = {
        "signal_key": signal_key,
        "fs_path": fs_path,
        "orientation": orientation,
        "raw_shape": tuple(signal.shape),
        "dtype": str(signal.dtype),
        "aux_channels": aux_channels,
    }

    return ecog, fs, metadata

def maybe_slice(ecog: np.ndarray, fs: float, start_sec: str, duration_sec: str):
    if not start_sec and not duration_sec:
        return ecog
    s0 = float(start_sec) if start_sec else 0.0
    dur = float(duration_sec) if duration_sec else (ecog.shape[0]/fs - s0)
    i0 = max(0, int(round(s0 * fs)))
    i1 = min(ecog.shape[0], int(round((s0 + dur) * fs)))
    return ecog[i0:i1]

if __name__ == "__main__":
    mat_path = env("ECOG_MAT_PATH", "data/KeywordReading_Overt_R01.mat")
    redis_url = env("REDIS_URL", "redis://localhost:6379/0")
    raw_key = env("RAW_STREAM", "ecog:raw")
    chunk_samples = int(env("CHUNK_SAMPLES", "200"))
    start_sec = env("START_SEC", "")
    duration_sec = env("DURATION_SEC", "")

    print(f"[acq] Loading: {mat_path}")
    ecog, fs, meta = load_ecog_from_mat(mat_path)
    fs_path_str = ".".join(meta["fs_path"]) if meta["fs_path"] else "unknown"
    print(
        "[acq] Parsed signal key='{key}', raw_shape={raw_shape}, dtype={dtype}, "
        "orientation={orientation}, fs source={fs_source}, aux_channels={aux}".format(
            key=meta["signal_key"],
            raw_shape=meta["raw_shape"],
            dtype=meta["dtype"],
            orientation=meta["orientation"],
            fs_source=fs_path_str,
            aux=meta["aux_channels"],
        )
    )
    ecog = maybe_slice(ecog, fs, start_sec, duration_sec)
    n_samples, n_chan = ecog.shape
    print(
        f"[acq] Ready: shape={n_samples}x{n_chan} (samples x channels), dtype={ecog.dtype}, "
        f"fs={fs} Hz, chunk={chunk_samples}"
    )

    r = redis.from_url(redis_url)

    for i0 in range(0, n_samples, chunk_samples):
        i1 = min(n_samples, i0 + chunk_samples)
        chunk = ecog[i0:i1]
        fields = {
            "fs": str(fs),
            "ts0": str(i0 / fs),
            "ts1": str(i1 / fs),
            "shape": json.dumps([int(i) for i in chunk.shape]),
            "data": serialize_array(chunk),
        }
        r.xadd(raw_key, fields)
        time.sleep((i1 - i0) / fs)     # pace like real-time
    print("[acq] Done.")
