import numpy as np
import redis
from scipy.signal import butter, sosfiltfilt, hilbert
from utils import env, serialize_array, deserialize_array

def design_bandpass(fs: float, low: float, high: float, order: int = 4):
    nyq = 0.5 * fs
    sos = butter(order, [low/nyq, high/nyq], btype="bandpass", output="sos")
    return sos

if __name__ == "__main__":
    redis_url = env("REDIS_URL", "redis://localhost:6379/0")
    raw_key = env("RAW_STREAM", "ecog:raw")
    proc_key = env("PROC_STREAM", "ecog:proc")
    bp_low = float(env("BP_LOW", "70"))
    bp_high = float(env("BP_HIGH", "150"))
    bp_order = int(env("BP_ORDER", "4"))

    r = redis.from_url(redis_url)
    last_id = "$"   # set to "0-0" to replay from beginning
    fs = None
    sos = None
    print(f"[proc] Waiting on {raw_key}...")

    while True:
        resp = r.xread({raw_key: last_id}, block=5000, count=1)
        if not resp:
            continue
        _, msgs = resp[0]
        for msg_id, fields in msgs:
            last_id = msg_id
            fs_msg = float(fields[b"fs"].decode())
            if fs is None:
                fs = fs_msg
                sos = design_bandpass(fs, bp_low, bp_high, bp_order)
                print(f"[proc] Filter {bp_low}-{bp_high} Hz (order {bp_order}) @ fs={fs}")

            chunk = deserialize_array(fields[b"data"].decode())  # (samples, channels)
            filt = sosfiltfilt(sos, chunk, axis=0)
            envp = np.abs(hilbert(filt, axis=0)).astype(np.float32)
            mean_env = envp.mean(axis=0, keepdims=True).astype(np.float32)  # (1, n_chan)

            out = {
                "fs": fields[b"fs"].decode(),
                "ts0": fields[b"ts0"].decode(),
                "ts1": fields[b"ts1"].decode(),
                "filt": serialize_array(filt.astype(np.float32)),
                "env": serialize_array(envp),
                "mean_env": serialize_array(mean_env),
            }
            r.xadd(proc_key, out)
