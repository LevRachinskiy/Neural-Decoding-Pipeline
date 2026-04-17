import time, json, base64, os
import numpy as np
import redis

def deserialize_array(s: str):
    payload = json.loads(s)
    data = base64.b64decode(payload["data_b64"])
    return np.frombuffer(data, dtype=np.dtype(payload["dtype"])).reshape(payload["shape"])

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
PROC_STREAM = os.getenv("PROC_STREAM", "ecog:proc")

if __name__ == "__main__":
    r = redis.from_url(REDIS_URL)
    last_id = "$"   # "0-0" to replay
    print(f"[tap] Listening on {PROC_STREAM} at {REDIS_URL}")
    n, t0 = 0, time.time()
    while True:
        resp = r.xread({PROC_STREAM: last_id}, block=3000, count=1)
        if not resp:
            continue
        _, msgs = resp[0]
        for msg_id, fields in msgs:
            last_id = msg_id
            ts0 = float(fields[b"ts0"].decode())
            ts1 = float(fields[b"ts1"].decode())
            mean_env = deserialize_array(fields[b"mean_env"].decode())[0]  # (n_chan,)
            print(f"[tap] {ts0:.3f}-{ts1:.3f}s  mean_env[:5]={np.round(mean_env[:5],2)}")
            n += 1
            if n % 20 == 0:
                dt = time.time() - t0
                print(f"[tap] received {n} chunks in {dt:.1f}s  (~{n/dt:.1f} Hz)")
