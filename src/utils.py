import os, json, base64
import numpy as np

def env(name: str, default: str = "") -> str:
    return os.getenv(name, default)

def serialize_array(arr: np.ndarray) -> str:
    arr = np.ascontiguousarray(arr)
    payload = {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "data_b64": base64.b64encode(arr.tobytes()).decode("ascii"),
    }
    return json.dumps(payload)

def deserialize_array(s: str) -> np.ndarray:
    payload = json.loads(s)
    data = base64.b64decode(payload["data_b64"])
    return np.frombuffer(data, dtype=np.dtype(payload["dtype"])).reshape(payload["shape"])
