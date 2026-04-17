import os
import sys

import redis
from dotenv import load_dotenv

from utils import env


def main() -> int:
    load_dotenv()
    redis_url = env("REDIS_URL", "redis://localhost:6379/0")
    evt_stream = env("EVT_STREAM", "ecog:events")

    client = redis.from_url(redis_url)
    last_id = "$"
    print(f"[events_tap] Listening on {evt_stream}")

    try:
        while True:
            response = client.xread({evt_stream: last_id}, block=5000, count=1)
            if not response:
                continue
            _, messages = response[0]
            for message_id, fields in messages:
                last_id = message_id
                try:
                    label = fields[b"label"].decode()
                    start = float(fields[b"ev_start"].decode())
                    stop = float(fields[b"ev_stop"].decode())
                except KeyError as exc:
                    print(f"[events_tap] Missing field {exc} in message {message_id}")
                    continue
                except ValueError:
                    print(f"[events_tap] Non-numeric ev_start/ev_stop in message {message_id}")
                    continue
                print(f"[events_tap] {start:.3f}-{stop:.3f}  {label}")
    except KeyboardInterrupt:
        print("[events_tap] Stopped")
        return 0


if __name__ == "__main__":
    sys.exit(main())
