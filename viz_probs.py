import os, json, argparse, time
from collections import deque

import numpy as np
import redis
import matplotlib.pyplot as plt

# --- args / env ---
ap = argparse.ArgumentParser()
ap.add_argument("--window-sec", type=float, default=float(os.getenv("VIZ_WINDOW_SEC", "15")),
                help="seconds visible on the x-axis")
ap.add_argument("--from-start-events", action="store_true",
                help="replay events from 0-0 so past trials show as overlays")
ap.add_argument("--topk", type=int, default=int(os.getenv("VIZ_TOPK", "4")),
                help="show top-k class lines in the legend")
ap.add_argument("--save", metavar="FILENAME",
                help="Save animation to GIF or MP4 (e.g., demo.gif or demo.mp4)")
ap.add_argument("--frames", type=int, default=300,
                help="Number of frames to record when --save is used (e.g., 600 ≈ 30s at 50ms interval).")
args = ap.parse_args()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
ML_KEY    = os.getenv("ML_STREAM", "ecog:ml")
EVT_KEY   = os.getenv("EVT_STREAM","ecog:events")

# --- redis ---
r = redis.from_url(REDIS_URL)
last_ml  = "$"  # live tail for ML stream
last_evt = "0-0" if args.from_start_events else "$"

# --- rolling buffers ---
WIN_SEC = float(args.window_sec)
times   = deque()
probs   = {}        # label -> deque of floats
events  = deque()   # (start, stop, label)

labels  = None      # discovered from first ML msg (or use feature_config.json if you prefer)
t0_ref  = None
fps_est = 5.0       # default ~5Hz chunks

# --- plotting ---
plt.figure(figsize=(10, 5))
ax = plt.gca()
ax.set_title("Live Class Probabilities (from ecog:ml)")
ax.set_xlabel("Time (s, relative)")
ax.set_ylabel("Probability")
ax.set_ylim(0, 1.0)

lines_by_label = {}
event_patches  = []

def trim_buffers(now_rel):
    # trim time axis and per-label buffers to window
    while times and now_rel - times[0] > WIN_SEC:
        times.popleft()
        for dq in probs.values():
            if dq: dq.popleft()
    # trim old events (keep a bit beyond window for nice fade)
    while events and now_rel - events[0][1] > 1.0:
        events.popleft()

def pull_events():
    global last_evt
    # drain events quickly (non-blocking)
    while True:
        resp = r.xread({EVT_KEY: last_evt}, block=1, count=128)
        if not resp:
            break
        _, msgs = resp[0]
        for mid, f in msgs:
            last_evt = mid
            s0 = float(f[b"ev_start"].decode())
            s1 = float(f[b"ev_stop"].decode())
            lab = f[b"label"].decode()
            events.append((s0, s1, lab))

def update(_frame):
    global last_ml, labels, t0_ref, fps_est

    # get one ML message (block briefly)
    resp = r.xread({ML_KEY: last_ml}, block=200, count=1)
    if not resp:
        # no new preds; just redraw current state
        redraw()
        return

    _, msgs = resp[0]
    for mid, f in msgs:
        last_ml = mid
        ts0 = float(f[b"ts0"].decode()); ts1 = float(f[b"ts1"].decode())
        probs_json = json.loads(f[b"probs_json"].decode())

        # discover labels once
        if labels is None:
            labels = list(probs_json.keys())
            # create per-label buffers and line objects
            for lab in labels:
                probs[lab] = deque()
                (line,) = ax.plot([], [], label=lab, linewidth=2)
                lines_by_label[lab] = line
            ax.legend(ncol=min(len(labels), 6), loc="upper right", fontsize=8)

        # time base (relative)
        t_mid = 0.5 * (ts0 + ts1)
        if t0_ref is None:
            t0_ref = t_mid
        t_rel = t_mid - t0_ref

        # append time + probs (in label order)
        times.append(t_rel)
        for lab in labels:
            probs[lab].append(float(probs_json.get(lab, 0.0)))

        # estimate fps (for nicer x-limits on first seconds)
        if len(times) >= 2:
            dt = times[-1] - times[-2]
            if dt > 1e-6:
                fps_est = 1.0 / dt

    # pull any new events too
    pull_events()

    # trim rolling window
    trim_buffers(times[-1])

    # redraw
    redraw()

def redraw():
    if not times:
        return

    # choose top-k labels by their last prob (for clarity)
    last_vals = {lab: (probs[lab][-1] if probs[lab] else 0.0) for lab in probs}
    ranked = sorted(last_vals.items(), key=lambda x: x[1], reverse=True)
    show_labels = set([lab for lab, _ in ranked[:args.topk]])

    # update line data
    t = np.array(times, dtype=float)
    for lab, line in lines_by_label.items():
        y = np.array(probs[lab], dtype=float) if probs[lab] else np.array([])
        line.set_data(t, y)
        line.set_visible(lab in show_labels)

    # clear old event patches and redraw current window's events
    for p in event_patches:
        p.remove()
    event_patches.clear()

    # find y span for shaded boxes
    ymin, ymax = ax.get_ylim()
    # draw semi-transparent spans where events overlap the visible x-range
    t0_vis, t1_vis = max(0.0, t[-1] - WIN_SEC), t[-1]
    for (s0, s1, lab) in events:
        s0r = s0 - (t0_ref or s0)
        s1r = s1 - (t0_ref or s1)
        if s1r < t0_vis or s0r > t1_vis:
            continue
        patch = ax.axvspan(max(s0r, t0_vis), min(s1r, t1_vis),
                           alpha=0.15, ymin=0.0, ymax=1.0, label=None)
        event_patches.append(patch)
        # label text near top
        ax.text((max(s0r, t0_vis)+min(s1r, t1_vis))/2, ymax*0.97, lab,
                ha="center", va="top", fontsize=8, alpha=0.7)

    # adjust x-limits
    ax.set_xlim(max(0.0, t[-1] - WIN_SEC), t[-1] + 0.2 / max(fps_est, 1e-3))
    ax.figure.canvas.draw_idle()

from matplotlib.animation import FuncAnimation

FPS = int(1000 / 50)  # 20 fps at 50 ms interval

ani = FuncAnimation(
    plt.gcf(),
    update,
    interval=50,
    cache_frame_data=False,
    save_count=args.frames,   # <-- set frames here
)
plt.tight_layout()

if args.save:
    print(f"[viz] Recording animation to: {args.save}  frames={args.frames}")
    if args.save.endswith(".gif"):
        ani.save(args.save, writer="pillow", fps=FPS)   # <-- no save_count here
    elif args.save.endswith(".mp4"):
        ani.save(args.save, writer="ffmpeg", fps=FPS)   # <-- no save_count here
    else:
        raise ValueError("Unsupported format. Use .gif or .mp4")
else:
    plt.show()
