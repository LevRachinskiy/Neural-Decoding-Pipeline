import os, json, base64, numpy as np, redis, torch, torch.nn as nn
from collections import deque, Counter
from pathlib import Path

REDIS_URL=os.getenv("REDIS_URL","redis://localhost:6379/0")
PROC_KEY =os.getenv("PROC_STREAM","ecog:proc")
OUT_KEY  =os.getenv("ML_STREAM","ecog:ml")
MODEL_DIR=Path(os.getenv("MODEL_DIR","models"))
CFG_PATH=MODEL_DIR/"feature_config.json"; PT_PATH=MODEL_DIR/"decoder.pt"
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
EMA_ALPHA=float(os.getenv("EMA_ALPHA","0.4"))
MIN_PROB=float(os.getenv("MIN_PROB","0.5"))
STICKY_N=max(1,int(os.getenv("STICKY_N","3")))

def deser(s:str):
    p=json.loads(s); b=base64.b64decode(p["data_b64"])
    return np.frombuffer(b, dtype=np.dtype(p["dtype"])).reshape(p["shape"])

class GRUClassifier(nn.Module):
    def __init__(self,in_ch,hidden,nc,dropout=0.1):
        super().__init__()
        self.gru=nn.GRU(in_ch,hidden,batch_first=True)
        self.drop=nn.Dropout(dropout); self.fc=nn.Linear(hidden,nc)
    def forward(self,x):
        _,h=self.gru(x); h=h[-1]; return self.fc(self.drop(h))

if __name__=="__main__":
    cfg=json.load(open(CFG_PATH))
    labels=cfg["labels"]; mu=np.array(cfg["mu"],np.float32); sd=np.array(cfg["sd"],np.float32)
    C=int(cfg["n_channels"]); win_sec=float(cfg["win_sec"])

    # Try common hidden sizes to load weights (matches trainer default 64)
    hidden_candidates=[64,32,96,128,48,80]
    for h in hidden_candidates:
        try:
            m=GRUClassifier(C,h,len(labels)); m.load_state_dict(torch.load(PT_PATH,map_location="cpu"))
            model=m.to(DEVICE).eval(); hidden=h; break
        except Exception: pass
    else:
        raise RuntimeError("Failed to load decoder.pt with common hidden sizes.")
    print(f"[ml] model loaded (hidden={hidden}) labels={labels}")

    r=redis.from_url(REDIS_URL); last="$"  # live tail
    buf=deque(); chunk_len=None; need=None
    ema_probs=None
    sticky=deque(maxlen=STICKY_N)
    label_to_idx={lab:i for i,lab in enumerate(labels)}

    while True:
        resp=r.xread({PROC_KEY:last}, block=3000, count=1)
        if not resp: continue
        _,msgs=resp[0]
        for mid,f in msgs:
            last=mid
            ts0=float(f[b"ts0"].decode()); ts1=float(f[b"ts1"].decode())
            mean_env=deser(f[b"mean_env"].decode())[0].astype(np.float32)  # (C,)

            if chunk_len is None:
                chunk_len=ts1-ts0; need=max(1,int(round(win_sec/chunk_len)))
                print(f"[ml] chunk_len={chunk_len:.3f}s, need={need}")

            buf.append((ts0,ts1,mean_env))
            while len(buf)>need: buf.popleft()
            if len(buf)<need: continue

            w0,w1=buf[0][0],buf[-1][1]
            X=np.stack([b[2] for b in buf],axis=0)     # (T,C)
            Xn=(X-mu)/sd
            Xt=torch.from_numpy(Xn[None,...]).to(DEVICE)

            with torch.inference_mode():
                logits=model(Xt); probs=torch.softmax(logits,dim=1).cpu().numpy()[0]

            if ema_probs is None:
                ema_probs=probs
            else:
                ema_probs=EMA_ALPHA*probs + (1.0-EMA_ALPHA)*ema_probs

            top=int(ema_probs.argmax()); lab=labels[top]; p=float(ema_probs[top])
            if p < MIN_PROB:
                print(f"[ml] {w0:.3f}-{w1:.3f}s  pred={lab}  p={p:.2f} (below MIN_PROB)")
                continue

            sticky.append(lab)
            if STICKY_N > 1:
                counts=Counter(sticky)
                held_lab=None; held_cnt=-1
                for candidate in reversed(sticky):
                    cnt=counts[candidate]
                    if cnt > held_cnt:
                        held_lab=candidate
                        held_cnt=cnt
                lab=held_lab if held_lab is not None else lab
                idx=label_to_idx[lab]
                p=float(ema_probs[idx])

            r.xadd(OUT_KEY, {
                "ts0": str(w0), "ts1": str(w1),
                "top_label": lab, "top_prob": f"{p:.4f}",
                "probs_json": json.dumps({labels[i]:float(ema_probs[i]) for i in range(len(labels))})
            })
            print(f"[ml] {w0:.3f}-{w1:.3f}s  pred={lab}  p={p:.2f}")
