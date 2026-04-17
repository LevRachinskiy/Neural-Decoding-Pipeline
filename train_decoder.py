import os, json, numpy as np, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter

DS_OUT   = Path(os.getenv("DS_OUT","datasets"))
MODELDIR = Path(os.getenv("MODEL_DIR","models")); MODELDIR.mkdir(parents=True, exist_ok=True)
EPOCHS   = int(os.getenv("EPOCHS","10")); LR=float(os.getenv("LR","1e-3"))
BATCH    = int(os.getenv("BATCH","64")); HIDDEN=int(os.getenv("HIDDEN","64"))
DROPOUT  = float(os.getenv("DROPOUT","0.1"))
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

npz = np.load(DS_OUT/"train.npz", allow_pickle=True)
X = npz["X"]              # (N,T,C)
y = npz["y"].astype(str)  # (N,)
win_sec=float(npz["win_sec"]); hop_sec=float(npz["hop_sec"])
mask = y != "None"; X = X[mask]; y = y[mask]
if len(X)==0: raise SystemExit("No labeled samples (all 'None'). Recreate dataset.")

labels = sorted(set(y.tolist())); label2id = {lab:i for i,lab in enumerate(labels)}
y_id = np.array([label2id[s] for s in y], dtype=np.int64)
N,T,C = X.shape
idx = np.arange(N); np.random.shuffle(idx); split=int(0.85*N)
tr,va = idx[:split], idx[split:]
Xtr,ytr = X[tr], y_id[tr]; Xva,yva = X[va], y_id[va]

mu = Xtr.mean(axis=(0,1)); sd = Xtr.std(axis=(0,1))+1e-8
Xtr=(Xtr-mu)/sd; Xva=(Xva-mu)/sd
Xtr_t=torch.from_numpy(Xtr); ytr_t=torch.from_numpy(ytr)
Xva_t=torch.from_numpy(Xva); yva_t=torch.from_numpy(yva)
train_loader=DataLoader(TensorDataset(Xtr_t,ytr_t),batch_size=BATCH,shuffle=True)
val_loader  =DataLoader(TensorDataset(Xva_t,yva_t),batch_size=BATCH,shuffle=False)

class GRUClassifier(nn.Module):
    def __init__(self,in_ch,hidden,nc,dropout=0.1):
        super().__init__()
        self.gru=nn.GRU(in_ch,hidden,batch_first=True)
        self.drop=nn.Dropout(dropout); self.fc=nn.Linear(hidden,nc)
    def forward(self,x):
        _,h=self.gru(x); h=h[-1]; return self.fc(self.drop(h))

model=GRUClassifier(C,HIDDEN,len(labels),DROPOUT).to(DEVICE)
opt=torch.optim.Adam(model.parameters(),lr=LR); loss_fn=nn.CrossEntropyLoss()

def run(loader,train=True):
    model.train(train); tot=ncorrect=loss_sum=0
    with torch.set_grad_enabled(train):
        for xb,yb in loader:
            xb=xb.to(DEVICE); yb=yb.to(DEVICE)
            logits=model(xb); loss=loss_fn(logits,yb)
            if train: opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            pred=logits.argmax(1); tot+=yb.numel(); ncorrect+=(pred==yb).sum().item()
            loss_sum += loss.item()*yb.numel()
    return loss_sum/tot, ncorrect/tot

print(f"[train] X={X.shape} classes={labels} train={len(tr)} val={len(va)}")
print(f"[train] train label counts: {Counter(ytr.tolist())}")

best=-1
for ep in range(1,EPOCHS+1):
    tl,ta=run(train_loader,True); vl,va_=run(val_loader,False)
    if va_>best: best=va_; torch.save(model.state_dict(), MODELDIR/"decoder.pt")
    print(f"[{ep:02d}] train {ta:.3f}/{tl:.4f} | val {va_:.3f}/{vl:.4f} best={best:.3f}")

cfg={"win_sec":win_sec,"hop_sec":hop_sec,"n_channels":int(C),
     "mu":mu.tolist(),"sd":sd.tolist(),"labels":labels,"label2id":label2id,
     "model":f"GRU({C},{HIDDEN})->Linear({len(labels)})"}
with open(MODELDIR/"feature_config.json","w") as f: json.dump(cfg,f,indent=2)
print(f"[train] saved models/decoder.pt and models/feature_config.json")
