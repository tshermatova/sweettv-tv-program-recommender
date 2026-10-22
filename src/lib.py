"""Shared data layer: calibrated hold-out, candidate pools, MAP@5."""
import numpy as np, pickle, csv
from collections import Counter, defaultdict

ROOT="/Users/tshermatova/Downloads/week7-exam/"
W=ROOT+"work/"; D=ROOT+"data/"; S=ROOT+"submissions/"
FEAT_THR = 0.20   # "watched" for building history/features
TGT_THR  = 1.00   # "watched" for the hold-out target (calibrated against Kaggle)
SEED = 20261021

_A=pickle.load(open(W+"airings.pkl","rb"))
AIRINGS=_A["airings"]; AIRINGS_CH=_A["airings_ch"]; B_AB=_A["B_AB"]; B_TEST=_A["B_TEST"]
SHOW_META=pickle.load(open(W+"show_meta.pkl","rb"))
CH_VIEWS=pickle.load(open(W+"ch_views.pkl","rb"))
USERS=open(W+"users.txt").read().split("\n")
UIDX={u:i for i,u in enumerate(USERS)}
TEST_USERS=[r[0] for r in list(csv.reader(open(D+"submission.csv")))[1:]]
TEST_UI=[UIDX[u] for u in TEST_USERS]

_M=np.load(W+"matches.npy")
UI,SID,OV,PDUR,PST=_M[:,0],_M[:,1],_M[:,2],_M[:,3],_M[:,4]
FRAC=OV/np.maximum(PDUR,1)

def history(thr=FEAT_THR, lo=None, hi=None, weight="count"):
    """user_idx -> Counter(show -> weight) over events with prog_start in [lo,hi)."""
    m = FRAC>=thr
    if lo is not None: m &= PST>=lo
    if hi is not None: m &= PST< hi
    H=defaultdict(Counter)
    if weight=="count":
        for u,s in zip(UI[m],SID[m]): H[int(u)][int(s)]+=1
    else:
        for u,s,o in zip(UI[m],SID[m],OV[m]): H[int(u)][int(s)]+=int(o)
    return H

def target(thr=TGT_THR, lo=None, hi=None, min_n=1):
    H=defaultdict(Counter); m=FRAC>=thr
    if lo is not None: m &= PST>=lo
    if hi is not None: m &= PST< hi
    for u,s in zip(UI[m],SID[m]): H[int(u)][int(s)]+=1
    return {u:set(s for s,n in c.items() if n>=min_n) for u,c in H.items()}

def popularity(H):
    p=Counter()
    for u,c in H.items():
        for s in c: p[s]+=1
    return p

def apk(actual,pred,k=5):
    """AP@k: sum of precision at each relevant rank, over min(k, |actual|).

    Verified against the worked example in the course slides: for relevant {A,B,C},
    [A,B,D,C,E] -> 0.9167 and [D,A,E,B,C] -> 0.5333. A repeated prediction scores
    once, as in the standard ml_metrics implementation; without that guard a list
    with duplicates can exceed 1.0.
    """
    sc=0.0; h=0; seen=set()
    for i,p in enumerate(pred[:k]):
        if p in actual and p not in seen: h+=1; sc+=h/(i+1)
        seen.add(p)
    return sc/min(len(actual),k)

def evaluate(preds, gt, users=None):
    """MAP@5 over `users` (default: all 1260 test users). Users absent from gt score 0,
    which mirrors Kaggle scoring every row of submission.csv."""
    if users is None: users=TEST_UI
    tot=0.0
    for u in users:
        g=gt.get(u)
        tot += apk(g, preds.get(u,[]), 5) if g else 0.0
    return tot/len(users)

# ---- the two evaluation regimes -------------------------------------------
# VALIDATION: features from window A (< 2020-05-18), target from window B.
# FINAL:      features from all of weeks 11-30, predictions for window T.
def val_split():
    return history(lo=None, hi=B_AB), target(lo=B_AB, hi=None)
def final_history():
    return history()

def write_submission(preds, name, pad_with):
    rows=[]
    for u in TEST_USERS:
        out=list(dict.fromkeys(preds.get(UIDX[u], [])))[:5]
        for s in pad_with:
            if len(out)>=5: break
            if s not in out: out.append(s)
        assert len(out)==5, (u,out)
        rows.append((u," ".join(map(str,out))))
    p=S+name+".csv"
    with open(p,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["user_id","tv_show_id"]); w.writerows(rows)
    return p

# ---- compact global item index (all labelled programmes) -------------------
ITEMS_ALL = np.array(sorted(set(int(x) for x in np.unique(SID))), dtype=np.int64)
IIDX_ALL  = {int(s):i for i,s in enumerate(ITEMS_ALL)}
SIDX      = np.array([IIDX_ALL[int(s)] for s in SID], dtype=np.int32)
N_ITEMS   = len(ITEMS_ALL)
N_USERS   = len(USERS)
DAY       = PST // 86400
HOUR      = (PST // 3600) % 24
