"""Candidate generation + feature-based re-ranking.

Nested design, so the hold-out is never fitted on:
  INNER  features < 2020-04-13          labels 2020-04-13..2020-05-18   -> fits the ranker
  OUTER  features < 2020-05-18          labels 2020-05-18..2020-07-27   -> selection / reporting
  FINAL  features all of weeks 11-30    predictions for weeks 31-42
"""
import sys, numpy as np, calendar
from collections import Counter, defaultdict
sys.path.insert(0,"/Users/tshermatova/Downloads/week7-exam/src")
import lib, methods

def ts(y,m,d): return calendar.timegm((y,m,d,0,0,0,0,0,0))
INNER_SPLIT = ts(2020,4,13)
N_POP_CAND  = 60

_AR = np.load(lib.W+"airings_raw.npy")      # show, channel, start (sorted by start)
def airings_in(lo, hi):
    i=np.searchsorted(_AR[:,2], lo, "left"); j=np.searchsorted(_AR[:,2], hi, "right")
    sl=_AR[i:j]
    cnt=Counter(); ch=defaultdict(Counter)
    for s,c,_ in sl:
        cnt[int(s)]+=1; ch[int(s)][int(c)]+=1
    main={s:max(d.items(),key=lambda x:x[1])[0] for s,d in ch.items()}
    return cnt, main

def collab_scores(R, items):
    out={}
    out["assoc"],_,_ = methods.assoc_scores(R, min_support=5)
    out["ppr"]       = methods.ppr_scores(R, alpha=0.85, iters=20)
    F,_              = methods.item_features(items)
    out["cbf"]       = methods.cbf_scores(R, F)
    return out

COLS=["own_log","own_rank","own_share","is_repeat","pop_log","ev_log",
      "air_log","ch_aff","assoc","ppr","cbf"]

def build(hist_H, air_lo, air_hi, label_gt=None, users=None, collab=None, pack=None):
    if pack is None:
        R, ridx, items, iidx = methods.matrix(hist_H)
        if collab is None: collab = collab_scores(R, items)
        pack=(R,ridx,items,iidx,collab)
    R,ridx,items,iidx,collab = pack
    air, main_ch = airings_in(air_lo, air_hi)
    pop_users = lib.popularity(hist_H)
    pop_events= Counter()
    for u,c in hist_H.items():
        for s,n in c.items(): pop_events[s]+=n
    top_pop=[s for s,_ in pop_users.most_common(N_POP_CAND)]
    if users is None: users = lib.TEST_UI
    X=[]; meta=[]; y=[]
    for u in users:
        h=hist_H.get(u, {})
        tot=sum(h.values()) or 1
        rank_of={s:i for i,(s,_) in enumerate(sorted(h.items(), key=lambda x:-x[1]))}
        cand=list(h.keys())+[s for s in top_pop if s not in h]
        r=ridx.get(u)
        chv=lib.CH_VIEWS.get(u,{}); chtot=sum(chv.values()) or 1
        for s in cand:
            j=iidx.get(s); n=h.get(s,0); ch=main_ch.get(s)
            ok = (r is not None and j is not None)
            X.append([np.log1p(n), 1.0/(1.0+rank_of.get(s,999)), n/tot, 1.0 if n>0 else 0.0,
                      np.log1p(pop_users.get(s,0)), np.log1p(pop_events.get(s,0)),
                      np.log1p(air.get(s,0)), (chv.get(ch,0)/chtot) if ch is not None else 0.0,
                      float(collab["assoc"][r,j]) if ok else 0.0,
                      float(collab["ppr"][r,j])   if ok else 0.0,
                      float(collab["cbf"][r,j])   if ok else 0.0])
            meta.append((u,s))
            if label_gt is not None: y.append(1 if s in label_gt.get(u,()) else 0)
    return (np.array(X,dtype=np.float32), np.array(meta,dtype=np.int64),
            (np.array(y,dtype=np.int8) if label_gt is not None else None), pack)

def top5_from_scores(sc, meta, fallback):
    out={}
    order=np.lexsort((-sc, meta[:,0]))
    cur=None; buf=[]
    for k in order:
        u=int(meta[k,0])
        if u!=cur:
            if cur is not None: out[cur]=buf
            cur=u; buf=[]
        if len(buf)<5: buf.append(int(meta[k,1]))
    if cur is not None: out[cur]=buf
    for u in lib.TEST_UI:
        o=list(out.get(u,[]))
        for s in fallback:
            if len(o)>=5: break
            if s not in o: o.append(s)
        out[u]=o[:5]
    return out
