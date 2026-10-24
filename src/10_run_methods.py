import os
import sys, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib, methods
from collections import Counter

t0=time.time()
HA,GT=lib.val_split()
R,ridx,items,iidx=methods.matrix(HA)
pop=lib.popularity(HA); FB=[s for s,_ in pop.most_common(5)]
popvec=methods.popularity_scores(HA,items)
airB=lib.AIRINGS["B"]
air=np.array([np.log1p(airB.get(s,0)) for s in items],dtype=np.float32)
items_arr=np.array(items)
print(f"users={R.shape[0]} items={R.shape[1]} [{time.time()-t0:.0f}s]")

def top5(Sc, name):
    preds={}
    for u in lib.TEST_UI:
        r=ridx.get(u)
        if r is None: preds[u]=list(FB); continue
        s=Sc[r]
        idx=np.argpartition(-s,5)[:5]
        idx=idx[np.argsort(-s[idx])]
        preds[u]=[int(items_arr[i]) for i in idx]
    m=lib.evaluate(preds,GT)
    print(f"  {m:.5f}  {name}   [{time.time()-t0:.0f}s]")
    return m, preds

RES={}
# 0 popularity
RES["popularity"]=top5(np.tile(popvec,(R.shape[0],1)),"popularity (baseline)")[0]
# repeat reference (what we already submitted)
RES["repeat x log(air)"]=top5(R*np.log1p(np.maximum(R,0))*0+R*air[None,:],"repeat x log(airings)")[0]

print("\n-- iALS --")
t=time.time(); X,Y=methods.ials(R,factors=64,reg=0.05,alpha=20.0,iters=15)
print(f"   fit {time.time()-t:.0f}s")
RES["MF (iALS)"]=top5(X@Y.T,"MF iALS f=64 reg=0.05 alpha=20")[0]

print("\n-- CBF --")
F,V=methods.item_features(items); print(f"   vocab={V}")
RES["CBF"]=top5(methods.cbf_scores(R,F),"content-based (TF-IDF attributes)")[0]

print("\n-- association rules --")
Sa,conf,supp=methods.assoc_scores(R,min_support=5)
RES["assoc rules"]=top5(Sa,"association rules (confidence)")[0]

print("\n-- personalized PageRank --")
t=time.time(); Sp=methods.ppr_scores(R,alpha=0.85,iters=20)
print(f"   {time.time()-t:.0f}s")
RES["PPR"]=top5(Sp,"personalized PageRank alpha=0.85")[0]

print("\n================ summary (hold-out MAP@5) ================")
for k,v in sorted(RES.items(),key=lambda x:-x[1]): print(f"  {v:.5f}  {k}")
np.save(lib.W+"val_items.npy", items_arr)
