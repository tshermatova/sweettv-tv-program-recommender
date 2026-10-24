import os
import sys, time, calendar, numpy as np, os, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib, features, features2, collab2
import lightgbm as lgb
t0=time.time()
def ts(y,m,d): return calendar.timegm((y,m,d,0,0,0,0,0,0))
GT=lib.target(lo=lib.B_AB,hi=lib.B_TEST)
CACHE=lib.W+"ds/"; os.makedirs(CACHE,exist_ok=True)
NB=len(features2.FEATS2)-4      # drop the 4 old collab cols, collab2 appends its own
FEATS=features2.FEATS2[:NB]+collab2.ALL

def dataset(split, lab_hi, tag, label=True):
    f=CACHE+tag+".npz"
    if os.path.exists(f):
        z=np.load(f); return z["X"],z["CU"],z["CI"],(z["y"] if "y" in z else None)
    hm=features.history_arrays(split)
    users=sorted(set(int(v) for v in np.unique(lib.UI[hm])))
    lm=(lib.FRAC>=1.0)&(lib.PST>=split)&(lib.PST<lab_hi) if label else None
    X,CU,CI,y,_=features2.build2(hm,split,lab_hi,users,n_pop=80,label_mask=lm)
    blk=collab2.block(hm)
    X=collab2.attach(X,CU,CI,blk,NB)
    del blk; gc.collect()
    if y is None: np.savez_compressed(f,X=X,CU=CU,CI=CI)
    else:         np.savez_compressed(f,X=X,CU=CU,CI=CI,y=y)
    print(f"   cached {tag}: {X.shape} [{time.time()-t0:.0f}s]", flush=True)
    return X,CU,CI,y

SPL=[("w0323",ts(2020,3,23)),("w0330",ts(2020,3,30)),("w0406",ts(2020,4,6)),
     ("w0413",ts(2020,4,13)),("w0420",ts(2020,4,20)),("w0427",ts(2020,4,27))]
for tag,sp in SPL: dataset(sp, lib.B_AB, tag)
Xo,CUo,CIo,_=dataset(lib.B_AB, lib.B_TEST, "outer", label=False)
aux_pop=np.bincount(CIo,minlength=lib.N_ITEMS)
HA=lib.history(hi=lib.B_AB); pop=lib.popularity(HA)
FB=[s for s,_ in pop.most_common(5)]
print(f"all cached [{time.time()-t0:.0f}s]")

def run(tags, tag_name, **kw):
    Xs=[];ys=[];Cs=[];off=0
    for t_ in tags:
        X,CU,CI,y=dataset(dict(SPL)[t_], lib.B_AB, t_)
        Xs.append(X); ys.append(y); Cs.append(CU+off); off+=lib.N_USERS
    X=np.vstack(Xs); y=np.concatenate(ys); C=np.concatenate(Cs)
    o=np.argsort(C,kind="stable"); X,y,C=X[o],y[o],C[o]
    g=np.diff(np.flatnonzero(np.r_[True,C[1:]!=C[:-1],True]))
    p=dict(objective="lambdarank",metric="map",eval_at=[5],learning_rate=0.03,num_leaves=31,
           n_estimators=800,min_child_samples=50,subsample=0.8,subsample_freq=1,
           colsample_bytree=0.8,random_state=lib.SEED,verbose=-1,n_jobs=4); p.update(kw)
    m=lgb.LGBMRanker(**p).fit(X,y,group=g)
    s=lib.evaluate(features.top5(m.predict(Xo),CUo,CIo,FB),GT)
    print(f"  {tag_name:<46} MAP@5 = {s:.5f}  [{time.time()-t0:.0f}s]", flush=True)
    return s,m
names=[t for t,_ in SPL]
run(["w0413"],"1 window, extended collab")
run(names[2:],"4 windows, extended collab")
run(names,"6 windows, extended collab")
