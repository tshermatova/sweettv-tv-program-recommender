import sys, time, numpy as np, calendar
sys.path.insert(0,"/Users/tshermatova/Downloads/week7-exam/src")
import lib, features, features2
import lightgbm as lgb
t0=time.time()
def ts(y,m,d): return calendar.timegm((y,m,d,0,0,0,0,0,0))
GT=lib.target(lo=lib.B_AB,hi=lib.B_TEST)

def graded(lo,hi,cap=3,thr=1.0):
    m=(lib.FRAC>=thr)&(lib.PST>=lo)&(lib.PST<hi)
    u=lib.UI[m].astype(np.int64); s=lib.SIDX[m].astype(np.int64)
    k,c=np.unique(u*lib.N_ITEMS+s,return_counts=True)
    return dict(zip(k.tolist(),np.minimum(c,cap).tolist()))

def make(split, lo_lab, hi_lab, cap):
    hm=features.history_arrays(split)
    users=sorted(set(int(v) for v in np.unique(lib.UI[hm])))
    blk=features.collab_block(hm)
    X,CU,CI,_,_=features2.build2(hm,split,hi_lab,users,n_pop=80)
    X=features2.attach(X,CU,CI,blk)
    G=graded(lo_lab,hi_lab,cap)
    y=np.array([G.get(int(a)*lib.N_ITEMS+int(b),0) for a,b in zip(CU,CI)],np.int32)
    return X,CU,CI,y

hm_out=features.history_arrays(lib.B_AB); blk_out=features.collab_block(hm_out)
Xo,CUo,CIo,_,aux=features2.build2(hm_out,lib.B_AB,lib.B_TEST,lib.TEST_UI,n_pop=80)
Xo=features2.attach(Xo,CUo,CIo,blk_out)
FB=[int(lib.ITEMS_ALL[k]) for k in np.argsort(-aux["pop_u"])[:5]]
print(f"outer ready [{time.time()-t0:.0f}s]")

SPLITS=[(ts(2020,4,13),ts(2020,4,13),lib.B_AB),
        (ts(2020,3,30),ts(2020,3,30),ts(2020,5,4)),
        (ts(2020,4,27),ts(2020,4,27),lib.B_AB)]
cache={}
def fit_eval(splits,cap,tag,**kw):
    Xs=[];ys=[];Cs=[]
    for sp in splits:
        k=(sp,cap)
        if k not in cache: cache[k]=make(*sp,cap)
        X,CU,CI,y=cache[k]
        off=(max([c.max() for c in Cs])+1) if Cs else 0
        Xs.append(X); ys.append(y); Cs.append(CU+off)
    X=np.vstack(Xs); y=np.concatenate(ys); C=np.concatenate(Cs)
    o=np.argsort(C,kind="stable"); X,y,C=X[o],y[o],C[o]
    g=np.diff(np.flatnonzero(np.r_[True,C[1:]!=C[:-1],True]))
    p=dict(objective="lambdarank",metric="map",eval_at=[5],learning_rate=0.03,num_leaves=31,
           n_estimators=800,min_child_samples=50,subsample=0.8,subsample_freq=1,
           colsample_bytree=0.8,random_state=lib.SEED,verbose=-1,n_jobs=4); p.update(kw)
    m=lgb.LGBMRanker(**p).fit(X,y,group=g)
    s=lib.evaluate(features.top5(m.predict(Xo),CUo,CIo,FB),GT)
    print(f"  {tag:<44} MAP@5 = {s:.5f}  [{time.time()-t0:.0f}s]", flush=True)
    return s,m

fit_eval(SPLITS[:1],1,"binary label, 1 window (reference)")
fit_eval(SPLITS[:1],3,"graded label cap=3, 1 window")
fit_eval(SPLITS[:1],5,"graded label cap=5, 1 window")
fit_eval(SPLITS,1,"binary label, 3 windows")
fit_eval(SPLITS,3,"graded cap=3, 3 windows")
