"""FINAL MODEL — the configuration that won the hold-out comparison.

LGBMRanker (lambdarank), 30 features, binary labels, THREE training windows.
The three windows are the hold-out's winning geometry shifted forward by exactly
70 days (= B_TEST - B_AB), so the final fit sees the same window shapes, only later:

   hold-out                            final
   hist < 03-30, labels 03-30..05-04   hist < 06-08, labels 06-08..07-13
   hist < 04-13, labels 04-13..05-18   hist < 06-22, labels 06-22..07-27
   hist < 04-27, labels 04-27..05-18   hist < 07-06, labels 07-06..07-27

Everything stays inside the training weeks 11-30. Weeks 31-42 contribute only the
broadcast grid (which programme airs when), never viewing behaviour.
"""
import sys, time, calendar, numpy as np, pickle, gc
sys.path.insert(0,"/Users/tshermatova/Downloads/week7-exam/src")
import lib, features, features2
import lightgbm as lgb
t0=time.time()
def ts(y,m,d): return calendar.timegm((y,m,d,0,0,0,0,0,0))
T_END=int(np.load(lib.W+"airings_raw.npy")[-1,2])+1
WIN=[(ts(2020,6,8), ts(2020,7,13)), (ts(2020,6,22), lib.B_TEST), (ts(2020,7,6), lib.B_TEST)]

Xs=[];ys=[];Cs=[];off=0
for split,lab_hi in WIN:
    hm=features.history_arrays(split)
    users=sorted(set(int(v) for v in np.unique(lib.UI[hm])))
    blk=features.collab_block(hm)
    X,CU,CI,y,_=features2.build2(hm,split,lab_hi,users,n_pop=80,
        label_mask=(lib.FRAC>=1.0)&(lib.PST>=split)&(lib.PST<lab_hi))
    X=features2.attach(X,CU,CI,blk); del blk; gc.collect()
    Xs.append(X); ys.append(y); Cs.append(CU+off); off+=lib.N_USERS
    print(f"  window {split}: {X.shape} pos={y.mean():.4f} [{time.time()-t0:.0f}s]", flush=True)
X=np.vstack(Xs); y=np.concatenate(ys); C=np.concatenate(Cs); del Xs,ys,Cs; gc.collect()
o=np.argsort(C,kind="stable"); X,y,C=X[o],y[o],C[o]
g=np.diff(np.flatnonzero(np.r_[True,C[1:]!=C[:-1],True]))
mdl=lgb.LGBMRanker(objective="lambdarank",metric="map",eval_at=[5],learning_rate=0.03,
    num_leaves=31,n_estimators=800,min_child_samples=50,subsample=0.8,subsample_freq=1,
    colsample_bytree=0.8,random_state=lib.SEED,verbose=-1,n_jobs=4).fit(X,y,group=g)
print(f"fitted on {X.shape[0]:,} rows / {len(g):,} groups [{time.time()-t0:.0f}s]")
del X,y,C; gc.collect()

hm_all=features.history_arrays(T_END)
blk_all=features.collab_block(hm_all)
Xt,CUt,CIt,_,aux=features2.build2(hm_all,lib.B_TEST,T_END,lib.TEST_UI,n_pop=80)
Xt=features2.attach(Xt,CUt,CIt,blk_all)
FB=[int(lib.ITEMS_ALL[k]) for k in np.argsort(-aux["pop_u"])[:5]]
preds=features.top5(mdl.predict(Xt),CUt,CIt,FB)
p=lib.write_submission(preds,"sub10_final_3win",FB)
print("wrote",p,f"[{time.time()-t0:.0f}s]")
imp=sorted(zip(features2.FEATS2,mdl.feature_importances_),key=lambda x:-x[1])
print("top features:",[(k,int(v)) for k,v in imp[:10]])
pickle.dump(mdl,open(lib.W+"final_3win.pkl","wb"))
