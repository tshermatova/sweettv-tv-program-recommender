import sys, time, numpy as np
sys.path.insert(0,"/Users/tshermatova/Downloads/week7-exam/src")
import lib, features, features2, rerank
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
t0=time.time()
GT=lib.target(lo=lib.B_AB,hi=lib.B_TEST)
LBL=lambda lo,hi,thr=1.0:(lib.FRAC>=thr)&(lib.PST>=lo)&(lib.PST<hi)

hm_in=features.history_arrays(rerank.INNER_SPLIT)
users_in=sorted(set(int(v) for v in np.unique(lib.UI[hm_in])))
blk_in=features.collab_block(hm_in)
Xi,CUi,CIi,yi,_=features2.build2(hm_in,rerank.INNER_SPLIT,lib.B_AB,users_in,label_mask=LBL(rerank.INNER_SPLIT,lib.B_AB))
Xi=features2.attach(Xi,CUi,CIi,blk_in)
hm_out=features.history_arrays(lib.B_AB)
blk_out=features.collab_block(hm_out)
Xo,CUo,CIo,_,aux=features2.build2(hm_out,lib.B_AB,lib.B_TEST,lib.TEST_UI)
Xo=features2.attach(Xo,CUo,CIo,blk_out)
FB=[int(lib.ITEMS_ALL[k]) for k in np.argsort(-aux["pop_u"])[:5]]
print(f"inner {Xi.shape} pos={yi.mean():.4f} | outer {Xo.shape}  [{time.time()-t0:.0f}s]")

mu,sd=Xi.mean(0),Xi.std(0)+1e-6
lr=LogisticRegression(max_iter=4000,random_state=lib.SEED).fit((Xi-mu)/sd,yi)
print(f"  LR  30 feats                MAP@5 = {lib.evaluate(features.top5(lr.decision_function((Xo-mu)/sd),CUo,CIo,FB),GT):.5f}")

oi=np.argsort(CUi,kind="stable"); Xi_s,yi_s,CUi_s=Xi[oi],yi[oi],CUi[oi]
gi=np.diff(np.flatnonzero(np.r_[True,CUi_s[1:]!=CUi_s[:-1],True]))
for lr_,leaves,n in [(0.05,31,400),(0.05,63,600),(0.03,31,800),(0.1,15,300)]:
    m=lgb.LGBMRanker(objective="lambdarank",metric="map",eval_at=[5],
        learning_rate=lr_,num_leaves=leaves,n_estimators=n,min_child_samples=50,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.8,
        random_state=lib.SEED,verbose=-1,n_jobs=4)
    m.fit(Xi_s,yi_s,group=gi)
    sc=m.predict(Xo)
    print(f"  LGBMRanker lr={lr_} leaves={leaves} n={n}   MAP@5 = {lib.evaluate(features.top5(sc,CUo,CIo,FB),GT):.5f}  [{time.time()-t0:.0f}s]", flush=True)
