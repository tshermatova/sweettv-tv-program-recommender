"""Does a wider candidate pool raise the achievable ceiling and the model?"""
import os
import sys, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib, features, features2
import lightgbm as lgb
t0=time.time()
GT=lib.target(lo=lib.B_AB,hi=lib.B_TEST)
LBL=lambda lo,hi,thr=1.0:(lib.FRAC>=thr)&(lib.PST>=lo)&(lib.PST<hi)
import rerank
hm_in=features.history_arrays(rerank.INNER_SPLIT); users_in=sorted(set(int(v) for v in np.unique(lib.UI[hm_in])))
blk_in=features.collab_block(hm_in)
hm_out=features.history_arrays(lib.B_AB); blk_out=features.collab_block(hm_out)
print(f"blocks [{time.time()-t0:.0f}s]")
for n_pop in [80,200,400]:
    Xi,CUi,CIi,yi,_=features2.build2(hm_in,rerank.INNER_SPLIT,lib.B_AB,users_in,n_pop=n_pop,label_mask=LBL(rerank.INNER_SPLIT,lib.B_AB))
    Xi=features2.attach(Xi,CUi,CIi,blk_in)
    Xo,CUo,CIo,yo,aux=features2.build2(hm_out,lib.B_AB,lib.B_TEST,lib.TEST_UI,n_pop=n_pop,label_mask=LBL(lib.B_AB,lib.B_TEST))
    Xo=features2.attach(Xo,CUo,CIo,blk_out)
    FB=[int(lib.ITEMS_ALL[k]) for k in np.argsort(-aux["pop_u"])[:5]]
    orc=lib.evaluate(features.top5(yo.astype(float)+1e-9*np.random.default_rng(0).random(len(yo)),CUo,CIo,FB),GT)
    o=np.argsort(CUi,kind="stable"); Xs,ys,Cs=Xi[o],yi[o],CUi[o]
    g=np.diff(np.flatnonzero(np.r_[True,Cs[1:]!=Cs[:-1],True]))
    m=lgb.LGBMRanker(objective="lambdarank",metric="map",eval_at=[5],learning_rate=0.03,
        num_leaves=31,n_estimators=800,min_child_samples=50,subsample=0.8,subsample_freq=1,
        colsample_bytree=0.8,random_state=lib.SEED,verbose=-1,n_jobs=4).fit(Xs,ys,group=g)
    sc=lib.evaluate(features.top5(m.predict(Xo),CUo,CIo,FB),GT)
    print(f"  n_pop={n_pop:>3}  rows={len(CUo):>7,}  oracle={orc:.5f}  LambdaRank={sc:.5f}  [{time.time()-t0:.0f}s]", flush=True)
