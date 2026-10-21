import sys, time, numpy as np, pickle
sys.path.insert(0,"/Users/tshermatova/Downloads/week7-exam/src")
import lib, features, features2
import lightgbm as lgb
t0=time.time()
T_END=int(np.load(lib.W+"airings_raw.npy")[-1,2])+1
LBL=lambda lo,hi,thr=1.0:(lib.FRAC>=thr)&(lib.PST>=lo)&(lib.PST<hi)

hm_fit=features.history_arrays(lib.B_AB)
users_fit=sorted(set(int(v) for v in np.unique(lib.UI[hm_fit])))
blk_fit=features.collab_block(hm_fit)
Xf,CUf,CIf,yf,_=features2.build2(hm_fit,lib.B_AB,lib.B_TEST,users_fit,label_mask=LBL(lib.B_AB,lib.B_TEST))
Xf=features2.attach(Xf,CUf,CIf,blk_fit)
o=np.argsort(CUf,kind="stable"); Xf,yf,CUf=Xf[o],yf[o],CUf[o]
g=np.diff(np.flatnonzero(np.r_[True,CUf[1:]!=CUf[:-1],True]))
print(f"fit {Xf.shape} pos={yf.mean():.4f} groups={len(g)} [{time.time()-t0:.0f}s]")
mdl=lgb.LGBMRanker(objective="lambdarank",metric="map",eval_at=[5],learning_rate=0.03,
    num_leaves=31,n_estimators=800,min_child_samples=50,subsample=0.8,subsample_freq=1,
    colsample_bytree=0.8,random_state=lib.SEED,verbose=-1,n_jobs=4).fit(Xf,yf,group=g)

hm_all=features.history_arrays(T_END)
blk_all=features.collab_block(hm_all)
Xt,CUt,CIt,_,aux=features2.build2(hm_all,lib.B_TEST,T_END,lib.TEST_UI)
Xt=features2.attach(Xt,CUt,CIt,blk_all)
FB=[int(lib.ITEMS_ALL[k]) for k in np.argsort(-aux["pop_u"])[:5]]
preds=features.top5(mdl.predict(Xt),CUt,CIt,FB)
p=lib.write_submission(preds,"sub09_lambdarank",FB)
print("wrote",p,f"[{time.time()-t0:.0f}s]")
imp=sorted(zip(features2.FEATS2,mdl.feature_importances_),key=lambda x:-x[1])
print("top features:", [(k,int(v)) for k,v in imp[:12]])
pickle.dump(mdl,open(lib.W+"lgb_final.pkl","wb"))
