"""FINAL: LR ranker over 17 features.
Fitted on the (window A -> window B) pair -- both inside the training weeks 11-30 --
then applied with the full weeks 11-30 history to rank candidates for weeks 31-42.
Week 31-42 viewing is never used; only that window's broadcast grid is."""
import os
import sys, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib, features
from sklearn.linear_model import LogisticRegression
t0=time.time()
T_END=int(np.load(lib.W+"airings_raw.npy")[-1,2])+1

# fit
hm_fit=features.history_arrays(lib.B_AB)
users_fit=sorted(set(int(v) for v in np.unique(lib.UI[hm_fit])))
blk_fit=features.collab_block(hm_fit)
Xf,CUf,CIf,yf,_=features.build(hm_fit,lib.B_AB,lib.B_TEST,users_fit,
    label_mask=(lib.FRAC>=1.0)&(lib.PST>=lib.B_AB)&(lib.PST<lib.B_TEST))
Xf=features.attach_collab(Xf,CUf,CIf,blk_fit)
mu,sd=Xf.mean(0),Xf.std(0)+1e-6
lr=LogisticRegression(max_iter=3000,random_state=lib.SEED).fit((Xf-mu)/sd,yf)
print(f"fit rows={len(Xf):,} pos={yf.mean():.4f} [{time.time()-t0:.0f}s]")
print("weights:",{c:round(float(w),3) for c,w in zip(features.FEATS,lr.coef_[0])})

# apply
hm_all=features.history_arrays(T_END)
blk_all=features.collab_block(hm_all)
Xt,CUt,CIt,_,aux=features.build(hm_all,lib.B_TEST,T_END,lib.TEST_UI)
Xt=features.attach_collab(Xt,CUt,CIt,blk_all)
FB=[int(lib.ITEMS_ALL[k]) for k in np.argsort(-aux["pop_u"])[:5]]
preds=features.top5(lr.decision_function((Xt-mu)/sd),CUt,CIt,FB)
p=lib.write_submission(preds,"sub08_final_lr17",FB)
print("wrote",p,f"[{time.time()-t0:.0f}s]")
import pickle; pickle.dump({"mu":mu,"sd":sd,"coef":lr.coef_[0],"int":lr.intercept_,
                            "feats":features.FEATS},open(lib.W+"final_model.pkl","wb"))
