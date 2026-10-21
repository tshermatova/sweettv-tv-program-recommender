import sys, time, numpy as np
sys.path.insert(0,"/Users/tshermatova/Downloads/week7-exam/src")
import lib, features, rerank
from sklearn.linear_model import LogisticRegression
t0=time.time()
GT=lib.target(lo=lib.B_AB,hi=lib.B_TEST)
hm_in=features.history_arrays(rerank.INNER_SPLIT)
users_in=sorted(set(int(v) for v in np.unique(lib.UI[hm_in])))
blk_in=features.collab_block(hm_in)
Xi,CUi,CIi,yi,_=features.build(hm_in,rerank.INNER_SPLIT,lib.B_AB,users_in,
    label_mask=(lib.FRAC>=1.0)&(lib.PST>=rerank.INNER_SPLIT)&(lib.PST<lib.B_AB))
Xi=features.attach_collab(Xi,CUi,CIi,blk_in)
hm_out=features.history_arrays(lib.B_AB)
blk_out=features.collab_block(hm_out)
Xo,CUo,CIo,_,aux=features.build(hm_out,lib.B_AB,lib.B_TEST,lib.TEST_UI)
Xo=features.attach_collab(Xo,CUo,CIo,blk_out)
FB=[int(lib.ITEMS_ALL[k]) for k in np.argsort(-aux["pop_u"])[:5]]
mu,sd=Xi.mean(0),Xi.std(0)+1e-6
lr=LogisticRegression(max_iter=3000,random_state=lib.SEED).fit((Xi-mu)/sd,yi)
s_lr=lr.decision_function((Xo-mu)/sd)
s_rep=Xo[:,0]*Xo[:,9]          # own_log x air_log  (the 0.415 Kaggle model)
print(f"ready [{time.time()-t0:.0f}s]")

def norm_rank(s,CU):
    z=np.empty(len(s))
    order=np.argsort(CU,kind="stable"); CUs=CU[order]
    bnd=np.flatnonzero(np.r_[True,CUs[1:]!=CUs[:-1],True])
    for a,b in zip(bnd[:-1],bnd[1:]):
        idx=order[a:b]; n=b-a
        r=np.empty(n); r[np.argsort(-s[idx],kind="stable")]=np.arange(n)
        z[idx]=1.0/(60.0+r)     # reciprocal-rank fusion
    return z
R_lr=norm_rank(s_lr,CUo); R_rep=norm_rank(s_rep,CUo)
print(f"  LR alone            {lib.evaluate(features.top5(s_lr,CUo,CIo,FB),GT):.5f}")
print(f"  repeat x air alone  {lib.evaluate(features.top5(s_rep,CUo,CIo,FB),GT):.5f}")
for w in [0.2,0.3,0.4,0.5,0.6,0.7,0.8]:
    m=lib.evaluate(features.top5(w*R_lr+(1-w)*R_rep,CUo,CIo,FB),GT)
    print(f"  RRF  w_LR={w:.1f}       {m:.5f}")
np.save(lib.W+"lr_coef.npy",lr.coef_[0]); np.save(lib.W+"lr_mu.npy",mu); np.save(lib.W+"lr_sd.npy",sd)
