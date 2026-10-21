import sys, time, numpy as np
sys.path.insert(0,"/Users/tshermatova/Downloads/week7-exam/src")
import lib, features, rerank
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
t0=time.time()
INNER=rerank.INNER_SPLIT

def lbl(lo,hi,thr=1.0): return (lib.FRAC>=thr)&(lib.PST>=lo)&(lib.PST<hi)

# INNER: fit
hm_in=features.history_arrays(INNER)
users_in=sorted(set(int(x) for x in np.unique(lib.UI[hm_in])))
blk_in=features.collab_block(hm_in)
Xi,CUi,CIi,yi,_=features.build(hm_in,INNER,lib.B_AB,users_in,label_mask=lbl(INNER,lib.B_AB))
Xi=features.attach_collab(Xi,CUi,CIi,blk_in)
print(f"inner rows={len(Xi):,} pos={yi.mean():.4f} [{time.time()-t0:.0f}s]")

# OUTER: evaluate
hm_out=features.history_arrays(lib.B_AB)
blk_out=features.collab_block(hm_out)
Xo,CUo,CIo,_,aux=features.build(hm_out,lib.B_AB,lib.B_TEST,lib.TEST_UI)
Xo=features.attach_collab(Xo,CUo,CIo,blk_out)
GT=lib.target(lo=lib.B_AB,hi=lib.B_TEST)
FB=[int(lib.ITEMS_ALL[k]) for k in np.argsort(-aux["pop_u"])[:5]]
print(f"outer rows={len(Xo):,} [{time.time()-t0:.0f}s]")

mu,sd=Xi.mean(0),Xi.std(0)+1e-6
lr=LogisticRegression(max_iter=3000,random_state=lib.SEED).fit((Xi-mu)/sd,yi)
p=features.top5(lr.decision_function((Xo-mu)/sd),CUo,CIo,FB)
print(f"\n  LR  (17 features)        MAP@5 = {lib.evaluate(p,GT):.5f}")

for lr_,leaves,iters in [(0.05,31,300),(0.05,63,500),(0.1,31,200)]:
    g=HistGradientBoostingClassifier(learning_rate=lr_,max_leaf_nodes=leaves,
        max_iter=iters,random_state=lib.SEED,early_stopping=False).fit(Xi,yi)
    p=features.top5(g.predict_proba(Xo)[:,1],CUo,CIo,FB)
    print(f"  GBM lr={lr_} leaves={leaves} it={iters}  MAP@5 = {lib.evaluate(p,GT):.5f}  [{time.time()-t0:.0f}s]")

print("\n  single-feature reference on same pool:")
for i,c in enumerate(features.FEATS):
    print(f"    {lib.evaluate(features.top5(Xo[:,i],CUo,CIo,FB),GT):.5f}  {c}")
np.save(lib.W+"Xi.npy",Xi); np.save(lib.W+"yi.npy",yi)
