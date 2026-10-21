import sys, time, numpy as np
sys.path.insert(0,"/Users/tshermatova/Downloads/week7-exam/src")
import lib, features, rerank
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
t0=time.time()
GT=lib.target(lo=lib.B_AB,hi=lib.B_TEST)

def within_user_rank(X, CU):
    """Replace each feature by its percentile rank inside the user's candidate list."""
    Z=np.empty_like(X)
    order=np.argsort(CU,kind="stable"); CUs=CU[order]
    bnd=np.flatnonzero(np.r_[True, CUs[1:]!=CUs[:-1], True])
    for a,b in zip(bnd[:-1],bnd[1:]):
        idx=order[a:b]; n=b-a
        for j in range(X.shape[1]):
            v=X[idx,j]
            r=np.empty(n); r[np.argsort(v,kind="stable")]=np.arange(n)
            Z[idx,j]=r/max(n-1,1)
    return Z

def add_interactions(X):
    pairs=[(0,9),(0,7),(1,9),(0,13),(0,14),(7,9)]
    extra=np.column_stack([X[:,a]*X[:,b] for a,b in pairs])
    return np.hstack([X,extra]).astype(np.float32)

# ---------- inner (fit) and outer (evaluate) blocks ----------
hm_in=features.history_arrays(rerank.INNER_SPLIT)
users_in=sorted(set(int(v) for v in np.unique(lib.UI[hm_in])))
blk_in=features.collab_block(hm_in)
hm_out=features.history_arrays(lib.B_AB)
blk_out=features.collab_block(hm_out)
Xo,CUo,CIo,_,aux=features.build(hm_out,lib.B_AB,lib.B_TEST,lib.TEST_UI)
Xo=features.attach_collab(Xo,CUo,CIo,blk_out)
FB=[int(lib.ITEMS_ALL[k]) for k in np.argsort(-aux["pop_u"])[:5]]
print(f"blocks ready [{time.time()-t0:.0f}s]")

Zo=within_user_rank(Xo,CUo); XoI=add_interactions(Xo); ZoI=add_interactions(Zo)
print(f"transforms ready [{time.time()-t0:.0f}s]")

print(f"\n{'fit label':>12} {'transform':>14} {'model':>6} {'MAP@5':>8}")
for lab_thr in [1.0, 0.5, 0.2]:
    Xi,CUi,CIi,yi,_=features.build(hm_in,rerank.INNER_SPLIT,lib.B_AB,users_in,
        label_mask=(lib.FRAC>=lab_thr)&(lib.PST>=rerank.INNER_SPLIT)&(lib.PST<lib.B_AB))
    Xi=features.attach_collab(Xi,CUi,CIi,blk_in)
    Zi=within_user_rank(Xi,CUi); XiI=add_interactions(Xi); ZiI=add_interactions(Zi)
    for tname,(A,B) in {"raw":(Xi,Xo),"within-user":(Zi,Zo),
                        "raw+inter":(XiI,XoI),"wu+inter":(ZiI,ZoI)}.items():
        mu,sd=A.mean(0),A.std(0)+1e-6
        lr=LogisticRegression(max_iter=3000,random_state=lib.SEED).fit((A-mu)/sd,yi)
        m=lib.evaluate(features.top5(lr.decision_function((B-mu)/sd),CUo,CIo,FB),GT)
        print(f"{lab_thr:>12} {tname:>14} {'LR':>6} {m:>8.5f}", flush=True)
    g=HistGradientBoostingClassifier(learning_rate=0.05,max_leaf_nodes=15,max_iter=200,
        random_state=lib.SEED,early_stopping=False).fit(Zi,yi)
    m=lib.evaluate(features.top5(g.predict_proba(Zo)[:,1],CUo,CIo,FB),GT)
    print(f"{lab_thr:>12} {'within-user':>14} {'GBM':>6} {m:>8.5f}", flush=True)
