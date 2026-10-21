import sys, time, numpy as np
sys.path.insert(0,"/Users/tshermatova/Downloads/week7-exam/src")
import lib, methods, rerank
from sklearn.linear_model import LogisticRegression
t0=time.time()

# ---- INNER: fit the ranker, entirely inside the training period ----
H_in  = lib.history(hi=rerank.INNER_SPLIT)
GT_in = lib.target(lo=rerank.INNER_SPLIT, hi=lib.B_AB)
users_in = sorted(H_in)
Xi, Mi, yi, _ = rerank.build(H_in, rerank.INNER_SPLIT, lib.B_AB, GT_in, users=users_in)
print(f"inner: rows={len(Xi):,} positives={yi.mean():.4f} users={len(users_in)} [{time.time()-t0:.0f}s]")
mu, sd = Xi.mean(0), Xi.std(0)+1e-6
clf=LogisticRegression(max_iter=2000, C=1.0, random_state=lib.SEED)
clf.fit((Xi-mu)/sd, yi)
print("weights:", {c: round(float(w),3) for c,w in zip(rerank.COLS, clf.coef_[0])})

# ---- OUTER: evaluate on the untouched hold-out ----
HA, GT = lib.val_split()
pop=lib.popularity(HA); FB=[s for s,_ in pop.most_common(5)]
Xo, Mo, _, pack = rerank.build(HA, lib.B_AB, lib.B_TEST)
print(f"outer: rows={len(Xo):,} [{time.time()-t0:.0f}s]")
sc = clf.decision_function((Xo-mu)/sd)
preds = rerank.top5_from_scores(sc, Mo, FB)
print(f"\nHYBRID (LR re-rank, fitted on inner split): MAP@5 = {lib.evaluate(preds,GT):.5f}")

# single-feature references on the same candidate pool
for i,c in enumerate(rerank.COLS):
    p=rerank.top5_from_scores(Xo[:,i], Mo, FB)
    print(f"   {lib.evaluate(p,GT):.5f}  feature-only: {c}")
np.save(lib.W+"hybrid_mu.npy", mu); np.save(lib.W+"hybrid_sd.npy", sd)
np.save(lib.W+"hybrid_coef.npy", clf.coef_[0]); np.save(lib.W+"hybrid_int.npy", clf.intercept_)
