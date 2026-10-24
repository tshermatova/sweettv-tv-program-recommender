"""Final model: ranker fitted on the (window A -> window B) pair, both inside the
training weeks 11-30; applied with the full 11-30 history to predict weeks 31-42."""
import os
import sys, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib, methods, rerank
from sklearn.linear_model import LogisticRegression
t0=time.time()
HA = lib.history(hi=lib.B_AB)
GT_B = lib.target(lo=lib.B_AB, hi=lib.B_TEST)
users_fit=sorted(HA)
Xf, Mf, yf, _ = rerank.build(HA, lib.B_AB, lib.B_TEST, GT_B, users=users_fit)
print(f"fit rows={len(Xf):,} pos={yf.mean():.4f} [{time.time()-t0:.0f}s]")
mu,sd=Xf.mean(0),Xf.std(0)+1e-6
clf=LogisticRegression(max_iter=2000,C=1.0,random_state=lib.SEED).fit((Xf-mu)/sd,yf)
print("weights:",{c:round(float(w),3) for c,w in zip(rerank.COLS,clf.coef_[0])})

H_all = lib.final_history()
pop=lib.popularity(H_all); FB=[s for s,_ in pop.most_common(5)]
T_END = int(np.load(lib.W+"airings_raw.npy")[-1,2])+1
Xt, Mt, _, _ = rerank.build(H_all, lib.B_TEST, T_END)
print(f"test rows={len(Xt):,} [{time.time()-t0:.0f}s]")
preds=rerank.top5_from_scores(clf.decision_function((Xt-mu)/sd), Mt, FB)
p=lib.write_submission(preds,"sub07_hybrid_lr",FB)
print("wrote",p)
