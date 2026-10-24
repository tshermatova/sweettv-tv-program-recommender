import os
import sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib, features
from collections import Counter
GT=lib.target(lo=lib.B_AB,hi=lib.B_TEST)
hm=features.history_arrays(lib.B_AB)
X,CU,CI,y,aux=features.build(hm,lib.B_AB,lib.B_TEST,lib.TEST_UI,
                             label_mask=(lib.FRAC>=1.0)&(lib.PST>=lib.B_AB)&(lib.PST<lib.B_TEST))
FB=[int(lib.ITEMS_ALL[k]) for k in np.argsort(-aux["pop_u"])[:5]]
print("candidate rows: %,d" .replace("%,d",f"{len(CU):,}"))

# oracle: rank candidates by the true label
orc=features.top5(y.astype(np.float64)+1e-9*np.random.default_rng(0).random(len(y)),CU,CI,FB)
print(f"ORACLE within the candidate pool          MAP@5 = {lib.evaluate(orc,GT):.5f}")

# candidate recall: how much of each user's relevant set is reachable at all
cand_by_u={}
for a,b in zip(CU,CI): cand_by_u.setdefault(int(a),set()).add(int(lib.ITEMS_ALL[b]))
rec=[]; rec5=[]
for u in lib.TEST_UI:
    g=GT.get(u)
    if not g: continue
    c=cand_by_u.get(u,set())
    rec.append(len(g&c)/len(g)); rec5.append(min(len(g&c),5)/min(len(g),5))
print(f"candidate recall: mean {np.mean(rec):.4f}  | recall of the best-5 achievable: {np.mean(rec5):.4f}")

# where our current model stands
from sklearn.linear_model import LogisticRegression
import rerank
hm_in=features.history_arrays(rerank.INNER_SPLIT)
users_in=sorted(set(int(v) for v in np.unique(lib.UI[hm_in])))
Xi,CUi,CIi,yi,_=features.build(hm_in,rerank.INNER_SPLIT,lib.B_AB,users_in,
      label_mask=(lib.FRAC>=1.0)&(lib.PST>=rerank.INNER_SPLIT)&(lib.PST<lib.B_AB))
mu,sd=Xi.mean(0),Xi.std(0)+1e-6
lr=LogisticRegression(max_iter=3000,random_state=lib.SEED).fit((Xi-mu)/sd,yi)
p=features.top5(lr.decision_function((X-mu)/sd),CU,CI,FB)
m=lib.evaluate(p,GT); print(f"current LR hybrid (13 collab-free feats)  MAP@5 = {m:.5f}")

aps=[]; hits=[]
for u in lib.TEST_UI:
    g=GT.get(u)
    if not g: aps.append(0.0); hits.append(0); continue
    pr=p[u]; aps.append(lib.apk(g,pr)); hits.append(sum(1 for x in pr if x in g))
aps=np.array(aps); hits=np.array(hits)
print("AP@5 distribution:", {f"p{q}":round(float(np.percentile(aps,q)),3) for q in [10,25,50,75,90]})
print("hits@5 histogram:", dict(Counter(hits.tolist())))
print("share of users with 0 hits: %.4f" % (hits==0).mean())
