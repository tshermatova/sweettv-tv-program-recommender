"""Section 7: two chosen users, every method's Top-5 side by side, with titles."""
import sys, numpy as np, pickle
sys.path.insert(0,"/Users/tshermatova/Downloads/week7-exam/src")
import lib, features, features2, methods
import lightgbm as lgb
from collections import Counter

def label(s):
    m=lib.SHOW_META.get(int(s))
    if not m: return f"id={s}  (no schedule metadata)"
    cat=m[1] or "-"
    return f"id={s:<8} {m[0][:46]:<46} [{cat}]"

hm=features.history_arrays(lib.B_AB)
GT=lib.target(lo=lib.B_AB,hi=lib.B_TEST)
cnt=Counter(int(x) for x in lib.UI[hm])
elig=[(cnt.get(u,0),u) for u in lib.TEST_UI if GT.get(u)]
elig.sort()
LIGHT=[u for n,u in elig if n>0][0]
HEAVY=elig[-1][1]
print(f"HEAVY user_idx={HEAVY} id={lib.USERS[HEAVY]}  history events={cnt[HEAVY]}  |relevant|={len(GT[HEAVY])}")
print(f"LIGHT user_idx={LIGHT} id={lib.USERS[LIGHT]}  history events={cnt[LIGHT]}  |relevant|={len(GT[LIGHT])}")

blk=features.collab_block(hm)
out,ru,ci,R=blk
items=[int(lib.ITEMS_ALL[k]) for k in sorted(ci,key=ci.get)]
items_arr=np.array(items)
pop=lib.popularity(lib.history(hi=lib.B_AB))
FB=[s for s,_ in pop.most_common(5)]
air,_,_=features.airing_stats(lib.B_AB,lib.B_TEST)

X,CU,CI,_,aux=features2.build2(hm,lib.B_AB,lib.B_TEST,lib.TEST_UI,n_pop=80)
X=features2.attach(X,CU,CI,blk)
mdl=pickle.load(open(lib.W+"lgb_holdout.pkl","rb")) if False else None

def top5_from(vec, pool_items):
    idx=np.argpartition(-vec,5)[:5]; idx=idx[np.argsort(-vec[idx])]
    return [int(pool_items[i]) for i in idx]

for tag,u in [("HEAVY",HEAVY),("LIGHT",LIGHT)]:
    print("\n"+"="*100); print(f"{tag}  user_id={lib.USERS[u]}   relevant programmes in the hold-out window: {len(GT[u])}")
    print("="*100)
    r=ru.get(u)
    hist=Counter()
    m=hm&(lib.UI==u)
    for s in lib.SIDX[m]: hist[int(lib.ITEMS_ALL[s])]+=1
    print("\nTop of the user's actual training history (watch count):")
    for s,n in hist.most_common(6): print(f"   {n:>4}x  {label(s)}")
    lists={}
    lists["popularity"]=FB
    own=[(n*np.log1p(air[lib.IIDX_ALL[s]]),s) for s,n in hist.items()]
    own.sort(key=lambda x:-x[0]); lists["repeat x airings"]=[s for _,s in own[:5]]
    if r is not None:
        for k,name in [("mf","MF (iALS)"),("cbf","content-based"),("assoc","association rules"),("ppr","PageRank (PPR)")]:
            lists[name]=top5_from(out[k][r],items_arr)
    for name,L in lists.items():
        print(f"\n   {name}:")
        for i,s in enumerate(L,1):
            hit="HIT " if s in GT[u] else "miss"
            print(f"      {i}. [{hit}] {label(s)}")
