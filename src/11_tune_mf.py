import sys, time, numpy as np
sys.path.insert(0,"/Users/tshermatova/Downloads/week7-exam/src")
import lib, methods
HA,GT=lib.val_split()
R,ridx,items,iidx=methods.matrix(HA)
pop=lib.popularity(HA); FB=[s for s,_ in pop.most_common(5)]
items_arr=np.array(items)
def ev(Sc):
    preds={}
    for u in lib.TEST_UI:
        r=ridx.get(u)
        if r is None: preds[u]=list(FB); continue
        s=Sc[r]; idx=np.argpartition(-s,5)[:5]; idx=idx[np.argsort(-s[idx])]
        preds[u]=[int(items_arr[i]) for i in idx]
    return lib.evaluate(preds,GT)
print(f"{'factors':>7} {'reg':>7} {'alpha':>6} {'iters':>5} {'MAP@5':>8} {'fit s':>6}")
for f,reg,al,it in [(64,0.05,20,8),(64,1.0,20,8),(64,10.0,20,8),(64,100.0,20,8),
                    (32,10.0,20,8),(128,10.0,20,8),(64,10.0,5,8),(64,10.0,40,8),
                    (64,30.0,20,8),(64,10.0,20,15)]:
    t=time.time(); X,Y=methods.ials(R,factors=f,reg=reg,alpha=al,iters=it)
    print(f"{f:>7} {reg:>7} {al:>6} {it:>5} {ev(X@Y.T):>8.5f} {time.time()-t:>6.0f}", flush=True)
