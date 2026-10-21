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
best=(0,None)
print(f"{'factors':>7} {'reg':>8} {'alpha':>6} {'MAP@5':>8} {'fit s':>6}")
for f,reg,al in [(64,100.,5),(64,300.,20),(64,1000.,20),(64,300.,5),(32,100.,5),
                 (16,100.,5),(32,300.,20),(64,100.,1),(64,100.,2),(16,300.,5),
                 (32,1000.,20),(64,3000.,20)]:
    t=time.time(); X,Y=methods.ials(R,factors=f,reg=reg,alpha=al,iters=8)
    m=ev(X@Y.T)
    if m>best[0]: best=(m,(f,reg,al))
    print(f"{f:>7} {reg:>8} {al:>6} {m:>8.5f} {time.time()-t:>6.0f}", flush=True)
print("BEST", best)
