import numpy as np, pickle, csv
from collections import Counter, defaultdict
W=ROOT + "work/"; D=ROOT + "data/"
A=pickle.load(open(W+"airings.pkl","rb")); AIR=A["airings"]; B_AB=A["B_AB"]
M=np.load(W+"matches.npy")
users=open(W+"users.txt").read().split("\n")
uidx={u:i for i,u in enumerate(users)}
test_users=[r[0] for r in list(csv.reader(open(D+"submission.csv")))[1:]]
test_ui=np.array([uidx[u] for u in test_users])

ui,sid,ov,pdur,pst = M[:,0],M[:,1],M[:,2],M[:,3],M[:,4]
frac = ov/np.maximum(pdur,1)
THR = 0.20
keep = frac>=THR
inA = keep & (pst <  B_AB)
inB = keep & (pst >= B_AB)
print(f"threshold frac>={THR}: events A={inA.sum():,}  B={inB.sum():,}")

def pairs(mask):
    d=defaultdict(Counter)
    for u,s in zip(ui[mask], sid[mask]): d[int(u)][int(s)]+=1
    return d
HA=pairs(inA)   # history in window A
HB=pairs(inB)   # ground truth in window B

tset=set(test_ui.tolist())
gt={u:set(HB[u]) for u in tset if u in HB}
print(f"test users with >=1 relevant item in B: {len(gt)}/{len(test_ui)}")
sizes=np.array([len(v) for v in gt.values()])
print("relevant items per test user: median %d  mean %.1f  p10 %d  p90 %d  min %d" %
      (np.median(sizes), sizes.mean(), np.percentile(sizes,10), np.percentile(sizes,90), sizes.min()))
print("share of users with >=5 relevant: %.3f" % (sizes>=5).mean())

# repeat rate: of items watched in B, how many were already watched in A
rep=[]; 
for u,g in gt.items():
    h=set(HA.get(u,{}))
    rep.append(len(g&h)/len(g))
print("share of B-items already seen in A: mean %.3f  median %.3f" % (np.mean(rep), np.median(rep)))
cov=[1.0 if (set(HA.get(u,{})) & g) else 0.0 for u,g in gt.items()]
print("users with >=1 repeat: %.3f" % np.mean(cov))

def apk(actual, pred, k=5):
    score=0.0; hits=0
    for i,p in enumerate(pred[:k]):
        if p in actual:
            hits+=1; score+=hits/(i+1)
    return score/min(len(actual),k)
def mapk(preds,k=5):
    return float(np.mean([apk(gt[u], preds[u], k) for u in gt]))

airB=AIR["B"]; airB_set=set(airB)
pop_all=Counter()
for u,c in HA.items():
    for s,n in c.items(): pop_all[s]+=1          # user-count popularity
pop_air=Counter({s:c for s,c in pop_all.items() if s in airB_set})

top_pop      = [s for s,_ in pop_all.most_common(5)]
top_pop_air  = [s for s,_ in pop_air.most_common(5)]
top_airings  = [s for s,_ in airB.most_common(5)]
top_mix      = [s for s,_ in Counter({s: pop_all.get(s,0)*np.log1p(airB[s]) for s in airB_set}).most_common(5)]

res={}
res["popularity (train only)"]        = mapk({u:top_pop for u in gt})
res["popularity ∩ airs in B"]         = mapk({u:top_pop_air for u in gt})
res["most airings in B"]              = mapk({u:top_airings for u in gt})
res["pop x log(airings)"]             = mapk({u:top_mix for u in gt})

def user_repeat(u, fallback):
    h=HA.get(u)
    if not h: return fallback
    cand=[(n,s) for s,n in h.items() if s in airB_set]
    cand.sort(key=lambda x:(-x[0], -airB.get(x[1],0)))
    out=[s for _,s in cand[:5]]
    for s in fallback:
        if len(out)>=5: break
        if s not in out: out.append(s)
    return out[:5]
res["own repeats (∩ airs in B) + pop fallback"] = mapk({u:user_repeat(u, top_pop_air) for u in gt})

def user_repeat_air(u, fallback):
    h=HA.get(u)
    if not h: return fallback
    cand=[(n*np.log1p(airB.get(s,0)),s) for s,n in h.items() if s in airB_set]
    cand.sort(key=lambda x:-x[0])
    out=[s for _,s in cand[:5]]
    for s in fallback:
        if len(out)>=5: break
        if s not in out: out.append(s)
    return out[:5]
res["own repeats x log(airings) + pop fallback"] = mapk({u:user_repeat_air(u, top_pop_air) for u in gt})

print("\n================ MAP@5 on hold-out (window B) ================")
for k,v in sorted(res.items(), key=lambda x:-x[1]):
    print(f"  {v:.5f}   {k}")
