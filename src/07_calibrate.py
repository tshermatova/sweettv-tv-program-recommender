"""Find a hold-out label definition that reproduces the ratios observed on Kaggle.
Kaggle anchors (public): popularity 0.29788 | repeat-count 0.40024 | repeat x log(air) 0.41510
                         repeat-seconds 0.29356 | repeat no-threshold 0.37887"""
import numpy as np, pickle, csv
from collections import Counter, defaultdict
W="/Users/tshermatova/Downloads/week7-exam/work/"; D="/Users/tshermatova/Downloads/week7-exam/data/"
A=pickle.load(open(W+"airings.pkl","rb")); AIR=A["airings"]; B_AB=A["B_AB"]
M=np.load(W+"matches.npy"); users=open(W+"users.txt").read().split("\n")
uidx={u:i for i,u in enumerate(users)}
test_ui=set(uidx[r[0]] for r in list(csv.reader(open(D+"submission.csv")))[1:])
ui,sid,ov,pdur,pst = M[:,0],M[:,1],M[:,2],M[:,3],M[:,4]
frac=ov/np.maximum(pdur,1)
airB=AIR["B"]

FEAT_THR=0.20
featmask = (frac>=FEAT_THR)&(pst<B_AB)
HA=defaultdict(Counter)
for u,s,o in zip(ui[featmask],sid[featmask],ov[featmask]): HA[int(u)][int(s)]+=1
HA_sec=defaultdict(Counter)
for u,s,o in zip(ui[featmask],sid[featmask],ov[featmask]): HA_sec[int(u)][int(s)]+=int(o)
featmask0=(pst<B_AB)
HA0=defaultdict(Counter)
for u,s in zip(ui[featmask0],sid[featmask0]): HA0[int(u)][int(s)]+=1
pop=Counter()
for u,c in HA.items():
    for s in c: pop[s]+=1
fb=[s for s,_ in pop.most_common(5)]

def apk(act,pred,k=5):
    sc=0.;h=0
    for i,p in enumerate(pred[:k]):
        if p in act: h+=1; sc+=h/(i+1)
    return sc/min(len(act),k)

def rank(h, weight):
    if not h: return list(fb)
    if weight=="count":  c=[(n,s) for s,n in h.items()]
    elif weight=="air":  c=[(n*np.log1p(airB.get(s,0)),s) for s,n in h.items()]
    c.sort(key=lambda x:-x[0]); out=[s for _,s in c[:5]]
    for s in fb:
        if len(out)>=5: break
        if s not in out: out.append(s)
    return out[:5]

print(f"{'target definition':<34} {'users':>6} {'med|R|':>7} {'pop':>7} {'cnt':>7} {'air':>7} {'sec':>7} {'nothr':>7} {'air/pop':>8}")
print("-"*95)
for name, tmask in [
    ("frac>=0.2 (current)",      (frac>=0.20)&(pst>=B_AB)),
    ("frac>=0.5",                (frac>=0.50)&(pst>=B_AB)),
    ("frac>=0.8",                (frac>=0.80)&(pst>=B_AB)),
    ("frac>=0.2 & overlap>=600", (frac>=0.20)&(ov>=600)&(pst>=B_AB)),
    ("overlap>=1200s",           (ov>=1200)&(pst>=B_AB)),
    ("no threshold",             (pst>=B_AB)),
]:
    HB=defaultdict(Counter)
    for u,s in zip(ui[tmask],sid[tmask]): HB[int(u)][int(s)]+=1
    for MINR, tag in [(1,""),(2," & seen>=2x")]:
        gt={u:set(s for s,n in c.items() if n>=MINR) for u,c in HB.items() if u in test_ui}
        gt={u:g for u,g in gt.items() if g}
        if not gt: continue
        med=int(np.median([len(g) for g in gt.values()]))
        sp=np.mean([apk(g,fb) for g in gt.values()])
        sc=np.mean([apk(g,rank(HA.get(u),"count")) for u,g in gt.items()])
        sa=np.mean([apk(g,rank(HA.get(u),"air"))   for u,g in gt.items()])
        ss=np.mean([apk(g,rank(HA_sec.get(u),"count")) for u,g in gt.items()])
        s0=np.mean([apk(g,rank(HA0.get(u),"air"))  for u,g in gt.items()])
        print(f"{name+tag:<34} {len(gt):>6} {med:>7} {sp:>7.4f} {sc:>7.4f} {sa:>7.4f} {ss:>7.4f} {s0:>7.4f} {sa/sp:>8.3f}")
print("-"*95)
print(f"{'KAGGLE (public, window T)':<34} {1260:>6} {'?':>7} {0.29788:>7.4f} {0.40024:>7.4f} {0.41510:>7.4f} {0.29356:>7.4f} {0.37887:>7.4f} {0.41510/0.29788:>8.3f}")
