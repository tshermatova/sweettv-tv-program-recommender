import numpy as np, pickle, csv, itertools
from collections import Counter, defaultdict
W=ROOT + "work/"; D=ROOT + "data/"
A=pickle.load(open(W+"airings.pkl","rb")); AIR=A["airings"]; B_AB=A["B_AB"]
M=np.load(W+"matches.npy"); users=open(W+"users.txt").read().split("\n")
uidx={u:i for i,u in enumerate(users)}
test_ui=set(uidx[r[0]] for r in list(csv.reader(open(D+"submission.csv")))[1:])
ui,sid,ov,pdur,pst=M[:,0],M[:,1],M[:,2],M[:,3],M[:,4]
frac=ov/np.maximum(pdur,1); airB=AIR["B"]
KA={"pop":0.29788,"cnt":0.40024,"air":0.41510,"sec":0.29356,"nothr":0.37887}

fm=(frac>=0.20)&(pst<B_AB)
HA=defaultdict(Counter); HAs=defaultdict(Counter)
for u,s,o in zip(ui[fm],sid[fm],ov[fm]): HA[int(u)][int(s)]+=1; HAs[int(u)][int(s)]+=int(o)
fm0=(pst<B_AB); HA0=defaultdict(Counter)
for u,s in zip(ui[fm0],sid[fm0]): HA0[int(u)][int(s)]+=1
pop=Counter()
for u,c in HA.items():
    for s in c: pop[s]+=1
fb=[s for s,_ in pop.most_common(5)]
def apk(a,p,k=5):
    sc=0.;h=0
    for i,x in enumerate(p[:k]):
        if x in a: h+=1; sc+=h/(i+1)
    return sc/min(len(a),k)
def rank(h,w):
    if not h: return list(fb)
    c=[(n,s) for s,n in h.items()] if w=="count" else [(n*np.log1p(airB.get(s,0)),s) for s,n in h.items()]
    c.sort(key=lambda x:-x[0]); out=[s for _,s in c[:5]]
    for s in fb:
        if len(out)>=5: break
        if s not in out: out.append(s)
    return out[:5]

best=[]
for fthr in [0.5,0.7,0.8,0.9,0.95,1.0]:
    for minn in [1,2,3,4,5,6,8]:
        tm=(frac>=fthr)&(pst>=B_AB)
        HB=defaultdict(Counter)
        for u,s in zip(ui[tm],sid[tm]): HB[int(u)][int(s)]+=1
        gt={u:set(s for s,n in c.items() if n>=minn) for u,c in HB.items() if u in test_ui}
        gt={u:g for u,g in gt.items() if g}
        if len(gt)<900: continue
        r={"pop":np.mean([apk(g,fb) for g in gt.values()]),
           "cnt":np.mean([apk(g,rank(HA.get(u),"count")) for u,g in gt.items()]),
           "air":np.mean([apk(g,rank(HA.get(u),"air")) for u,g in gt.items()]),
           "sec":np.mean([apk(g,rank(HAs.get(u),"count")) for u,g in gt.items()]),
           "nothr":np.mean([apk(g,rank(HA0.get(u),"air")) for u,g in gt.items()])}
        order_ok = sorted(r,key=lambda k:-r[k])==sorted(KA,key=lambda k:-KA[k])
        mae=np.mean([abs(r[k]-KA[k]) for k in KA])
        med=int(np.median([len(g) for g in gt.values()]))
        best.append((mae,order_ok,fthr,minn,len(gt),med,r))
best.sort(key=lambda x:(not x[1], x[0]))
print(f"{'frac':>5} {'min_n':>5} {'users':>5} {'med|R|':>6} {'ord':>4} {'MAE':>7} | {'pop':>6} {'cnt':>6} {'air':>6} {'sec':>6} {'nothr':>6}")
print("-"*80)
for mae,ok,f_,n_,nu,med,r in best[:14]:
    print(f"{f_:>5} {n_:>5} {nu:>5} {med:>6} {'OK' if ok else '--':>4} {mae:>7.4f} | {r['pop']:>6.4f} {r['cnt']:>6.4f} {r['air']:>6.4f} {r['sec']:>6.4f} {r['nothr']:>6.4f}")
print("-"*80)
print(f"{'KAGGLE':>5} {'':>5} {1260:>5} {'?':>6} {'':>4} {'':>7} | {KA['pop']:>6.4f} {KA['cnt']:>6.4f} {KA['air']:>6.4f} {KA['sec']:>6.4f} {KA['nothr']:>6.4f}")
