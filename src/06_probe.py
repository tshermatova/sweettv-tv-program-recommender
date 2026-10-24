import numpy as np, pickle, csv, os, sys
from collections import Counter, defaultdict
import os
# Repository root, derived from this file's own location so the code runs from any checkout.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
W=ROOT + "work/"; D=ROOT + "data/"
S=ROOT + "submissions/"
A=pickle.load(open(W+"airings.pkl","rb")); airT=A["airings"]["T"]; airT_set=set(airT)
M=np.load(W+"matches.npy"); users=open(W+"users.txt").read().split("\n")
uidx={u:i for i,u in enumerate(users)}
test_users=[r[0] for r in list(csv.reader(open(D+"submission.csv")))[1:]]
ui,sid,ov,pdur = M[:,0],M[:,1],M[:,2],M[:,3]
frac = ov/np.maximum(pdur,1)

def build(name, thr, mode, restrict=True):
    keep = frac>=thr
    H=defaultdict(Counter)
    for u,s,o in zip(ui[keep], sid[keep], ov[keep]): H[int(u)][int(s)]+= (1 if mode!="sec" else int(o))
    pop=Counter()
    for u,c in H.items():
        for s in c: pop[s]+=1
    pool = airT_set if restrict else set(pop)
    fb=[s for s,_ in Counter({s:c for s,c in pop.items() if s in pool}).most_common(5)]
    rows=[]
    for u in test_users:
        if mode=="poponly":
            out=list(fb)
        else:
            h=H.get(uidx[u]); out=[]
            if h:
                if mode=="count": cand=[(n,s) for s,n in h.items() if s in pool]
                elif mode=="sec": cand=[(n,s) for s,n in h.items() if s in pool]
                else:             cand=[(n*np.log1p(airT.get(s,0)),s) for s,n in h.items() if s in pool]
                cand.sort(key=lambda x:-x[0]); out=[s for _,s in cand[:5]]
            for s in fb:
                if len(out)>=5: break
                if s not in out: out.append(s)
        rows.append((u," ".join(map(str,out[:5]))))
    p=S+name+".csv"
    with open(p,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["user_id","tv_show_id"]); w.writerows(rows)
    return p

jobs=[("sub02_pop_only",      0.20,"poponly",True),
      ("sub03_repeat_count",  0.20,"count",  True),
      ("sub04_repeat_nothr",  0.00,"airmix", True),
      ("sub05_repeat_seconds",0.20,"sec",    True),
      ("sub06_repeat_norestr",0.20,"airmix", False)]
for n,t,m,r in jobs:
    print("built", build(n,t,m,r))
