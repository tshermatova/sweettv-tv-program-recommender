import numpy as np, pickle, csv, os
from collections import Counter, defaultdict
W=ROOT + "work/"; D=ROOT + "data/"
S=ROOT + "submissions/"; os.makedirs(S, exist_ok=True)
A=pickle.load(open(W+"airings.pkl","rb")); AIR=A["airings"]
airT=AIR["T"]; airT_set=set(airT)
M=np.load(W+"matches.npy"); users=open(W+"users.txt").read().split("\n")
uidx={u:i for i,u in enumerate(users)}
test_users=[r[0] for r in list(csv.reader(open(D+"submission.csv")))[1:]]

ui,sid,ov,pdur = M[:,0],M[:,1],M[:,2],M[:,3]
keep = (ov/np.maximum(pdur,1))>=0.20          # all of weeks 11-30 = full training data
H=defaultdict(Counter)
for u,s in zip(ui[keep], sid[keep]): H[int(u)][int(s)]+=1
pop=Counter()
for u,c in H.items():
    for s in c: pop[s]+=1
fallback=[s for s,_ in Counter({s:c for s,c in pop.items() if s in airT_set}).most_common(5)]
print("global fallback top-5:", fallback)

rows=[]; n_fb=0
for u in test_users:
    h=H.get(uidx[u])
    out=[]
    if h:
        cand=sorted(((n*np.log1p(airT.get(s,0)), s) for s,n in h.items() if s in airT_set),
                    key=lambda x:-x[0])
        out=[s for _,s in cand[:5]]
    if len(out)<5: n_fb+=1
    for s in fallback:
        if len(out)>=5: break
        if s not in out: out.append(s)
    rows.append((u, " ".join(str(x) for x in out[:5])))
print("users needing fallback padding:", n_fb)
p=S+"sub01_repeat_x_airings.csv"
with open(p,"w",newline="") as f:
    w=csv.writer(f); w.writerow(["user_id","tv_show_id"]); w.writerows(rows)
print("wrote", p, len(rows), "rows")
print(open(p).read().split("\n")[1])
