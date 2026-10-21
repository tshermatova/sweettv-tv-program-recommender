import csv, calendar, numpy as np
D="/Users/tshermatova/Downloads/week7-exam/data/"; W="/Users/tshermatova/Downloads/week7-exam/work/"
def ts(s): return calendar.timegm((int(s[6:10]),int(s[3:5]),int(s[0:2]),int(s[11:13]),int(s[14:16]),int(s[17:19]),0,0,0))
rows=[]; seen=set()
for fn in ["export_arh_11-20-final.csv","export_arh_21-30-final.csv","export_arh_31-42-final.csv"]:
    with open(D+fn, newline="") as f:
        r=csv.reader(f); next(r)
        for row in r:
            sid=int(row[5])
            if not sid: continue
            ch=int(row[0]); st=ts(row[2])
            k=(ch,st,sid)
            if k in seen: continue
            seen.add(k); rows.append((sid,ch,st))
A=np.array(rows,dtype=np.int64)
A=A[np.argsort(A[:,2])]
np.save(W+"airings_raw.npy", A)
print("labelled airings:", len(A), "| shows:", len(np.unique(A[:,0])),
      "| range:", A[0,2], "->", A[-1,2])
