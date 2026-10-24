"""Airing counts per (show, window) and per (show, channel, window) from the schedule."""
import csv, pickle, calendar, numpy as np
from collections import Counter, defaultdict
import os
# Repository root, derived from this file's own location so the code runs from any checkout.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
D=ROOT + "data/"; W=ROOT + "work/"
def ts(s): return calendar.timegm((int(s[6:10]),int(s[3:5]),int(s[0:2]),int(s[11:13]),int(s[14:16]),int(s[17:19]),0,0,0))
# window boundaries (epoch, UTC)
B_AB = ts("18.05.2020 00:00:00")   # end of feature window A / start of target window B
B_TEST = ts("27.07.2020 00:00:00") # start of test window
out = {w: Counter() for w in ("A","B","T")}
out_ch = {w: Counter() for w in ("A","B","T")}
seen=set()
for fn in ["export_arh_11-20-final.csv","export_arh_21-30-final.csv","export_arh_31-42-final.csv"]:
    with open(D+fn, newline="") as f:
        r=csv.reader(f); next(r)
        for row in r:
            sid=int(row[5])
            if not sid: continue
            ch=int(row[0]); st=ts(row[2])
            k=(ch,st,sid)
            if k in seen: continue
            seen.add(k)
            w = "A" if st < B_AB else ("B" if st < B_TEST else "T")
            out[w][sid]+=1; out_ch[w][(ch,sid)]+=1
for w in ("A","B","T"):
    print(f"window {w}: {len(out[w])} distinct labelled shows, {sum(out[w].values())} airings")
print("shows in T also in A or B:", len(set(out['T']) & (set(out['A'])|set(out['B']))), "/", len(out['T']))
print("shows in B also in A:", len(set(out['B']) & set(out['A'])), "/", len(out['B']))
pickle.dump({"airings":out,"airings_ch":out_ch,"B_AB":B_AB,"B_TEST":B_TEST}, open(W+"airings.pkl","wb"))
