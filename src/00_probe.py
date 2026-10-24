import csv, sys, time
from collections import Counter
import os
# Repository root, derived from this file's own location so the code runs from any checkout.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
D = ROOT + "data/"
t0 = time.time()

# ---- submission ----
sub_users = []
with open(D+"submission.csv", newline="") as f:
    r = csv.reader(f); hdr = next(r)
    for row in r: sub_users.append(row[0])
print("submission header:", hdr, "| rows:", len(sub_users), "| unique:", len(set(sub_users)))

# ---- view log ----
vs_ch = Counter(); log_users = set(); tmin=tmax=None; n=0; neg=0; mism=0
with open(D+"dataset11-30.csv", newline="") as f:
    r = csv.reader(f); vh = next(r)
    for row in r:
        n += 1
        log_users.add(row[0]); vs_ch[row[1]] += 1
        st, sp, du = row[2], row[3], row[4]
        if tmin is None or st < tmin: tmin = st
        if sp > (tmax or ""): tmax = sp
        if int(du) <= 0: neg += 1
print("view log header:", vh)
print("  rows:", n, "| users:", len(log_users), "| channels(vsetv_id):", len(vs_ch))
print("  time range:", tmin, "->", tmax, "| non-positive duraton:", neg)

# ---- schedules ----
sched_ch = {}
for fn in ["export_arh_11-20-final.csv","export_arh_21-30-final.csv","export_arh_31-42-final.csv"]:
    ch = set(); shows = set(); lab = 0; tot = 0; smin=smax=None
    with open(D+fn, newline="") as f:
        r = csv.reader(f); sh = next(r)
        for row in r:
            tot += 1; ch.add(row[0])
            d = row[2]  # DD.MM.YYYY HH:MM:SS
            k = d[6:10]+d[3:5]+d[0:2]+d[11:]
            if smin is None or k < smin: smin = k
            if smax is None or k > smax: smax = k
            if row[5] != "0": lab += 1; shows.add(row[5])
    sched_ch[fn] = ch
    print(f"{fn}: rows={tot} channels={len(ch)} labelled={lab} ({100*lab/tot:.1f}%) distinct_shows={len(shows)}")
    print(f"   start_time range: {smin} -> {smax}")

# ---- the join key question ----
all_sched_ch = set().union(*sched_ch.values())
vs = set(vs_ch)
print()
print("CHANNEL ID OVERLAP vsetv_id vs channel_id:")
print("  vsetv_id distinct:", len(vs), "| channel_id distinct:", len(all_sched_ch))
print("  intersection:", len(vs & all_sched_ch))
print("  in log but not in schedule:", len(vs - all_sched_ch))
print("  share of view rows on matched channels: %.4f" % (sum(c for k,c in vs_ch.items() if k in all_sched_ch)/n))

# ---- test users coverage ----
su = set(sub_users)
print()
print("test users:", len(su), "| present in view log:", len(su & log_users), "| cold (no rows at all):", len(su - log_users))
print("elapsed %.1fs" % (time.time()-t0))
