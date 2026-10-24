"""Interval join: view log x broadcast schedule -> user-programme interactions."""
import csv, time, calendar, numpy as np, os
# Repository root, derived from this file's own location so the code runs from any checkout.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
D = ROOT + "data/"
OUT = ROOT + "work/"
os.makedirs(OUT, exist_ok=True)
t0 = time.time()

def ts_log(s):   # 'YYYY-MM-DD HH:MM:SS'
    return calendar.timegm((int(s[0:4]),int(s[5:7]),int(s[8:10]),int(s[11:13]),int(s[14:16]),int(s[17:19]),0,0,0))
def ts_sched(s): # 'DD.MM.YYYY HH:MM:SS'
    return calendar.timegm((int(s[6:10]),int(s[3:5]),int(s[0:2]),int(s[11:13]),int(s[14:16]),int(s[17:19]),0,0,0))

# ---------- 1. schedule ----------
SCHED = ["export_arh_11-20-final.csv","export_arh_21-30-final.csv","export_arh_31-42-final.csv"]
rows = []           # ch, start, end, show
seen = set()        # dedup across overlapping files
show_meta = {}      # show -> (title, category, g1,g2,g3, year, director, actors)
for fn in SCHED:
    with open(D+fn, newline="") as f:
        r = csv.reader(f); next(r)
        for row in r:
            sid = int(row[5])
            st = ts_sched(row[2]); dur = int(row[3]) if row[3] else 0
            key = (int(row[0]), st, sid)
            if key in seen: continue
            seen.add(key)
            rows.append((int(row[0]), st, st+dur, sid))
            if sid and sid not in show_meta:
                show_meta[sid] = (row[4], row[6], row[7], row[8], row[9], row[10], row[11], row[12])
del seen
sched = np.array(rows, dtype=np.int64); del rows
print(f"schedule rows (deduped): {len(sched):,}  labelled: {(sched[:,3]!=0).sum():,}  shows with meta: {len(show_meta):,}  [{time.time()-t0:.1f}s]")

# per-channel index, sorted by start
order = np.lexsort((sched[:,1], sched[:,0]))
sched = sched[order]
ch_col = sched[:,0]
uch, ch_start_idx = np.unique(ch_col, return_index=True)
ch_bounds = {int(c): (int(ch_start_idx[i]), int(ch_start_idx[i+1]) if i+1 < len(uch) else len(sched))
             for i, c in enumerate(uch)}
S_START, S_END, S_SHOW = sched[:,1].copy(), sched[:,2].copy(), sched[:,3].copy()
MAXDUR = int((S_END - S_START).max())
print(f"channels in schedule: {len(ch_bounds)}  max programme duration: {MAXDUR}s  [{time.time()-t0:.1f}s]")

# ---------- 2. view log, streamed and joined ----------
matches = []        # user_idx, show, overlap_sec, prog_dur, prog_start
uid_map = {}; uids = []
ch_views = {}       # user_idx -> {channel: seconds}  (channel affinity, uses unlabelled airtime too)
n = 0; n_unmatched_ch = 0; n_no_prog = 0; n_lab_hits = 0
with open(D+"dataset11-30.csv", newline="") as f:
    r = csv.reader(f); next(r)
    for row in r:
        n += 1
        u = row[0]
        ui = uid_map.get(u)
        if ui is None:
            ui = uid_map[u] = len(uids); uids.append(u); ch_views[ui] = {}
        ch = int(row[1])
        vs = ts_log(row[2]); ve = ts_log(row[3])
        ch_views[ui][ch] = ch_views[ui].get(ch, 0) + (ve - vs)
        b = ch_bounds.get(ch)
        if b is None:
            n_unmatched_ch += 1; continue
        lo, hi = b
        # programmes with start < ve and end > vs
        j = int(np.searchsorted(S_START[lo:hi], ve, side="right")) + lo
        i = int(np.searchsorted(S_START[lo:hi], vs - MAXDUR, side="left")) + lo
        hit = False
        for k in range(i, j):
            ps, pe = S_START[k], S_END[k]
            if pe <= vs: continue
            ov = min(ve, pe) - max(vs, ps)
            if ov <= 0: continue
            hit = True
            sid = S_SHOW[k]
            if sid:
                matches.append((ui, int(sid), int(ov), int(pe-ps), int(ps)))
                n_lab_hits += 1
        if not hit: n_no_prog += 1
        if n % 1_000_000 == 0:
            print(f"  {n:,} view rows  matches={len(matches):,}  [{time.time()-t0:.1f}s]")

print(f"\nview rows: {n:,}  users: {len(uids):,}")
print(f"  rows on channels absent from schedule: {n_unmatched_ch:,} ({100*n_unmatched_ch/n:.2f}%)")
print(f"  rows overlapping no programme at all:  {n_no_prog:,} ({100*n_no_prog/n:.2f}%)")
print(f"  labelled (user,programme) overlap events: {n_lab_hits:,}")

M = np.array(matches, dtype=np.int64); del matches
np.save(OUT+"matches.npy", M)
with open(OUT+"users.txt","w") as f: f.write("\n".join(uids))
import pickle
with open(OUT+"show_meta.pkl","wb") as f: pickle.dump(show_meta, f)
with open(OUT+"ch_views.pkl","wb") as f: pickle.dump(ch_views, f)
print(f"saved -> {OUT}  [{time.time()-t0:.1f}s]")
