"""Section 7: two chosen users, every method's Top-5 side by side, with titles.

Run in the HOLD-OUT regime, not the test regime, for one reason: only there do we know
what the user actually went on to watch, so each recommendation can be marked HIT or
miss. In the test window (weeks 31-42) the ground truth is Kaggle's and we never see it.

The four base methods are read straight out of the cached feature matrix (columns 26-29
of FEATS2 are assoc / ppr / cbf / mf, fitted on the same history window), so nothing is
recomputed and nothing can drift between this table and the model that was scored.
"""
import os
import sys, time, importlib, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib, features, features2
from collections import Counter

lab = importlib.import_module("31_lab")
cache2 = importlib.import_module("30_cache2")
_t0 = time.time()
OUT = open(lib.W + "two_users.txt", "w")


def say(s=""):
    print(s, flush=True)
    OUT.write(s + "\n")


def label(s):
    m = lib.SHOW_META.get(int(s))
    if not m:
        return f"id={s:<9} (no schedule metadata)"
    title = m[0].strip().strip("'.").strip()
    return f"id={s:<9} {title[:52]:<52} [{m[1] or '-'}]"


# ---- the two users ---------------------------------------------------------------
# The organisers' real target: each user's five most frequent airings at screen time
# >= 0.8, not every programme they watched. The earlier version of this table marked
# HIT/miss against the wider set, so its hit counts were too generous.
GT = {u: set(v) for u, v in lib.target_top5(thr=0.80, lo=lib.B_AB, hi=lib.B_TEST).items()}
hm = features.history_arrays(lib.B_AB)
ev = Counter(int(u) for u in lib.UI[hm])
dis = {}
for u, s in zip(lib.UI[hm], lib.SIDX[hm]):
    dis.setdefault(int(u), set()).add(int(s))

elig = [(ev.get(u, 0), u) for u in lib.TEST_UI if len(GT.get(u, ())) == 5 and ev.get(u, 0) > 0]
elig.sort()
LIGHT = elig[0][1]
HEAVY = elig[-1][1]
med = int(np.median([n for n, _ in elig]))

say("=" * 94)
say("SECTION 7 — two users, every method's Top-5 (scored against the organisers' top-5 target)")
say("=" * 94)
say()
say("Why these two. There is no cold-start case to show: all 1 260 test users appear in")
say("the view log, so no user has an empty history. The informative contrast is therefore")
say("volume of evidence. These are the extremes of the eligible range, which makes them")
say(f"the hardest and easiest cases the ranker faces. Median history is {med} events;")
say(f"{len(elig)} of 1 260 test users have both a history and something to predict.")
say()
for tag, u in [("HEAVY", HEAVY), ("LIGHT", LIGHT)]:
    say(f"  {tag:<6} user_id={lib.USERS[u]:<22} history {ev[u]:>5} events over "
        f"{len(dis[u]):>4} distinct programmes;  relevant in the hold-out window: {len(GT[u])}")
say()

# ---- scores ----------------------------------------------------------------------
OPT = dict(feats3=True)
Xo, CUo, CIo, FB = lab.evalset(**OPT)
say(f"fitting the final configuration on the hold-out geometry [{time.time()-_t0:.0f}s]")
mdl = lab.fit(lab.BASE, OPT)
pred = mdl.predict(Xo)
say(f"done [{time.time()-_t0:.0f}s]")

COLS = {"association rules": 26, "PageRank (PPR)": 27, "content-based (CBF)": 28,
        "matrix factorization (iALS)": 29}


def top5(vec, rows):
    k = rows[np.argsort(-vec[rows])][:5]
    return [int(lib.ITEMS_ALL[CIo[i]]) for i in k]


for tag, u in [("HEAVY", HEAVY), ("LIGHT", LIGHT)]:
    rows = np.flatnonzero(CUo == u)
    g = GT[u]
    say()
    say("=" * 94)
    say(f"{tag}  user_id={lib.USERS[u]}   candidate pool {len(rows)}   "
        f"actually watched in the hold-out window: {len(g)} programmes")
    say("=" * 94)

    h = Counter()
    for s in lib.SIDX[hm & (lib.UI == u)]:
        h[int(lib.ITEMS_ALL[s])] += 1
    say("\n  what this user watched most in the training window:")
    for s, n in h.most_common(6):
        say(f"     {n:>4}x  {label(s)}")

    lists = {"popularity (baseline)": FB}
    for name, c in COLS.items():
        lists[name] = top5(Xo[:, c], rows)
    lists["FINAL — LambdaRank, 41 features"] = top5(pred, rows)

    for name, L in lists.items():
        hits = sum(1 for s in L if s in g)
        say(f"\n  {name}   ({hits}/5 hit)")
        for i, s in enumerate(L, 1):
            say(f"     {i}. [{'HIT ' if s in g else 'miss'}] {label(s)}")

say()
say(f"written to {lib.W}two_users.txt  [{time.time()-_t0:.0f}s]")
OUT.close()
