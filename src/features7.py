"""The four collaborative signals, recomputed on recent history only.

`features.collab_block` fits association rules, personalised PageRank, content
similarity and iALS on the entire history window — up to nineteen weeks of it. A
co-view from March therefore counts as much as one from last week, although what
people watch together drifts: a series ends, a tournament starts, a film leaves the
grid. These four columns are the same four methods fitted on a short recent window,
so the ranker can tell a durable association from a stale one.

They are appended alongside the full-history versions rather than replacing them; the
comparison between the pair is itself the signal.
"""
import sys, numpy as np
sys.path.insert(0, "/Users/tshermatova/Downloads/week7-exam/src")
import lib, features

FEATS7 = ["assoc_rec", "ppr_rec", "cbf_rec", "mf_rec"]
WEEKS = 4


def extra(split, CU, CI, weeks=WEEKS):
    """Fit the four methods on [split - weeks, split) and read them off per candidate."""
    lo = split - weeks * 7 * 86400
    hm = features.history_arrays(split, lo=lo)
    if hm.sum() < 1000:                       # too little to fit anything meaningful
        return np.zeros((len(CU), len(FEATS7)), np.float32)
    blk = features.collab_block(hm)
    out, ru, ci, _ = blk
    r = np.array([ru.get(int(a), -1) for a in CU])
    c = np.array([ci.get(int(b), -1) for b in CI])
    ok = (r >= 0) & (c >= 0)
    X = np.zeros((len(CU), len(FEATS7)), np.float32)
    for n, k in enumerate(["assoc", "ppr", "cbf", "mf"]):
        v = np.zeros(len(CU), np.float32)
        v[ok] = out[k][r[ok], c[ok]]
        X[:, n] = v
    return X
