"""Disk cache of FEATS2 (30-column) datasets, so experiments do not rebuild features.

Hold-out regime only: every training window lies entirely before B_AB (2020-05-18),
the evaluation set ("outer") takes history < B_AB and the airing grid B_AB..B_TEST.
Identical geometry to src/25_graded.py, which produced the recorded 0.54464.
"""
import sys, os, time, calendar, gc, numpy as np
sys.path.insert(0, "/Users/tshermatova/Downloads/week7-exam/src")
import lib, features, features2

N_POP = int(os.environ.get("N_POP", 80))   # global-popularity candidates per user
_SUF = "" if abs(lib.FEAT_THR - 0.20) < 1e-9 else f"_t{int(round(lib.FEAT_THR*100))}"
_SUF += "" if N_POP == 80 else f"_p{N_POP}"
CACHE = lib.W + f"ds2{_SUF}/"
os.makedirs(CACHE, exist_ok=True)
_t0 = time.time()


def ts(y, m, d):
    return calendar.timegm((y, m, d, 0, 0, 0, 0, 0, 0))


# tag -> (split = history cut = airing window start, label/airing window end)
WIN = {
    "s0323_lAB":   (ts(2020, 3, 23), lib.B_AB),
    "s0330_l0504": (ts(2020, 3, 30), ts(2020, 5, 4)),
    "s0330_lAB":   (ts(2020, 3, 30), lib.B_AB),
    "s0406_lAB":   (ts(2020, 4, 6),  lib.B_AB),
    "s0413_lAB":   (ts(2020, 4, 13), lib.B_AB),
    "s0420_lAB":   (ts(2020, 4, 20), lib.B_AB),
    "s0427_lAB":   (ts(2020, 4, 27), lib.B_AB),
    "s0504_lAB":   (ts(2020, 5, 4),  lib.B_AB),
    # FINAL regime: the hold-out geometry shifted forward by 70 days (= B_TEST - B_AB),
    # so the shipped model sees the same window shapes, only later. Same as 28_final_3win.
    "f0608_l0713": (ts(2020, 6, 8),  ts(2020, 7, 13)),
    "f0622_lT":    (ts(2020, 6, 22), lib.B_TEST),
    "f0706_lT":    (ts(2020, 7, 6),  lib.B_TEST),
    # SECOND hold-out, five weeks earlier: target 04-13..B_AB, everything that feeds it
    # ends before 04-13. A change is only believed when it wins on both instruments.
    "e0323_l0413": (ts(2020, 3, 23), ts(2020, 4, 13)),
    "e0330_l0413": (ts(2020, 3, 30), ts(2020, 4, 13)),
    "e0406_l0413": (ts(2020, 4, 6),  ts(2020, 4, 13)),
}

B_EVAL2 = ts(2020, 4, 13)
EVAL2_WIN = ["e0330_l0413", "e0323_l0413", "e0406_l0413"]

T_END = int(np.load(lib.W + "airings_raw.npy")[-1, 2]) + 1
FINAL_WIN = ["f0608_l0713", "f0622_lT", "f0706_lT"]


def build(tag, split, hi, label=True, users=None, n_pop=None):
    """Build one dataset. Returns X (30 cols), CU, CI, y (None when label=False)."""
    n_pop = N_POP if n_pop is None else n_pop
    f = CACHE + tag + ".npz"
    if os.path.exists(f):
        z = np.load(f)
        return z["X"], z["CU"], z["CI"], (z["y"] if "y" in z.files else None)
    hm = features.history_arrays(split)
    if users is None:
        users = sorted(set(int(v) for v in np.unique(lib.UI[hm])))
    lm = (lib.FRAC >= lib.TGT_THR) & (lib.PST >= split) & (lib.PST < hi) if label else None
    blk = features.collab_block(hm)
    X, CU, CI, y, aux = features2.build2(hm, split, hi, users, n_pop=n_pop, label_mask=lm)
    X = features2.attach(X, CU, CI, blk)
    del blk
    gc.collect()
    if y is None:
        np.savez_compressed(f, X=X, CU=CU, CI=CI, pop_u=aux["pop_u"])
    else:
        np.savez_compressed(f, X=X, CU=CU, CI=CI, y=y, pop_u=aux["pop_u"])
    pos = "-" if y is None else f"{y.mean():.4f}"
    print(f"   cached {tag:<14} {str(X.shape):>14}  pos={pos}  [{time.time()-_t0:.0f}s]", flush=True)
    return X, CU, CI, y


def get(tag):
    return build(tag, *WIN[tag])


def outer():
    """Evaluation set: the 1 260 test users, scored against the hold-out ground truth."""
    return build("outer", lib.B_AB, lib.B_TEST, label=False, users=lib.TEST_UI)


def outer2():
    """Second evaluation set: same 1 260 users, target window 04-13..B_AB."""
    return build("outer2", B_EVAL2, lib.B_AB, label=False, users=lib.TEST_UI)


def final():
    """Prediction set: all of weeks 11-30 as history, weeks 31-42 as the airing grid.

    Weeks 31-42 contribute only which programme airs when; no viewing behaviour from
    the test window exists in the log at all, so there is nothing there to leak.
    """
    return build("final", lib.B_TEST, T_END, label=False, users=lib.TEST_UI)


def fallback():
    z = np.load(CACHE + "outer.npz")
    return [int(lib.ITEMS_ALL[k]) for k in np.argsort(-z["pop_u"])[:5]]


if __name__ == "__main__":
    want = sys.argv[1:] or (["outer"] + list(WIN))
    for t in want:
        if t == "outer":
            outer()
        elif t == "final":
            final()
        else:
            get(t)
    print(f"done [{time.time()-_t0:.0f}s]")
