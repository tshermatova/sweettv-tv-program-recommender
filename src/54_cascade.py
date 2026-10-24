"""Cascade: a second ranker over the first one's top 20.

The one requirement that makes this valid is that stage 1 must never score rows it was
fitted on — otherwise its score is optimistic exactly where stage 2 learns to trust it,
which is the same class of leak that cost a night on `features9`. The three training
windows give that split for free: fit stage 1 on two of them, score the third.

Cheap by design, because this is a screen, not a final model:
  * one fold, so stage 2 trains on a single window's top-20 rather than all three
  * stage 1 inside the fold runs at 400 trees, since its only job there is to choose
    which twenty candidates stage 2 sees, not to order them finally
  * the stage 1 used for evaluation is the full 1600-tree model, as shipped

Stage 2 sees the 41 features plus what stage 1 thought: its score, the candidate's rank
within the user, and the gap to that user's best candidate.
"""
import sys, time, importlib, numpy as np
sys.path.insert(0, "/Users/tshermatova/Downloads/week7-exam/src")
import lib, features
import lightgbm as lgb

lab = importlib.import_module("31_lab")
cache2 = importlib.import_module("30_cache2")
ex = importlib.import_module("33_extra")
_t0 = time.time()

GT5 = {u: set(v) for u, v in lib.target_top5(thr=0.80, lo=lib.B_AB, hi=lib.B_TEST).items()}
OPT = dict(feats3=True)
TOP = 20
FOLD_TREES = 400


def stack(tags):
    Xs, ys, Cs, off = [], [], [], 0
    for t in tags:
        X, CU, CI, _ = lab.load(t, **OPT)
        Xs.append(X); ys.append(ex.labels_top5(t, 0.80))
        Cs.append(CU.astype(np.int64) + off); off += lib.N_USERS
    X = np.vstack(Xs); y = np.concatenate(ys); C = np.concatenate(Cs)
    o = np.argsort(C, kind="stable"); X, y, C = X[o], y[o], C[o]
    g = np.diff(np.flatnonzero(np.r_[True, C[1:] != C[:-1], True]))
    return X, y, g


def fit(X, y, g, seed, trees=1600, lr=0.015):
    p = dict(lab.PARAMS)
    p.update(random_state=seed, n_estimators=trees, learning_rate=lr)
    return lgb.LGBMRanker(**p).fit(X, y, group=g)


def top_rows(score, CU, k=TOP):
    """Indices of each user's k best rows, plus rank and gap-to-best for those rows."""
    o = np.lexsort((-score, CU))
    starts = np.flatnonzero(np.r_[True, CU[o][1:] != CU[o][:-1], True])
    keep, rank, gap = [], [], []
    for a, b in zip(starts[:-1], starts[1:]):
        sel = o[a:min(a + k, b)]
        keep.append(sel)
        rank.append(np.arange(len(sel), dtype=np.float32))
        gap.append(score[sel][0] - score[sel])
    return (np.concatenate(keep), np.concatenate(rank).astype(np.float32),
            np.concatenate(gap).astype(np.float32))


Xo, CUo, CIo, _ = lab.load("outer", **OPT)
z = np.load(cache2.CACHE + "outer.npz")
FB = [int(lib.ITEMS_ALL[k]) for k in np.argsort(-z["pop_u"])[:5]]
HELD = lab.BASE[0]                     # the window stage 2 trains on
OTHERS = lab.BASE[1:]

base_sc, casc_sc = [], []
for i in range(3):
    seed = lib.SEED + 1000 * i

    # stage 1 for evaluation: the shipped configuration, all three windows
    X, y, g = stack(lab.BASE)
    m1 = fit(X, y, g, seed)
    v_out = m1.predict(Xo)
    base_sc.append(lib.evaluate(features.top5(v_out, CUo, CIo, FB), GT5))
    del X, y

    # stage 1 inside the fold: fitted WITHOUT the held window, so its scores there are
    # out-of-sample and stage 2 learns from honest numbers
    Xf, yf, gf = stack(OTHERS)
    mf = fit(Xf, yf, gf, seed, trees=FOLD_TREES, lr=0.05)
    del Xf, yf

    Xh, CUh, CIh, _ = lab.load(HELD, **OPT)
    yh = ex.labels_top5(HELD, 0.80)
    vh = mf.predict(Xh)
    sel, rk, gp = top_rows(vh, CUh.astype(np.int64))
    X2 = np.column_stack([Xh[sel], vh[sel], rk, gp]).astype(np.float32)
    y2, C2 = yh[sel], CUh[sel].astype(np.int64)
    o2 = np.argsort(C2, kind="stable")
    X2, y2, C2 = X2[o2], y2[o2], C2[o2]
    g2 = np.diff(np.flatnonzero(np.r_[True, C2[1:] != C2[:-1], True]))
    m2 = fit(X2, y2, g2, seed, trees=800, lr=0.03)
    del X2, y2

    # apply: stage 1 picks twenty, stage 2 reorders them, everything else sinks
    sel_o, rk_o, gp_o = top_rows(v_out, CUo.astype(np.int64))
    Xo2 = np.column_stack([Xo[sel_o], v_out[sel_o], rk_o, gp_o]).astype(np.float32)
    final = np.full(len(CUo), -1e9, np.float64)
    final[sel_o] = m2.predict(Xo2)
    casc_sc.append(lib.evaluate(features.top5(final, CUo, CIo, FB), GT5))
    print(f"  сид {i}: база {base_sc[-1]:.5f}   каскад {casc_sc[-1]:.5f}"
          f"  [{time.time()-_t0:.0f}s]", flush=True)
    del m1, mf, m2, Xo2

a, b = np.array(base_sc), np.array(casc_sc)
print(f"\n  база          {a.mean():.5f} +- {a.std():.5f}")
print(f"  каскад топ-20 {b.mean():.5f} +- {b.std():.5f}   ({b.mean()-a.mean():+.5f})")
open("work/cascade_result.txt", "w").write(f"{a.mean():.5f} {b.mean():.5f}\n")
