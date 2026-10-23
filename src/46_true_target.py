"""Re-measure against the organisers' actual target, and align the training label to it.

Until now the hold-out counted every programme with a full view as relevant (median 11
per user) and the training label said "watched at all". The data page says the truth is
the five most frequent airings with screen time >= 0.8 — exactly five per user. Every
number measured before this script was therefore optimistic and chose features against
the wrong objective.
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
BIG = dict(n_estimators=1600, learning_rate=0.015)


def run(name, label):
    Xo, CUo, CIo, _ = lab.load("outer", **OPT)
    z = np.load(cache2.CACHE + "outer.npz")
    FB = [int(lib.ITEMS_ALL[k]) for k in np.argsort(-z["pop_u"])[:5]]
    sc = []
    for i in range(3):
        Xs, ys, Cs, off = [], [], [], 0
        for t in lab.BASE:
            X, CU, CI, y = lab.load(t, **OPT)
            if label == "watch08":
                y = ex.labels(t, 0.80)
            elif label == "top5":
                y = ex.labels_top5(t, 0.80)
            Xs.append(X); ys.append(y); Cs.append(CU.astype(np.int64) + off); off += lib.N_USERS
        X = np.vstack(Xs); y = np.concatenate(ys); C = np.concatenate(Cs)
        o = np.argsort(C, kind="stable"); X, y, C = X[o], y[o], C[o]
        g = np.diff(np.flatnonzero(np.r_[True, C[1:] != C[:-1], True]))
        p = dict(lab.PARAMS); p["random_state"] = lib.SEED + 1000 * i; p.update(BIG)
        m = lgb.LGBMRanker(**p).fit(X, y, group=g)
        sc.append(lib.evaluate(features.top5(m.predict(Xo), CUo, CIo, FB), GT5))
        del X, y, m
    a = np.array(sc)
    print(f"  {name:<40} MAP@5 (истинная цель) = {a.mean():.5f} +- {a.std():.5f}"
          f"  [{time.time()-_t0:.0f}s]", flush=True)
    return a.mean()


print("Оценка против ИСТИННОЙ цели: топ-5 по частоте, screen time >= 0.8")
run("метка: всё с frac>=1.0 (как было)", "watch10")
run("метка: всё с frac>=0.8", "watch08")
run("метка: топ-5 по частоте, frac>=0.8", "top5")
