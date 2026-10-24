"""Does encoding the target's ordering in the label beat plain top-5 membership?"""
import os
import sys, time, importlib, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib, features
import lightgbm as lgb
lab = importlib.import_module("31_lab"); cache2 = importlib.import_module("30_cache2")
ex = importlib.import_module("33_extra")
_t0 = time.time()
GT5 = {u: set(v) for u, v in lib.target_top5(thr=0.80, lo=lib.B_AB, hi=lib.B_TEST).items()}
OPT = dict(feats3=True, feats9=True)   # 46 features: the base we will actually ship
BIG = dict(n_estimators=1600, learning_rate=0.015)

def run(name, fn):
    Xo, CUo, CIo, _ = lab.load("outer", **OPT)
    z = np.load(cache2.CACHE + "outer.npz")
    FB = [int(lib.ITEMS_ALL[k]) for k in np.argsort(-z["pop_u"])[:5]]
    sc = []
    for i in range(3):
        Xs, ys, Cs, off = [], [], [], 0
        for t in lab.BASE:
            X, CU, CI, _ = lab.load(t, **OPT)
            Xs.append(X); ys.append(fn(t)); Cs.append(CU.astype(np.int64) + off); off += lib.N_USERS
        X = np.vstack(Xs); y = np.concatenate(ys); C = np.concatenate(Cs)
        o = np.argsort(C, kind="stable"); X, y, C = X[o], y[o], C[o]
        g = np.diff(np.flatnonzero(np.r_[True, C[1:] != C[:-1], True]))
        p = dict(lab.PARAMS); p["random_state"] = lib.SEED + 1000 * i; p.update(BIG)
        m = lgb.LGBMRanker(**p).fit(X, y, group=g)
        sc.append(lib.evaluate(features.top5(m.predict(Xo), CUo, CIo, FB), GT5))
        del X, y, m
    a = np.array(sc)
    print(f"  {name:<44} {a.mean():.5f} +- {a.std():.5f}  [{time.time()-_t0:.0f}s]", flush=True)

# The binary top-5 label at 46 features is already measured: 0.46667 +- 0.00073
# (src/47_f9_true.py). Only the two graded variants are missing, so only they are run.
print("Форма метки на 46 признаках, против истинной цели")
print("  бинарный топ-5 (известно из блока 2)           0.46667 +- 0.00073")
run("градуированная: ранг 5..1 внутри топ-5", lambda t: ex.labels_graded(t, "rank"))
run("градуированная: счётчик показов, cap 5", lambda t: ex.labels_graded(t, "count", 5))
