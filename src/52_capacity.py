"""Block 3: capacity, retuned for the sparser aligned label.

1600 trees / lr 0.015 was chosen when positives were 3.6% of rows. The top-5 frequency
label leaves 2.05% — the signal is nearly half as dense, and the optimum moves with it.
Measured on the 41-feature base, which is the configuration actually shipped.
"""
import sys, time, importlib, numpy as np
sys.path.insert(0, "/Users/tshermatova/Downloads/week7-exam/src")
import lib, features
import lightgbm as lgb
lab = importlib.import_module("31_lab"); cache2 = importlib.import_module("30_cache2")
ex = importlib.import_module("33_extra")
_t0 = time.time()
GT5 = {u: set(v) for u, v in lib.target_top5(thr=0.80, lo=lib.B_AB, hi=lib.B_TEST).items()}
OPT = dict(feats3=True)

def run(name, **kw):
    Xo, CUo, CIo, _ = lab.load("outer", **OPT)
    z = np.load(cache2.CACHE + "outer.npz")
    FB = [int(lib.ITEMS_ALL[k]) for k in np.argsort(-z["pop_u"])[:5]]
    sc = []
    for i in range(3):
        Xs, ys, Cs, off = [], [], [], 0
        for t in lab.BASE:
            X, CU, CI, _ = lab.load(t, **OPT)
            Xs.append(X); ys.append(ex.labels_top5(t, 0.80))
            Cs.append(CU.astype(np.int64) + off); off += lib.N_USERS
        X = np.vstack(Xs); y = np.concatenate(ys); C = np.concatenate(Cs)
        o = np.argsort(C, kind="stable"); X, y, C = X[o], y[o], C[o]
        g = np.diff(np.flatnonzero(np.r_[True, C[1:] != C[:-1], True]))
        p = dict(lab.PARAMS); p["random_state"] = lib.SEED + 1000 * i
        p.update(n_estimators=1600, learning_rate=0.015); p.update(kw)
        m = lgb.LGBMRanker(**p).fit(X, y, group=g)
        sc.append(lib.evaluate(features.top5(m.predict(Xo), CUo, CIo, FB), GT5))
        del X, y, m
    a = np.array(sc)
    print(f"  {name:<46} {a.mean():.5f} +- {a.std():.5f}  [{time.time()-_t0:.0f}s]", flush=True)

print("Блок 3: ёмкость под метку с 2.05% позитивов (1600/0.015 известно = 0.43949)")
run("2400 деревьев, lr 0.010", n_estimators=2400, learning_rate=0.010)
run("800 деревьев, lr 0.030", n_estimators=800, learning_rate=0.030)
run("num_leaves 63", num_leaves=63)
run("num_leaves 15", num_leaves=15)
