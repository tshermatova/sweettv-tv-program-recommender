"""Block 1: more expected-frequency variants, and capacity retuned for the sparser label."""
import os
import sys, time, importlib, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib, features
import lightgbm as lgb
lab = importlib.import_module("31_lab"); cache2 = importlib.import_module("30_cache2")
ex = importlib.import_module("33_extra")
_t0 = time.time()
GT5 = {u: set(v) for u, v in lib.target_top5(thr=0.80, lo=lib.B_AB, hi=lib.B_TEST).items()}

def run(name, opt, **kw):
    Xo, CUo, CIo, _ = lab.load("outer", **opt)
    z = np.load(cache2.CACHE + "outer.npz")
    FB = [int(lib.ITEMS_ALL[k]) for k in np.argsort(-z["pop_u"])[:5]]
    sc = []
    for i in range(3):
        Xs, ys, Cs, off = [], [], [], 0
        for t in lab.BASE:
            X, CU, CI, _ = lab.load(t, **opt)
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

print("Блок 1: вариации ожидаемой частоты (46 известно = 0.46667)")
run("52 = 46 + FEATS10", dict(feats3=True, feats9=True, feats10=True))
print("Блок 3: ёмкость под разрежённую метку (позитивов 2.05% вместо 3.6%)")
run("52, 2400 деревьев lr 0.010", dict(feats3=True, feats9=True, feats10=True),
    n_estimators=2400, learning_rate=0.010)
run("52, 800 деревьев lr 0.03", dict(feats3=True, feats9=True, feats10=True),
    n_estimators=800, learning_rate=0.03)
run("52, num_leaves 63", dict(feats3=True, feats9=True, feats10=True), num_leaves=63)
print("Блок 2: расписание обучающих недель")
run("52 + FEATS11 (58)", dict(feats3=True, feats9=True, feats10=True, feats11=True))
run("46 + FEATS11 (52, без FEATS10)", dict(feats3=True, feats9=True, feats11=True))
