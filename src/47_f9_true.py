"""Expected-frequency features, measured against the organisers' true target."""
import sys, time, importlib, numpy as np
sys.path.insert(0, "/Users/tshermatova/Downloads/week7-exam/src")
import lib, features
import lightgbm as lgb
lab = importlib.import_module("31_lab"); cache2 = importlib.import_module("30_cache2")
ex = importlib.import_module("33_extra")
_t0 = time.time()
GT5 = {u: set(v) for u, v in lib.target_top5(thr=0.80, lo=lib.B_AB, hi=lib.B_TEST).items()}
BIG = dict(n_estimators=1600, learning_rate=0.015)

def run(name, opt, drop_dead=False):
    Xo, CUo, CIo, _ = lab.load("outer", **opt)
    z = np.load(cache2.CACHE + "outer.npz")
    FB = [int(lib.ITEMS_ALL[k]) for k in np.argsort(-z["pop_u"])[:5]]
    keep_o = None
    if drop_dead:
        AR = np.load(lib.W + "airings_raw.npy")
        i = np.searchsorted(AR[:, 2], lib.B_AB, "left"); j = np.searchsorted(AR[:, 2], lib.B_TEST, "right")
        live = set(int(x) for x in np.unique(AR[i:j, 0]) if int(x))
        keep_o = np.array([int(lib.ITEMS_ALL[b]) in live for b in CIo])
    sc = []
    for s_ in range(3):
        Xs, ys, Cs, off = [], [], [], 0
        for t in lab.BASE:
            X, CU, CI, _ = lab.load(t, **opt)
            y = ex.labels_top5(t, 0.80)
            if drop_dead:
                lo, hi = cache2.WIN[t]
                AR = np.load(lib.W + "airings_raw.npy")
                i = np.searchsorted(AR[:, 2], lo, "left"); j = np.searchsorted(AR[:, 2], hi, "right")
                lv = set(int(x) for x in np.unique(AR[i:j, 0]) if int(x))
                k = np.array([int(lib.ITEMS_ALL[b]) in lv for b in CI])
                X, CU, y = X[k], CU[k], y[k]
            Xs.append(X); ys.append(y); Cs.append(CU.astype(np.int64) + off); off += lib.N_USERS
        X = np.vstack(Xs); y = np.concatenate(ys); C = np.concatenate(Cs)
        o = np.argsort(C, kind="stable"); X, y, C = X[o], y[o], C[o]
        g = np.diff(np.flatnonzero(np.r_[True, C[1:] != C[:-1], True]))
        p = dict(lab.PARAMS); p["random_state"] = lib.SEED + 1000 * s_; p.update(BIG)
        m = lgb.LGBMRanker(**p).fit(X, y, group=g)
        v = m.predict(Xo)
        if keep_o is not None:
            v = np.where(keep_o, v, -1e9)      # candidates that never air cannot be right
        sc.append(lib.evaluate(features.top5(v, CUo, CIo, FB), GT5))
        del X, y, m
    a = np.array(sc)
    print(f"  {name:<46} {a.mean():.5f} +- {a.std():.5f}  [{time.time()-_t0:.0f}s]", flush=True)

print("Против истинной цели, метка = топ-5 по частоте")
run("41 (опорная, выровненная метка)", dict(feats3=True))
run("46 = 41 + ожидаемая частота", dict(feats3=True, feats9=True))
run("46 + отсечение кандидатов без эфира", dict(feats3=True, feats9=True), drop_dead=True)
