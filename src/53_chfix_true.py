"""Is the shipped model hurt by the ch_aff train/serve mismatch?

`features2.build2` computes ch_aff straight from `lib.CH_VIEWS`, which sums channel
seconds over the whole log. In the FINAL regime that means the training windows see a
channel profile running to 2020-07-26 while their history stops in June, and the
prediction set sees an honest one — the model is trained on one distribution and applied
to another. `ch_fix` recomputes those two columns with the profile cut at each window's
own history boundary.

Measured before building anything: the hold-out answers this in ten minutes, a build
takes forty-five and would have to be thrown away if the answer is no.
"""
import os
import sys, time, importlib, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib, features
import lightgbm as lgb
lab = importlib.import_module("31_lab"); cache2 = importlib.import_module("30_cache2")
ex = importlib.import_module("33_extra")
_t0 = time.time()
GT5 = {u: set(v) for u, v in lib.target_top5(thr=0.80, lo=lib.B_AB, hi=lib.B_TEST).items()}

def run(name, opt):
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
        p.update(n_estimators=1600, learning_rate=0.015)
        m = lgb.LGBMRanker(**p).fit(X, y, group=g)
        sc.append(lib.evaluate(features.top5(m.predict(Xo), CUo, CIo, FB), GT5))
        del X, y, m
    a = np.array(sc)
    print(f"  {name:<46} {a.mean():.5f} +- {a.std():.5f}  [{time.time()-_t0:.0f}s]", flush=True)

print("Честный ch_aff (41 известно = 0.43949)")
run("41 с честными ch_aff / ch_rank", dict(feats3=True, chfix=True))
