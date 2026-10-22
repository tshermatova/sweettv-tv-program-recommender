"""Experiment harness on the calibrated hold-out. Every number printed here is a run.

Usage:  python3 src/31_lab.py <experiment> [<experiment> ...]
"""
import sys, time, importlib, numpy as np
sys.path.insert(0, "/Users/tshermatova/Downloads/week7-exam/src")
import lib, features, features2
import lightgbm as lgb

cache2 = importlib.import_module("30_cache2")
ex33 = importlib.import_module("33_extra")
_t0 = time.time()

GT = lib.target(lo=lib.B_AB, hi=lib.B_TEST)
BASE = ["s0413_lAB", "s0330_l0504", "s0427_lAB"]   # the recorded 3-window geometry
MORE = ["s0406_lAB", "s0420_lAB", "s0323_lAB", "s0504_lAB"]

_cache = {}


def load(tag, feats3=False, feats4=False, chfix=False):
    k = (tag, feats3, feats4, chfix)
    if k in _cache:
        return _cache[k]
    X, CU, CI, y = ex33._base(tag)
    X = X.copy()
    if chfix:
        X[:, 21:23] = ex33.ch_fix(tag)
    if feats3:
        X = np.hstack([X, ex33.extras(tag)]).astype(np.float32)
    if feats4:
        X = np.hstack([X, ex33.extras4(tag)]).astype(np.float32)
    _cache[k] = (X, CU, CI, y)
    return _cache[k]


def colnames(feats3=False, feats4=False, **_):
    import features3, features4
    n = list(features2.FEATS2)
    if feats3:
        n += features3.FEATS3
    if feats4:
        n += features4.FEATS4
    return n


def evalset(**opt):
    X, CU, CI, _ = load("outer", **opt)
    return X, CU, CI, cache2.fallback()


def stack(tags, **opt):
    """Concatenate cached windows, offsetting user ids so groups never merge."""
    Xs, ys, Cs, off = [], [], [], 0
    for t in tags:
        X, CU, CI, y = load(t, **opt)
        Xs.append(X); ys.append(y); Cs.append(CU.astype(np.int64) + off)
        off += lib.N_USERS
    X = np.vstack(Xs); y = np.concatenate(ys); C = np.concatenate(Cs)
    o = np.argsort(C, kind="stable")
    X, y, C = X[o], y[o], C[o]
    g = np.diff(np.flatnonzero(np.r_[True, C[1:] != C[:-1], True]))
    return X, y, g


PARAMS = dict(objective="lambdarank", metric="map", eval_at=[5], learning_rate=0.03,
              num_leaves=31, n_estimators=800, min_child_samples=50, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, random_state=lib.SEED,
              verbose=-1, n_jobs=3)   # fanless M1 Air: leave a core free, it throttles


def fit(tags=BASE, opt=None, **kw):
    X, y, g = stack(tags, **(opt or {}))
    p = dict(PARAMS); p.update(kw)
    return lgb.LGBMRanker(**p).fit(X, y, group=g)


def score_of(raw, opt=None):
    _, CUo, CIo, FB = evalset(**(opt or {}))
    return lib.evaluate(features.top5(raw, CUo, CIo, FB), GT)


def run(name, tags=BASE, opt=None, **kw):
    opt = opt or {}
    m = fit(tags, opt, **kw)
    Xo, _, _, _ = evalset(**opt)
    s = score_of(m.predict(Xo), opt)
    print(f"  {name:<52} MAP@5 = {s:.5f}  [{time.time()-_t0:.0f}s]", flush=True)
    return s, m


GT2 = lib.target(lo=cache2.B_EVAL2, hi=lib.B_AB)
EVAL2 = cache2.EVAL2_WIN


def run2(name, opt=None, n=3, drop=None, thr=None, wweight=None, **kw):
    """Score a configuration on BOTH hold-outs and report each.

    One window has fooled us four times: a change that wins on the 05-18..07-27 target
    by less than ~0.005 has repeatedly failed on Kaggle. A change is only believed here
    when it wins on the 04-13..05-18 target as well, fitted on windows that end before
    that target starts.
    """
    opt = opt or {}
    names = colnames(**opt)
    keep = [i for i, c in enumerate(names) if c not in drop] if drop else None

    def one(tags, eval_tag, gt):
        Xo, CUo, CIo, _ = load(eval_tag, **opt)
        z = np.load(cache2.CACHE + eval_tag + ".npz")
        FB = [int(lib.ITEMS_ALL[k]) for k in np.argsort(-z["pop_u"])[:5]]
        Xe = Xo if keep is None else Xo[:, keep]
        out = []
        for i in range(n):
            Xs, ys, Cs, Ws, off = [], [], [], [], 0
            for wi, t in enumerate(tags):
                X, CU, CI, y = load(t, **opt)
                if thr is not None:
                    y = ex33.labels(t, thr)
                Xs.append(X if keep is None else X[:, keep])
                ys.append(y); Cs.append(CU.astype(np.int64) + off); off += lib.N_USERS
                Ws.append(np.full(len(CU), wweight[wi] if wweight else 1.0, np.float32))
            X = np.vstack(Xs); y = np.concatenate(ys); C = np.concatenate(Cs)
            W = np.concatenate(Ws)
            o = np.argsort(C, kind="stable"); X, y, C, W = X[o], y[o], C[o], W[o]
            g = np.diff(np.flatnonzero(np.r_[True, C[1:] != C[:-1], True]))
            p = dict(PARAMS); p["random_state"] = lib.SEED + 1000 * i; p.update(kw)
            m = lgb.LGBMRanker(**p).fit(X, y, group=g, sample_weight=W)
            out.append(lib.evaluate(features.top5(m.predict(Xe), CUo, CIo, FB), gt))
            del X, y, C, m
        return np.array(out)

    a = one(BASE, "outer", GT)
    b = one(EVAL2, "outer2", GT2)
    print(f"  {name:<40} A {a.mean():.5f}+-{a.std():.5f}   "
          f"B {b.mean():.5f}+-{b.std():.5f}  [{time.time()-_t0:.0f}s]", flush=True)
    return a.mean(), b.mean()


def exp_validate2():
    """Does the second instrument reproduce Kaggle's verdict on 30 / 41 / 51?

    Kaggle said: 30 (0.44008) < 41 (0.44994) > 51 (0.44472). Hold-out A said
    30 < 41 < 51, which is where it went wrong. If B puts 51 below 41, B is worth
    trusting; if it agrees with A, the second window adds nothing and is dropped.
    """
    print("\n[validate2] calibrating the second hold-out against what Kaggle already told us")
    print("            Kaggle: 30 -> 0.44008,  41 -> 0.44994,  51 -> 0.44472")
    run2("30 base", opt=dict())
    run2("41 = base + FEATS3", opt=dict(feats3=True))
    run2("51 = base + FEATS3 + FEATS4", opt=dict(feats3=True, feats4=True))


def runm(name, tags=BASE, opt=None, n=3, drop=None, **kw):
    """Mean of n seeds. The seed-to-seed sd is 0.00178, so a single fit cannot
    separate configurations that differ by less than ~0.005. Everything from here
    on is measured this way."""
    opt = opt or {}
    names = colnames(**opt)
    keep = None
    if drop:
        keep = [i for i, c in enumerate(names) if c not in drop]
    Xo, CUo, CIo, FB = evalset(**opt)
    Xe = Xo if keep is None else Xo[:, keep]
    scores = []
    for i in range(n):
        Xs, ys, Cs, off = [], [], [], 0
        for t in tags:
            X, CU, CI, y = load(t, **opt)
            Xs.append(X if keep is None else X[:, keep])
            ys.append(y); Cs.append(CU.astype(np.int64) + off); off += lib.N_USERS
        X = np.vstack(Xs); y = np.concatenate(ys); C = np.concatenate(Cs)
        o = np.argsort(C, kind="stable"); X, y, C = X[o], y[o], C[o]
        g = np.diff(np.flatnonzero(np.r_[True, C[1:] != C[:-1], True]))
        p = dict(PARAMS); p["random_state"] = lib.SEED + 1000 * i; p.update(kw)
        m = lgb.LGBMRanker(**p).fit(X, y, group=g)
        scores.append(lib.evaluate(features.top5(m.predict(Xe), CUo, CIo, FB), GT))
        del X, y, C, m
    a = np.array(scores)
    print(f"  {name:<46} {a.mean():.5f} +- {a.std():.5f}   "
          f"({', '.join(f'{v:.5f}' for v in a)})  [{time.time()-_t0:.0f}s]", flush=True)
    return a.mean()


def rank_norm(v, CU):
    """Within-user reciprocal rank; lets scores from different fits be averaged."""
    o = np.lexsort((-v, CU))
    r = np.empty(len(v), np.float64)
    _, i0 = np.unique(CU[o], return_index=True)
    within = np.arange(len(CU)) - np.repeat(i0, np.diff(np.append(i0, len(CU))))
    r[o] = 1.0 / (1.0 + within)
    return r


# --------------------------------------------------------------------------- experiments
def exp_baseline():
    print("\n[baseline] the recorded 3-window configuration")
    run("3 windows, binary labels, defaults")


def exp_trunc():
    print("\n[trunc] lambdarank_truncation_level (LightGBM default 30, metric is MAP@5)")
    for t in [5, 8, 10, 15, 20, 30, 50]:
        run(f"truncation_level={t}", lambdarank_truncation_level=t)


def exp_feats3():
    print("\n[feats3] + 11 habit-shape features (streak, slot, watch completeness, channel)")
    run("baseline (30 features, reference)")
    run("+ FEATS3 (41 features)", opt=dict(feats3=True))


def exp_chfix():
    print("\n[chfix] channel affinity recomputed from history only (removes hold-out leak)")
    run("honest ch_aff / ch_rank", opt=dict(chfix=True))
    run("honest ch_aff + FEATS3", opt=dict(chfix=True, feats3=True))


def exp_windows():
    print("\n[windows] how many training windows")
    run("1 window (s0413)", ["s0413_lAB"])
    run("2 windows", BASE[:2])
    run("4 windows", BASE + MORE[:1])
    run("5 windows", BASE + MORE[:2])
    run("6 windows", BASE + MORE[:3])
    run("7 windows", BASE + MORE)


def exp_capacity():
    print("\n[capacity] trees, depth, learning rate")
    for lr, ne in [(0.03, 400), (0.03, 1600), (0.05, 800), (0.02, 1600)]:
        run(f"lr={lr} n_estimators={ne}", learning_rate=lr, n_estimators=ne)
    for nl in [15, 63, 127]:
        run(f"num_leaves={nl}", num_leaves=nl)
    for mcs in [20, 200]:
        run(f"min_child_samples={mcs}", min_child_samples=mcs)


def seeds(n=7, tags=BASE, opt=None, label="baseline", **kw):
    """Rank-averaged ensemble; also measures the seed-to-seed spread."""
    opt = opt or {}
    print(f"\n[seeds] {label}: {n} fits, rank-averaged")
    Xo, CUo, CIo, FB = evalset(**opt)
    acc = np.zeros(len(CUo)); singles = []
    for i in range(n):
        m = fit(tags, opt, random_state=lib.SEED + 1000 * i, **kw)
        v = m.predict(Xo)
        s1 = score_of(v, opt); singles.append(s1)
        acc += rank_norm(v, CUo)
        print(f"  seed {i}: single = {s1:.5f}   ensemble({i+1}) = {score_of(acc, opt):.5f}"
              f"  [{time.time()-_t0:.0f}s]", flush=True)
    a = np.array(singles)
    print(f"  spread: min {a.min():.5f}  max {a.max():.5f}  mean {a.mean():.5f}  sd {a.std():.5f}")
    return acc


def exp_seeds():
    seeds(7)


def exp_seeds3():
    seeds(7, opt=dict(feats3=True), label="41 features")


def exp_windows3():
    print("\n[windows3] number of training windows, on 41 features")
    o = dict(feats3=True)
    run("1 window (s0413)", ["s0413_lAB"], opt=o)
    run("2 windows", BASE[:2], opt=o)
    run("3 windows (reference)", BASE, opt=o)
    run("4 windows", BASE + MORE[:1], opt=o)
    run("5 windows", BASE + MORE[:2], opt=o)
    run("6 windows", BASE + MORE[:3], opt=o)


def exp_ablate():
    """Drop each new feature in turn: which of the 11 actually carries the gain."""
    import features3
    print("\n[ablate] leave-one-out over FEATS3 (41 features = 30 base + 11 new)")
    o = dict(feats3=True)
    full, _ = run("all 41", opt=o)
    Xo, CUo, CIo, FB = evalset(**o)
    for j, nm in enumerate(features3.FEATS3):
        keep = [c for c in range(41) if c != 30 + j]
        Xs, ys, Cs, off = [], [], [], 0
        for t in BASE:
            X, CU, CI, y = load(t, **o)
            Xs.append(X[:, keep]); ys.append(y); Cs.append(CU.astype(np.int64) + off)
            off += lib.N_USERS
        X = np.vstack(Xs); y = np.concatenate(ys); C = np.concatenate(Cs)
        ordr = np.argsort(C, kind="stable"); X, y, C = X[ordr], y[ordr], C[ordr]
        g = np.diff(np.flatnonzero(np.r_[True, C[1:] != C[:-1], True]))
        m = lgb.LGBMRanker(**PARAMS).fit(X, y, group=g)
        s = lib.evaluate(features.top5(m.predict(Xo[:, keep]), CUo, CIo, FB), GT)
        print(f"  without {nm:<12} MAP@5 = {s:.5f}   delta = {s-full:+.5f}"
              f"  [{time.time()-_t0:.0f}s]", flush=True)


DEAD = ["streak", "frac_max", "own7", "last_week"]   # positive leave-one-out deltas


def exp_bigknobs():
    """Only changes big enough to clear the noise. Both hold-outs, 3 seeds each.

    Measured noise: seed sd 0.00178 on A, and Kaggle's own public/private halves
    disagree by 0.005 on the SAME pair of models. So anything under ~0.005 that does
    not move both A and B in the same direction is not a result.
    """
    print("\n[bigknobs] capacity and objective, on 41 features, A and B")
    o = dict(feats3=True)
    run2("reference: 800 trees / 31 leaves / lr .03", opt=o)
    run2("1600 trees, lr 0.015", opt=o, n_estimators=1600, learning_rate=0.015)
    run2("400 trees, lr 0.06", opt=o, n_estimators=400, learning_rate=0.06)
    run2("num_leaves 127", opt=o, num_leaves=127)
    run2("num_leaves 15", opt=o, num_leaves=15)
    run2("objective rank_xendcg", opt=o, objective="rank_xendcg")


def exp_feats4():
    print("\n[feats4] the second added block, 3 seeds per configuration")
    runm("30 base", opt=dict())
    runm("41 = base + FEATS3", opt=dict(feats3=True))
    runm("51 = base + FEATS3 + FEATS4", opt=dict(feats3=True, feats4=True))
    runm("41 only FEATS4 (base + FEATS4)", opt=dict(feats4=True))


def exp_prune():
    print("\n[prune] drop the FEATS3 columns whose leave-one-out delta was positive")
    print(f"        dead list: {DEAD}")
    runm("41 full", opt=dict(feats3=True))
    runm("41 minus dead", opt=dict(feats3=True), drop=DEAD)
    runm("51 full", opt=dict(feats3=True, feats4=True))
    runm("51 minus dead", opt=dict(feats3=True, feats4=True), drop=DEAD)



def exp_capacity2():
    """Is the capacity trend real, or does it saturate? 400/800/1600 rose monotonically
    on BOTH hold-outs, which noise does not do. Push one more doubling."""
    print("\n[capacity2] does more capacity keep paying, on 41 features, A and B")
    o = dict(feats3=True)
    run2("1600 trees, lr 0.015 (current best)", opt=o, n_estimators=1600, learning_rate=0.015)
    run2("3200 trees, lr 0.0075", opt=o, n_estimators=3200, learning_rate=0.0075)


def exp_labels():
    """Threshold that counts as 'watched' when building the TRAINING label.

    Shipped: FRAC >= 1.0, ~3.5% positives. The evaluation target stays at 1.0 in both
    hold-outs -- only the training signal changes, so this asks whether the ranker
    learns better from more, noisier positives.
    """
    print("\n[labels] training-label threshold (evaluation target unchanged)")
    o = dict(feats3=True)
    run2("thr 1.00 (shipped)", opt=o)
    run2("thr 0.80", opt=o, thr=0.80)
    run2("thr 0.50", opt=o, thr=0.50)
    run2("thr 0.20", opt=o, thr=0.20)


def exp_wweight():
    """Windows are currently equal. The latest window is closest in time to the target."""
    print("\n[wweight] weighting the three training windows (order: s0413, s0330, s0427)")
    o = dict(feats3=True)
    run2("equal (shipped)", opt=o)
    run2("favour the latest window 1:1:2", opt=o, wweight=[1.0, 1.0, 2.0])
    run2("favour the latest window 1:0.5:2", opt=o, wweight=[1.0, 0.5, 2.0])


def exp_fuse():
    """Rank-fuse two models trained on DIFFERENT feature sets.

    Averaging seeds of one model bought +0.0006 -- nothing, because the fits make the
    same mistakes. 41 and 51 features are indistinguishable in score (Kaggle public
    +0.005 one way, private -0.00005 the other) but are built from different inputs,
    so their mistakes need not coincide. That is the condition under which fusion pays.
    """
    print("\n[fuse] rank-average of the 41-feature and 51-feature rankers, A and B")
    specs = [("41", dict(feats3=True)), ("51", dict(feats3=True, feats4=True))]

    for tags, eval_tag, gt, which in [(BASE, "outer", GT, "A"), (EVAL2, "outer2", GT2, "B")]:
        z = np.load(cache2.CACHE + eval_tag + ".npz")
        FB = [int(lib.ITEMS_ALL[k]) for k in np.argsort(-z["pop_u"])[:5]]
        per, fused = {k: [] for k, _ in specs}, []
        for i in range(3):
            ranks = []
            for nm, opt in specs:
                X, y, g = stack(tags, **opt)
                Xo, CUo, CIo, _ = load(eval_tag, **opt)
                p = dict(PARAMS); p["random_state"] = lib.SEED + 1000 * i
                m = lgb.LGBMRanker(**p).fit(X, y, group=g)
                v = m.predict(Xo)
                per[nm].append(lib.evaluate(features.top5(v, CUo, CIo, FB), gt))
                ranks.append(rank_norm(v, CUo))
                del X, y, m
            fused.append(lib.evaluate(
                features.top5(sum(ranks), CUo, CIo, FB), gt))
        for nm, _ in specs:
            a = np.array(per[nm])
            print(f"  [{which}] {nm} alone              {a.mean():.5f}+-{a.std():.5f}", flush=True)
        a = np.array(fused)
        print(f"  [{which}] 41+51 fused            {a.mean():.5f}+-{a.std():.5f}"
              f"  [{time.time()-_t0:.0f}s]", flush=True)


def exp_catboost():
    """A genuinely different family: CatBoost YetiRank.

    The 41/51 fusion failed because 51 is 41 plus ten inert columns -- the two models
    were never independent. CatBoost builds oblivious trees and optimises a different
    ranking loss, so its mistakes have a real chance of falling elsewhere. Measured
    alone first, then fused with LightGBM.
    """
    from catboost import CatBoostRanker, Pool
    print("\n[catboost] YetiRank alone, and fused with LambdaRank, on 41 features")
    o = dict(feats3=True)
    for tags, eval_tag, gt, which in [(BASE, "outer", GT, "A"), (EVAL2, "outer2", GT2, "B")]:
        z = np.load(cache2.CACHE + eval_tag + ".npz")
        FB = [int(lib.ITEMS_ALL[k]) for k in np.argsort(-z["pop_u"])[:5]]
        X, y, g = stack(tags, **o)
        grp = np.repeat(np.arange(len(g)), g)
        Xo, CUo, CIo, _ = load(eval_tag, **o)
        og = np.argsort(CUo, kind="stable")
        inv = np.empty(len(og), np.int64); inv[og] = np.arange(len(og))
        pool = Pool(X, label=y, group_id=grp)
        epool = Pool(Xo[og], group_id=CUo[og])
        cb, lg = [], []
        for i in range(3):
            m = CatBoostRanker(loss_function="YetiRank", iterations=800, depth=6,
                               learning_rate=0.05, random_seed=lib.SEED + 1000 * i,
                               verbose=0, thread_count=3, allow_writing_files=False)
            m.fit(pool)
            v = m.predict(epool)[inv]
            cb.append((lib.evaluate(features.top5(v, CUo, CIo, FB), gt), rank_norm(v, CUo)))
            p2 = dict(PARAMS); p2["random_state"] = lib.SEED + 1000 * i
            ml = lgb.LGBMRanker(**p2).fit(X, y, group=g)
            vl = ml.predict(Xo)
            lg.append((lib.evaluate(features.top5(vl, CUo, CIo, FB), gt), rank_norm(vl, CUo)))
            del m, ml
        fu = [lib.evaluate(features.top5(a[1] + b[1], CUo, CIo, FB), gt)
              for a, b in zip(cb, lg)]
        for nm, arr in [("LightGBM", [a[0] for a in lg]), ("CatBoost", [a[0] for a in cb]),
                        ("fused", fu)]:
            a = np.array(arr)
            print(f"  [{which}] {nm:<9} {a.mean():.5f}+-{a.std():.5f}"
                  f"  [{time.time()-_t0:.0f}s]", flush=True)
        del X, y

EXP = {"baseline": exp_baseline, "trunc": exp_trunc, "feats3": exp_feats3,
       "feats4": exp_feats4, "prune": exp_prune, "validate2": exp_validate2, "bigknobs": exp_bigknobs, "capacity2": exp_capacity2, "labels": exp_labels, "fuse": exp_fuse, "catboost": exp_catboost, "wweight": exp_wweight,
       "chfix": exp_chfix, "windows": exp_windows, "capacity": exp_capacity,
       "seeds": exp_seeds, "seeds3": exp_seeds3, "windows3": exp_windows3,
       "ablate": exp_ablate}

if __name__ == "__main__":
    for name in sys.argv[1:]:
        EXP[name]()
    print(f"done [{time.time()-_t0:.0f}s]")
