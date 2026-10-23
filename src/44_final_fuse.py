"""Final submission from a rank-fusion of LightGBM lambdarank and CatBoost YetiRank.

The only candidate measured on the hold-outs and never sent to Kaggle: fusing the two
rankers scored +0.0015 on hold-out A and +0.0030 on hold-out B over LightGBM alone.
The evidence is weak — CatBoost alone flipped sign between the two instruments — but
the graded criterion is the highest private score over all submissions, so an unsent
measured candidate is worth the compute.

Both models are fitted on the same three final-regime windows and the same 41 features.
Predictions are combined by within-user reciprocal rank, which is scale-free; the raw
scores of a lambdarank model and a YetiRank model are not comparable.
"""
import sys, time, importlib, numpy as np
sys.path.insert(0, "/Users/tshermatova/Downloads/week7-exam/src")
import lib, features
import lightgbm as lgb
from catboost import CatBoostRanker, Pool

lab = importlib.import_module("31_lab")
cache2 = importlib.import_module("30_cache2")
_t0 = time.time()

N_SEEDS = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else 3
NAME = sys.argv[sys.argv.index("--name") + 1] if "--name" in sys.argv else "sub17_fuse_lgb_cat"
OPT = dict(feats3=True)

X, y, g = lab.stack(cache2.FINAL_WIN, **OPT)
grp = np.repeat(np.arange(len(g)), g)
Xt, CUt, CIt, _ = lab.load("final", **OPT)
z = np.load(cache2.CACHE + "final.npz")
FB = [int(lib.ITEMS_ALL[k]) for k in np.argsort(-z["pop_u"])[:5]]
print(f"fit set {X.shape}, prediction set {Xt.shape} [{time.time()-_t0:.0f}s]", flush=True)

# CatBoost needs its rows grouped by group_id, so predict on a sorted copy and invert
o = np.argsort(CUt, kind="stable")
inv = np.empty(len(o), np.int64); inv[o] = np.arange(len(o))
pool = Pool(X, label=y, group_id=grp)
epool = Pool(Xt[o], group_id=CUt[o])

acc = np.zeros(len(CUt))
for i in range(N_SEEDS):
    p = dict(lab.PARAMS)
    p.update(random_state=lib.SEED + 1000 * i, n_estimators=1600, learning_rate=0.015)
    ml = lgb.LGBMRanker(**p).fit(X, y, group=g)
    acc += lab.rank_norm(ml.predict(Xt), CUt)
    print(f"  lightgbm seed {i} [{time.time()-_t0:.0f}s]", flush=True)
    mc = CatBoostRanker(loss_function="YetiRank", iterations=800, depth=6,
                        learning_rate=0.05, random_seed=lib.SEED + 1000 * i,
                        verbose=0, thread_count=3, allow_writing_files=False).fit(pool)
    acc += lab.rank_norm(mc.predict(epool)[inv], CUt)
    print(f"  catboost seed {i} [{time.time()-_t0:.0f}s]", flush=True)
    del ml, mc

path = lib.write_submission(features.top5(acc, CUt, CIt, FB), NAME, FB)
print(f"wrote {path} [{time.time()-_t0:.0f}s]")
import csv
a = {r[0]: r[1] for r in list(csv.reader(open(lib.S + "sub13_feats3_1600.csv")))[1:]}
b = {r[0]: r[1] for r in list(csv.reader(open(path)))[1:]}
print(f"vs sub13: identical rows {sum(1 for u in a if a[u]==b[u])}/{len(a)}, "
      f"mean shared ids {np.mean([len(set(a[u].split()) & set(b[u].split())) for u in a]):.2f}/5")
