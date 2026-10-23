"""Final model with the 41-feature set: fit in the FINAL regime, write a submission.

Hold-out regime measures; this regime ships. The three training windows are the
hold-out's winning geometry shifted forward by exactly 70 days (= B_TEST - B_AB),
so the fitted model sees the same window shapes, only later. Prediction uses all of
weeks 11-30 as history and weeks 31-42 only as the broadcast grid — the view log
stops at week 30, so no viewing behaviour from the test window exists to leak.

Optionally rank-averages several seeds (--seeds N).
"""
import sys, time, importlib, numpy as np, pickle
sys.path.insert(0, "/Users/tshermatova/Downloads/week7-exam/src")
import lib, features, features2, features3
import lightgbm as lgb

cache2 = importlib.import_module("30_cache2")
lab = importlib.import_module("31_lab")
_t0 = time.time()

N_SEEDS = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else 1
NAME = sys.argv[sys.argv.index("--name") + 1] if "--name" in sys.argv else "sub11_feats3_3win"
OPT = dict(feats3=True, feats4="--feats4" in sys.argv, feats5="--feats5" in sys.argv,
           feats6="--feats6" in sys.argv, feats7="--feats7" in sys.argv)

print(f"building final-regime datasets [{time.time()-_t0:.0f}s]", flush=True)
X, y, g = lab.stack(cache2.FINAL_WIN, **OPT)
print(f"fit set: {X.shape[0]:,} rows / {len(g):,} groups / {X.shape[1]} features "
      f"pos={y.mean():.4f} [{time.time()-_t0:.0f}s]", flush=True)

Xt, CUt, CIt, _ = lab.load("final", **OPT)
z = np.load(cache2.CACHE + "final.npz")
FB = [int(lib.ITEMS_ALL[k]) for k in np.argsort(-z["pop_u"])[:5]]
print(f"prediction set: {Xt.shape} fallback={FB} [{time.time()-_t0:.0f}s]", flush=True)

acc = np.zeros(len(CUt))
for i in range(N_SEEDS):
    p = dict(lab.PARAMS); p["random_state"] = lib.SEED + 1000 * i
    if "--big" in sys.argv:          # 1600/0.015: the one configuration change that
        p["n_estimators"] = 1600     # survived both hold-outs and a monotone sweep
        p["learning_rate"] = 0.015
    m = lgb.LGBMRanker(**p).fit(X, y, group=g)
    acc += lab.rank_norm(m.predict(Xt), CUt)
    print(f"  seed {i} fitted [{time.time()-_t0:.0f}s]", flush=True)
    if i == 0:
        imp = sorted(zip(lab.colnames(**OPT), m.feature_importances_),
                     key=lambda x: -x[1])
        print("  top features:", [(k, int(v)) for k, v in imp[:12]])
        pickle.dump(m, open(lib.W + "final_feats3.pkl", "wb"))

preds = features.top5(acc, CUt, CIt, FB)
path = lib.write_submission(preds, NAME, FB)
print(f"wrote {path} [{time.time()-_t0:.0f}s]")

prev = lib.S + "sub10_final_3win.csv"
import csv
a = {r[0]: r[1] for r in list(csv.reader(open(prev)))[1:]}
b = {r[0]: r[1] for r in list(csv.reader(open(path)))[1:]}
same = sum(1 for u in a if a[u] == b[u])
overlap = np.mean([len(set(a[u].split()) & set(b[u].split())) for u in a])
print(f"vs sub10: identical rows {same}/{len(a)}, mean shared ids per user {overlap:.2f}/5")
