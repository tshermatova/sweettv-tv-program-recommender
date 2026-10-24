"""Table 2 of the report: each required method as a standalone ranker.

All five methods rank the same 139 candidates per user and return that user's five
highest-scoring ones, so the comparison isolates the ranking signal and nothing else.
Scored against the organisers' target: each user's five most frequent airings at screen
time >= 0.8, in the hold-out window 2020-05-18 .. 2020-07-27.

The four collaborative scores are read from the cached feature matrix (columns 26-29 of
FEATS2), which is where the final model sees them, so the numbers here and the numbers
the ranker is given are the same numbers.
"""
import sys, importlib, numpy as np
sys.path.insert(0, "/Users/tshermatova/Downloads/week7-exam/src")
import lib, features

ex = importlib.import_module("33_extra")
c2 = importlib.import_module("30_cache2")

GT = {u: set(v) for u, v in lib.target_top5(thr=0.80, lo=lib.B_AB, hi=lib.B_TEST).items()}
X, CU, CI, _ = ex._base("outer")
z = np.load(c2.CACHE + "outer.npz")
FB = [int(lib.ITEMS_ALL[k]) for k in np.argsort(-z["pop_u"])[:5]]

H = lib.history(hi=lib.B_AB)
top5 = [s for s, _ in lib.popularity(H).most_common(5)]

print(f"{'method':<40}{'MAP@5':>10}")
print(f"{'popularity (required baseline)':<40}"
      f"{lib.evaluate({u: top5 for u in lib.TEST_UI}, GT):>10.5f}")
for name, col in [("association rules", 26), ("personalised PageRank", 27),
                  ("content-based filtering", 28), ("matrix factorisation (iALS)", 29)]:
    s = lib.evaluate(features.top5(X[:, col], CU, CI, FB), GT)
    print(f"{name:<40}{s:>10.5f}")

AR = np.load(lib.W + "airings_raw.npy")
i = np.searchsorted(AR[:, 2], lib.B_AB, "left")
j = np.searchsorted(AR[:, 2], lib.B_TEST, "right")
air = np.zeros(lib.N_ITEMS, np.float32)
for s_ in AR[i:j, 0]:
    if int(s_) in lib.IIDX_ALL:
        air[lib.IIDX_ALL[int(s_)]] += 1
own = X[:, 0] * np.log1p(air[CI])          # own_log x log(airings in the target window)
print(f"{'own repeats weighted by airings':<40}"
      f"{lib.evaluate(features.top5(own, CU, CI, FB), GT):>10.5f}")
print(f"{'final LambdaRank, 41 features':<40}{0.43949:>10.5f}"
      "   (src/46_true_target.py, 3 seeds)")
