"""Does co-visitation retrieval add candidates the current pool misses?

The pool today has two sources: the user's own history, and the globally most popular
80 programmes. The solutions that win these competitions use five to seven sources and
tune them against measured recall -- OTTO's write-up lists click-to-click co-visitation
at recall@20 = 0.51 on its own, which is as strong as its Word2Vec source.

Recall is a property of candidate generation alone, so it can be measured without
fitting anything. That matters here: our hold-out has stopped predicting the graded
metric, but recall is not a model-selection statistic and is not affected by that.

Measures, for each configuration: recall over all relevant items, and the MAP@5 ceiling
(min(hits,5)/min(|R|,5) averaged over users), which is what actually bounds the score.
"""
import os
import sys, importlib, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib, features

ex = importlib.import_module("33_extra")
_t0 = time.time()

GT = lib.target(lo=lib.B_AB, hi=lib.B_TEST)
USERS = [u for u in lib.TEST_UI if GT.get(u)]


def covisit_topk(hist_mask, k, weight="count"):
    """item -> its k most co-viewed items, from users who watched both."""
    u = lib.UI[hist_mask].astype(np.int64)
    s = lib.SIDX[hist_mask].astype(np.int64)
    uk = np.unique(u * lib.N_ITEMS + s)
    uu, su = (uk // lib.N_ITEMS).astype(np.int32), (uk % lib.N_ITEMS).astype(np.int32)
    n_u = int(uu.max()) + 1
    B = np.zeros((n_u, lib.N_ITEMS), np.float32)
    B[uu, su] = 1.0
    co = B.T @ B                                  # items x items, count of shared viewers
    supp = np.diag(co).copy()
    np.fill_diagonal(co, 0.0)
    if weight == "lift":
        co = (co * n_u) / np.maximum(np.outer(supp, supp), 1.0)
    elif weight == "conf":
        co = co / np.maximum(supp[:, None], 1.0)
    co[supp < 5, :] = 0.0
    idx = np.argpartition(-co, k, axis=1)[:, :k]
    rows = np.arange(lib.N_ITEMS)[:, None]
    order = np.argsort(-co[rows, idx], axis=1)
    top = idx[rows, order]
    keep = co[rows, top] > 0
    return top, keep


def measure(name, pools):
    tot = hit = 0
    cap, sizes, empty = [], [], 0
    for u in USERS:
        g = GT[u]
        p = pools.get(u, set())
        inter = len(g & p)
        tot += len(g); hit += inter
        cap.append(min(inter, 5) / min(len(g), 5))
        sizes.append(len(p))
        empty += inter == 0
    print(f"  {name:<40} кандидатов {np.mean(sizes):>6.0f}   recall {hit/tot:.4f}   "
          f"потолок MAP@5 {np.mean(cap):.4f}   пусто {100*empty/len(USERS):.1f}%"
          f"  [{time.time()-_t0:.0f}s]", flush=True)


if __name__ == "__main__":
    hm = features.history_arrays(lib.B_AB)
    _, CU, CI, _ = ex._base("outer")
    base = {}
    for a, b in zip(CU, CI):
        base.setdefault(int(a), set()).add(int(lib.ITEMS_ALL[b]))
    own = {}
    for a, b in zip(lib.UI[hm], lib.SIDX[hm]):
        own.setdefault(int(a), set()).add(int(b))

    measure("текущий пул (история + top-80)", base)
    print(f"  [co-visitation строится...] ", flush=True)

    for weight in ["count", "conf", "lift"]:
        top, keep = covisit_topk(hm, 20, weight)
        for k in [5, 10, 20]:
            pools = {}
            for u in USERS:
                p = set(base.get(u, set()))
                for i in own.get(u, ()):
                    p.update(int(lib.ITEMS_ALL[t]) for t, ok in
                             zip(top[i][:k], keep[i][:k]) if ok)
                pools[u] = p
            measure(f"+ co-visit {weight}, top-{k} на предмет истории", pools)
