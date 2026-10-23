"""Features from the Word2Vec item embeddings trained in src/45_w2v.py.

Each feature asks the same question in a different way: how close is this candidate to
what the user has actually been watching? iALS answers "did this pair occur"; these
answer "is this programme in the neighbourhood of their viewing", where neighbourhood
comes from watch order rather than from co-occurrence anywhere in the window.

Items seen fewer than three times get no embedding (min_count=3), so their features are
zero — coverage is 83% of items on the final window and as low as 35% on the short ones.
"""
import sys, os, numpy as np
sys.path.insert(0, "/Users/tshermatova/Downloads/week7-exam/src")
import lib, features

FEATS8 = ["w2v_mean", "w2v_recent", "w2v_max", "w2v_rank", "w2v_known"]
LAST_K = 20


def extra(split, CU, CI):
    f = lib.W + f"w2v_s{split}.npy"
    if not os.path.exists(f):
        return np.zeros((len(CU), len(FEATS8)), np.float32)
    E = np.load(f)                                   # (N_ITEMS, dim), L2-normalised
    hm = features.history_arrays(split)
    u = lib.UI[hm].astype(np.int64)
    s = lib.SIDX[hm].astype(np.int64)
    t = lib.PST[hm]

    # user profile: mean of everything watched, and mean of the last LAST_K watched
    P = np.zeros((lib.N_USERS, E.shape[1]), np.float32)
    np.add.at(P, u, E[s])
    cnt = np.maximum(np.bincount(u, minlength=lib.N_USERS), 1)[:, None]
    P = P / cnt

    o = np.lexsort((-t, u))                          # newest first within each user
    us, ss = u[o], s[o]
    starts = np.flatnonzero(np.r_[True, us[1:] != us[:-1]])
    R = np.zeros_like(P)
    for a, st in zip(us[starts], starts):
        R[a] = E[ss[st:st + LAST_K]].mean(0)

    def nrm(M):
        n = np.linalg.norm(M, axis=1, keepdims=True); n[n == 0] = 1
        return M / n
    P, R = nrm(P), nrm(R)

    # best match against any single watched item, per (user, candidate)
    hist = {}
    for a, b in zip(u, s):
        hist.setdefault(int(a), set()).add(int(b))

    CUi, CIi = CU.astype(np.int64), CI.astype(np.int64)
    mean_sim = (P[CUi] * E[CIi]).sum(1)
    rec_sim = (R[CUi] * E[CIi]).sum(1)
    max_sim = np.zeros(len(CU), np.float32)
    cache = {}
    for n_, (a, b) in enumerate(zip(CUi, CIi)):
        H = cache.get(int(a))
        if H is None:
            idx = np.fromiter(hist.get(int(a), ()), np.int64)
            H = E[idx] if len(idx) else np.zeros((0, E.shape[1]), np.float32)
            cache[int(a)] = H
        if len(H):
            max_sim[n_] = float((H @ E[b]).max())

    rk = np.zeros(len(CU), np.float32)
    ordr = np.lexsort((-mean_sim, CUi))
    _, i0 = np.unique(CUi[ordr], return_index=True)
    rk[ordr] = np.arange(len(CU)) - np.repeat(i0, np.diff(np.append(i0, len(CU))))

    return np.column_stack([
        mean_sim, rec_sim, max_sim, 1.0 / (1.0 + rk),
        (np.abs(E[CIi]).sum(1) > 0).astype(np.float32),   # does this item have a vector
    ]).astype(np.float32)
