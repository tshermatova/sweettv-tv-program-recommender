"""Third added block, aimed at the two things the model already leans on hardest.

The fitted model's most used features are `ch_aff` and `ch_air_aff` — which channel,
and does it air on channels this person watches — followed by `frac_mean`, how fully
they watch it. Everything here extends one of those two, or repairs the data feeding
them:

  * time profiles rebuilt from the WHOLE view log instead of the 1/6 that is matched
    to a labelled programme (see src/39_timeprof.py)
  * channel affinity restricted to the recent past, so a changed habit is visible
  * the programme's own stickiness, averaged over all viewers — a property of the
    content rather than of the person
"""
import sys, importlib, numpy as np
sys.path.insert(0, "/Users/tshermatova/Downloads/week7-exam/src")
import lib

tp = importlib.import_module("39_timeprof")
chv = importlib.import_module("32_chviews")

_AR = np.load(lib.W + "airings_raw.npy")
_ARS = np.array([lib.IIDX_ALL.get(int(s), -1) for s in _AR[:, 0]], dtype=np.int32)
_CH_IDS = np.unique(_AR[:, 1])
_CHI = {int(c): i for i, c in enumerate(_CH_IDS)}
N_CH = len(_CH_IDS)

FEATS5 = ["hour_full", "dow_full", "slot_full", "ch_recent", "ch_air_recent",
          "item_frac", "item_rep", "item_reach", "pop_trend"]


def _nrm(M):
    n = np.linalg.norm(M, axis=1, keepdims=True); n[n == 0] = 1
    return M / n


def _dot(A, B, rows, cols, chunk=50000):
    out = np.zeros(len(rows), np.float32)
    for i in range(0, len(rows), chunk):
        j = min(i + chunk, len(rows))
        out[i:j] = (A[rows[i:j]] * B[cols[i:j]]).sum(1)
    return out


def extra(hist_mask, CU, CI, air_lo, air_hi):
    N = lib.N_ITEMS
    CUi, CIi = CU.astype(np.int64), CI.astype(np.int64)
    u = lib.UI[hist_mask].astype(np.int64)
    s = lib.SIDX[hist_mask].astype(np.int64)
    t = lib.PST[hist_mask]
    fr = lib.FRAC[hist_mask].astype(np.float32)
    t_end = int(t.max())

    # ---- target-window airings, by slot and by channel -----------------------
    i0 = np.searchsorted(_AR[:, 2], air_lo, "left")
    i1 = np.searchsorted(_AR[:, 2], air_hi, "right")
    ss, ast, ach = _ARS[i0:i1], _AR[i0:i1, 2], _AR[i0:i1, 1]
    ok = ss >= 0
    ss, ast, ach = ss[ok].astype(np.int64), ast[ok], ach[ok]
    hh = ((ast // 3600) % 24).astype(np.int64)
    dd = (((ast // 86400) + 4) % 7).astype(np.int64)
    IH = np.zeros((N, 24), np.float32); np.add.at(IH, (ss, hh), 1.0); IH = _nrm(IH)
    ID = np.zeros((N, 7), np.float32);  np.add.at(ID, (ss, dd), 1.0); ID = _nrm(ID)
    IS = np.zeros((N, 168), np.float32); np.add.at(IS, (ss, dd * 24 + hh), 1.0); IS = _nrm(IS)
    AC = np.zeros((N, N_CH), np.float32)
    np.add.at(AC, (ss, np.array([_CHI[int(c)] for c in ach], np.int64)), 1.0)
    AC = _nrm(AC)

    # ---- user profiles from the whole log ------------------------------------
    UH, UD, US = tp.profiles(air_lo)

    # ---- channel affinity over the last 28 days of history only --------------
    share_now, _ = chv.ch_share(air_lo)
    share_old, _ = chv.ch_share(air_lo - 28 * 86400)
    UCr = np.zeros((lib.N_USERS, N_CH), np.float32)
    for uu, v in share_now.items():
        o = share_old.get(int(uu), {})
        tot = 0.0
        row = {}
        for c, sec in v.items():
            d = sec - o.get(c, 0)
            if d > 0:
                k = _CHI.get(int(c))
                if k is not None:
                    row[k] = d; tot += d
        if tot:
            for k, d in row.items():
                UCr[int(uu), k] = d / tot
    UCr = _nrm(UCr)
    main = np.full(N, -1, np.int64)
    if len(ss):
        key = ss * 100000 + ach
        uk, c = np.unique(key, return_counts=True)
        ks, kc = uk // 100000, uk % 100000
        order = np.lexsort((-c, ks)); ks, kc = ks[order], kc[order]
        first = np.ones(len(ks), bool); first[1:] = ks[1:] != ks[:-1]
        main[ks[first]] = kc[first]
    ch_recent = np.zeros(len(CU), np.float32)
    for n_, (a, b) in enumerate(zip(CUi, CIi)):
        c = main[b]
        if c >= 0:
            k = _CHI.get(int(c))
            if k is not None:
                ch_recent[n_] = UCr[a, k]

    # ---- the programme's own stickiness, across all viewers ------------------
    item_fr = np.bincount(s, weights=fr, minlength=N) / np.maximum(np.bincount(s, minlength=N), 1)
    key = u * N + s
    uk, cnt = np.unique(key, return_counts=True)
    pi = (uk % N).astype(np.int64)
    item_reach = np.bincount(pi, minlength=N).astype(np.float32)          # distinct viewers
    item_rep = np.bincount(pi, weights=(cnt > 1).astype(np.float32), minlength=N) / \
        np.maximum(item_reach, 1)                                          # share who came back
    rec = t >= t_end - 28 * 86400
    p_rec = np.bincount(s[rec], minlength=N).astype(np.float32)
    p_all = np.bincount(s, minlength=N).astype(np.float32)
    pop_trend = p_rec / np.maximum(p_all, 1)

    return np.column_stack([
        _dot(UH, IH, CUi, CIi),
        _dot(UD, ID, CUi, CIi),
        _dot(US, IS, CUi, CIi),
        ch_recent,
        _dot(UCr, AC, CUi, CIi),
        item_fr[CIi].astype(np.float32),
        item_rep[CIi].astype(np.float32),
        np.log1p(item_reach[CIi]),
        pop_trend[CIi].astype(np.float32),
    ]).astype(np.float32)
