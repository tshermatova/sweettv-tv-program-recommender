"""Extra per-(user, programme) signals appended after the 30 columns of FEATS2.

The existing set measures how much and how popular. These measure the shape of the
habit: how recently, how completely, how many weeks in a row, and whether the
programme still airs in the slot and on the channel that user actually watches.
"""
import os
import sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib

_AR = np.load(lib.W + "airings_raw.npy")                       # show, channel, start
_ARS = np.array([lib.IIDX_ALL.get(int(s), -1) for s in _AR[:, 0]], dtype=np.int32)
_CH_IDS = np.unique(_AR[:, 1])
_CHI = {int(c): i for i, c in enumerate(_CH_IDS)}
N_CH = len(_CH_IDS)

FEATS3 = ["own7", "own14", "frac_mean", "frac_max", "streak", "last_week",
          "slot_conc", "n_slots", "slot_aff", "ch_air_aff", "share14"]


def _norm(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1
    return M / n


def _chunked_dot(A, B, rows, cols, chunk=50000):
    """sum(A[rows] * B[cols], axis=1) without materialising the whole product."""
    out = np.zeros(len(rows), np.float32)
    for i in range(0, len(rows), chunk):
        j = min(i + chunk, len(rows))
        out[i:j] = (A[rows[i:j]] * B[cols[i:j]]).sum(1)
    return out


def extra(hist_mask, CU, CI, air_lo, air_hi, ch_share=None):
    """Returns (len(CU), len(FEATS3)) float32, aligned with the cached CU/CI."""
    N = lib.N_ITEMS
    u = lib.UI[hist_mask].astype(np.int64)
    s = lib.SIDX[hist_mask].astype(np.int64)
    t = lib.PST[hist_mask]
    d = lib.DAY[hist_mask]
    h = lib.HOUR[hist_mask].astype(np.int64)
    fr = lib.FRAC[hist_mask].astype(np.float32)
    t_end = int(t.max())
    slot = (((d + 4) % 7) * 24 + h).astype(np.int64)           # 0..167, Mon-based dow x hour

    key = u * N + s
    uk, inv = np.unique(key, return_inverse=True)
    inv = inv.ravel()
    npair = len(uk)
    cnt = np.bincount(inv, minlength=npair).astype(np.float32)
    c7 = np.bincount(inv[t >= t_end - 7 * 86400], minlength=npair).astype(np.float32)
    c14 = np.bincount(inv[t >= t_end - 14 * 86400], minlength=npair).astype(np.float32)

    o = np.argsort(inv, kind="stable")
    inv_s = inv[o]
    starts = np.flatnonzero(np.r_[True, inv_s[1:] != inv_s[:-1]])
    fr_sum = np.bincount(inv, weights=fr, minlength=npair).astype(np.float32)
    fr_max = np.maximum.reduceat(fr[o], starts).astype(np.float32)
    fr_mean = fr_sum / np.maximum(cnt, 1)

    # longest run of consecutive calendar weeks in which the pair occurs
    wk = np.unique(inv.astype(np.int64) * 100000 + (d // 7))
    wp, ww = wk // 100000, wk % 100000
    newrun = np.r_[True, (wp[1:] != wp[:-1]) | (ww[1:] != ww[:-1] + 1)]
    runlen = np.bincount(np.cumsum(newrun) - 1).astype(np.float32)
    streak = np.zeros(npair, np.float32)
    np.maximum.at(streak, wp[newrun], runlen)

    # how concentrated the pair is in one weekly slot
    sk, sc = np.unique(inv.astype(np.int64) * 1000 + slot, return_counts=True)
    sp = sk // 1000
    n_slots = np.bincount(sp, minlength=npair).astype(np.float32)
    slot_max = np.zeros(npair, np.float32)
    np.maximum.at(slot_max, sp, sc.astype(np.float32))
    slot_conc = slot_max / np.maximum(cnt, 1)

    u14 = np.bincount(u[t >= t_end - 14 * 86400], minlength=lib.N_USERS).astype(np.float32)

    # ---- airing profiles over the target window --------------------------------
    i0 = np.searchsorted(_AR[:, 2], air_lo, "left")
    i1 = np.searchsorted(_AR[:, 2], air_hi, "right")
    ss, ast, ach = _ARS[i0:i1], _AR[i0:i1, 2], _AR[i0:i1, 1]
    ok = ss >= 0
    ss, ast, ach = ss[ok], ast[ok], ach[ok]
    aslot = ((((ast // 86400 + 4) % 7) * 24 + (ast // 3600) % 24)).astype(np.int64)
    IP = np.zeros((N, 168), np.float32)
    np.add.at(IP, (ss.astype(np.int64), aslot), 1.0)
    IP = _norm(IP)
    UP = np.zeros((lib.N_USERS, 168), np.float32)
    np.add.at(UP, (u, slot), 1.0)
    UP = _norm(UP)

    AC = np.zeros((N, N_CH), np.float32)
    achi = np.array([_CHI[int(c)] for c in ach], dtype=np.int64)
    np.add.at(AC, (ss.astype(np.int64), achi), 1.0)
    AC = _norm(AC)
    if ch_share is None:
        ch_share = {uu: dict(v) for uu, v in lib.CH_VIEWS.items()}
    US = np.zeros((lib.N_USERS, N_CH), np.float32)
    for uu, v in ch_share.items():
        tot = sum(v.values()) or 1
        for c, sec in v.items():
            k = _CHI.get(int(c))
            if k is not None:
                US[int(uu), k] = sec / tot
    US = _norm(US)

    # ---- gather onto the candidate rows ---------------------------------------
    lut = {int(k): i for i, k in enumerate(uk)}
    idx = np.array([lut.get(int(a) * N + int(b), -1) for a, b in zip(CU, CI)])
    has = idx >= 0
    g = np.clip(idx, 0, None)

    def pick(arr, default=0.0):
        return np.where(has, arr[g], default).astype(np.float32)

    CUi = CU.astype(np.int64)
    CIi = CI.astype(np.int64)
    X = np.column_stack([
        np.log1p(pick(c7)),
        np.log1p(pick(c14)),
        pick(fr_mean),
        pick(fr_max),
        pick(streak),
        (pick(c7) > 0).astype(np.float32),
        pick(slot_conc),
        pick(n_slots),
        _chunked_dot(UP, IP, CUi, CIi),
        _chunked_dot(US, AC, CUi, CIi),
        pick(c14) / np.maximum(u14[CUi], 1),
    ]).astype(np.float32)
    return X
