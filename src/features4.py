"""Second round of added signals, aimed where the first round actually paid.

The leave-one-out over FEATS3 said the gain came from watch completeness (frac_mean),
not from the serial-continuity features (streak came out slightly negative). So this
block pushes on completeness and on repeat timing instead:

  * how completely the user watched this programme the LAST time, not on average
  * whether the user is "due" — time since the last view measured in units of that
    pair's own typical gap, which is the standard repeat-consumption signal
  * when the programme next airs inside the target window, and how regularly

Everything is computed from the history window only; the airing columns use the
target window's broadcast grid, which carries no viewing behaviour.
"""
import os
import sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib

_AR = np.load(lib.W + "airings_raw.npy")                       # show, channel, start
_ARS = np.array([lib.IIDX_ALL.get(int(s), -1) for s in _AR[:, 0]], dtype=np.int32)

FEATS4 = ["frac_last", "frac_p75", "cur_streak", "gap_mean", "overdue",
          "next_air_d", "air_weeks", "air_regular", "frac_rank", "u_frac_mean"]


def extra(hist_mask, CU, CI, air_lo, air_hi):
    N = lib.N_ITEMS
    u = lib.UI[hist_mask].astype(np.int64)
    s = lib.SIDX[hist_mask].astype(np.int64)
    t = lib.PST[hist_mask]
    d = lib.DAY[hist_mask]
    fr = lib.FRAC[hist_mask].astype(np.float32)
    t_end = int(t.max())
    last_week = int(d.max()) // 7

    key = u * N + s
    uk, inv = np.unique(key, return_inverse=True)
    inv = inv.ravel()
    npair = len(uk)
    cnt = np.bincount(inv, minlength=npair).astype(np.float32)

    # ---- completeness, ordered in time ------------------------------------
    o = np.lexsort((t, inv))                       # by pair, then by time
    inv_o, fr_o = inv[o], fr[o]
    ends = np.flatnonzero(np.r_[inv_o[1:] != inv_o[:-1], True])
    frac_last = np.zeros(npair, np.float32)
    frac_last[inv_o[ends]] = fr_o[ends]            # the most recent view of each pair
    starts = np.r_[0, ends[:-1] + 1]
    frac_p75 = np.zeros(npair, np.float32)
    for i, (a, b) in enumerate(zip(starts, ends + 1)):
        if b - a > 0:
            frac_p75[inv_o[a]] = np.percentile(fr_o[a:b], 75)

    # ---- repeat timing ----------------------------------------------------
    dk = np.unique(inv.astype(np.int64) * 100000 + d)           # distinct view-days
    dp, dd = dk // 100000, dk % 100000
    ndays = np.bincount(dp, minlength=npair).astype(np.float32)
    first_d = np.full(npair, 0, np.float32)
    last_d = np.full(npair, 0, np.float32)
    np.maximum.at(last_d, dp, dd.astype(np.float32))
    first_d.fill(1e9)
    np.minimum.at(first_d, dp, dd.astype(np.float32))
    span = np.maximum(last_d - first_d, 0.0)
    gap_mean = np.where(ndays > 1, span / np.maximum(ndays - 1, 1), 999.0)
    since = (t_end // 86400) - last_d
    overdue = np.where(ndays > 1, since / np.maximum(gap_mean, 1.0), 0.0)

    # consecutive weeks ending at the final week of history: a habit still running,
    # as opposed to FEATS3's `streak`, which counts the longest run anywhere
    wk = np.unique(inv.astype(np.int64) * 100000 + (d // 7))
    wp, ww = wk // 100000, wk % 100000
    cur = np.zeros(npair, np.float32)
    run = np.ones(len(wp), np.float32)
    for i in range(1, len(wp)):
        if wp[i] == wp[i - 1] and ww[i] == ww[i - 1] + 1:
            run[i] = run[i - 1] + 1
    tail = (ww == last_week)
    np.maximum.at(cur, wp[tail], run[tail])

    # within-user rank of completeness
    pu = (uk // N).astype(np.int64)
    fr_sum = np.bincount(inv, weights=fr, minlength=npair).astype(np.float32)
    fr_mean = fr_sum / np.maximum(cnt, 1)
    u_fr = np.bincount(pu, weights=fr_mean, minlength=lib.N_USERS) / \
        np.maximum(np.bincount(pu, minlength=lib.N_USERS), 1)
    ordr = np.lexsort((-fr_mean, pu))
    rk = np.zeros(npair, np.float32)
    _, i0 = np.unique(pu[ordr], return_index=True)
    rk[ordr] = np.arange(npair) - np.repeat(i0, np.diff(np.append(i0, npair)))
    frac_rank = 1.0 / (1.0 + rk)

    # ---- airings inside the target window ---------------------------------
    i0a = np.searchsorted(_AR[:, 2], air_lo, "left")
    i1a = np.searchsorted(_AR[:, 2], air_hi, "right")
    ss, ast = _ARS[i0a:i1a], _AR[i0a:i1a, 2]
    ok = ss >= 0
    ss, ast = ss[ok].astype(np.int64), ast[ok]
    next_air = np.full(N, 999.0, np.float32)
    np.minimum.at(next_air, ss, ((ast - air_lo) / 86400.0).astype(np.float32))
    aw = np.bincount(np.unique(ss * 100000 + (ast // 86400) // 7) // 100000,
                     minlength=N).astype(np.float32)
    acnt = np.bincount(ss, minlength=N).astype(np.float32)
    air_regular = aw / np.maximum(acnt, 1)          # 1.0 = exactly one airing per week

    # ---- gather -----------------------------------------------------------
    lut = {int(k): i for i, k in enumerate(uk)}
    idx = np.array([lut.get(int(a) * N + int(b), -1) for a, b in zip(CU, CI)])
    has = idx >= 0
    g = np.clip(idx, 0, None)
    CIi = CI.astype(np.int64)
    CUi = CU.astype(np.int64)

    def pick(arr, default=0.0):
        return np.where(has, arr[g], default).astype(np.float32)

    return np.column_stack([
        pick(frac_last),
        pick(frac_p75),
        pick(cur),
        np.log1p(pick(gap_mean, 999.0)),
        np.clip(pick(overdue), 0, 20),
        np.log1p(next_air[CIi]),
        np.log1p(aw[CIi]),
        air_regular[CIi],
        pick(frac_rank),
        u_fr[CUi].astype(np.float32),
    ]).astype(np.float32)
