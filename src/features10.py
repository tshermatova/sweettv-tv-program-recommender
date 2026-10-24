"""More of what worked: variations on expected viewing frequency.

`features9` estimated the target quantity — the number of airings a user will watch at
screen time >= 0.8 — as sum over airings of P(slot) * P(channel), and it was worth
+0.0272 on the corrected hold-out against a seed spread of 0.00073. That is by far the
richest signal found in the project, so these are six refinements of the same estimate
rather than a new idea:

  decay    the slot profile from the last 28 days only, for users whose habit moved
  own_ch   restricted to the user's three main channels, where they actually are
  comp     multiplied by how completely they finish what they start — the >= 0.8 rule
           means starting is not enough
  early    airings in the first four weeks of the window only, since a user who drifts
           away still watched the early ones
  share    the estimate as a share of the user's own total, so heavy and light viewers
           become comparable
  per_air  the estimate divided by the number of airings: probability per airing rather
           than total volume
"""
import sys, numpy as np
sys.path.insert(0, "/Users/tshermatova/Downloads/week7-exam/src")
import lib, features

FEATS10 = ["ef_decay", "ef_own_ch", "ef_comp", "ef_early", "ef_share", "ef_per_air"]

_AR = np.load(lib.W + "airings_raw.npy")
_ARS = np.array([lib.IIDX_ALL.get(int(s), -1) for s in _AR[:, 0]], dtype=np.int32)
_CH_IDS = np.unique(_AR[:, 1])
_CHI = {int(c): i for i, c in enumerate(_CH_IDS)}
N_CH = len(_CH_IDS)


def _slot_profile(u, slot, w=None):
    P = np.zeros((lib.N_USERS, 168), np.float32)
    np.add.at(P, (u, slot), 1.0 if w is None else w)
    return P / np.maximum(P.sum(1, keepdims=True), 1.0)


def extra(hist_mask, CU, CI, air_lo, air_hi):
    N = lib.N_ITEMS
    u = lib.UI[hist_mask].astype(np.int64)
    d = lib.DAY[hist_mask]
    h = lib.HOUR[hist_mask].astype(np.int64)
    t = lib.PST[hist_mask]
    fr = lib.FRAC[hist_mask].astype(np.float32)
    slot = (((d + 4) % 7) * 24 + h).astype(np.int64)
    t_end = int(t.max())

    US_all = _slot_profile(u, slot)
    rec = t >= t_end - 28 * 86400
    US_rec = _slot_profile(u[rec], slot[rec]) if rec.any() else US_all

    # channel shares, and the three channels each user actually watches
    # Same cut as features9: channel shares must stop at the history boundary, not run
    # over the whole log, or the hold-out sees the window it is predicting.
    import importlib
    _share = importlib.import_module("32_chviews").ch_share(air_lo)[0]
    UC = np.zeros((lib.N_USERS, N_CH), np.float32)
    for uu, v in _share.items():
        tot = sum(v.values()) or 1
        for c, sec in v.items():
            k = _CHI.get(int(c))
            if k is not None:
                UC[int(uu), k] = sec / tot
    UC_top = np.zeros_like(UC)
    if N_CH > 3:
        idx3 = np.argpartition(-UC, 3, axis=1)[:, :3]
        rows = np.arange(lib.N_USERS)[:, None]
        UC_top[rows, idx3] = UC[rows, idx3]

    # completion rate: of the airings this user started, how fully do they watch
    comp = np.bincount(u, weights=(fr >= 0.8).astype(np.float64), minlength=lib.N_USERS) / \
        np.maximum(np.bincount(u, minlength=lib.N_USERS), 1)
    comp = comp.astype(np.float32)

    i0 = np.searchsorted(_AR[:, 2], air_lo, "left")
    i1 = np.searchsorted(_AR[:, 2], air_hi, "right")
    ss, ast, ach = _ARS[i0:i1], _AR[i0:i1, 2], _AR[i0:i1, 1]
    ok = ss >= 0
    ss, ast, ach = ss[ok].astype(np.int64), ast[ok], ach[ok]
    aslot = ((((ast // 86400) + 4) % 7) * 24 + (ast // 3600) % 24).astype(np.int64)
    achi = np.array([_CHI[int(c)] for c in ach], dtype=np.int64)
    early = ast < air_lo + 28 * 86400

    n_air = np.bincount(ss, minlength=N).astype(np.float32)

    CUi, CIi = CU.astype(np.int64), CI.astype(np.int64)
    want = {}
    for n_, (a, b) in enumerate(zip(CUi, CIi)):
        want.setdefault(int(b), []).append((int(a), n_))

    out = np.zeros((len(CU), 4), np.float32)     # decay, own_ch, comp, early
    order = np.argsort(ss, kind="stable")
    ss_s, sl_s, ch_s, ea_s = ss[order], aslot[order], achi[order], early[order]
    bnd = np.flatnonzero(np.r_[True, ss_s[1:] != ss_s[:-1], True])
    for k in range(len(bnd) - 1):
        it = int(ss_s[bnd[k]])
        rows = want.get(it)
        if not rows:
            continue
        a0, a1 = bnd[k], bnd[k + 1]
        sl, ch, ea = sl_s[a0:a1], ch_s[a0:a1], ea_s[a0:a1]
        for a, n_ in rows:
            ps_r, ps_a = US_rec[a, sl], US_all[a, sl]
            pc, pc_t = UC[a, ch], UC_top[a, ch]
            out[n_, 0] = float((ps_r * pc).sum())
            out[n_, 1] = float((ps_a * pc_t).sum())
            out[n_, 2] = float((ps_a * pc).sum()) * comp[a]
            out[n_, 3] = float((ps_a[ea] * pc[ea]).sum()) if ea.any() else 0.0

    base = out[:, 2] / np.maximum(comp[CUi], 1e-6)       # the undecayed joint estimate
    tot = np.zeros(lib.N_USERS, np.float64)
    np.add.at(tot, CUi, base.astype(np.float64))
    share = base / np.maximum(tot[CUi], 1e-9)
    per_air = base / np.maximum(n_air[CIi], 1.0)

    return np.column_stack([
        np.log1p(out[:, 0] * 100.0), np.log1p(out[:, 1] * 100.0),
        np.log1p(out[:, 2] * 100.0), np.log1p(out[:, 3] * 100.0),
        share.astype(np.float32), np.log1p(per_air * 1000.0),
    ]).astype(np.float32)
