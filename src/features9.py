"""Expected frequency: an explicit estimate of the quantity the metric is built on.

The organisers' answer file is each user's five programmes with the highest **frequency**
of airings watched at screen time >= 0.8. So the target is a count, and that count
decomposes:

    expected_frequency(u, p) = sum over airings a of p in weeks 31-42 of P(u watches a)

and P(u watches a) factorises through the channel a runs on and the weekly slot it starts
in, both of which we already estimate. Everything needed has been in the feature matrix
for two days — as separate factors. A gradient-boosted tree splits on axes; it does not
multiply them, so handing it the factors is not the same as handing it the product.

Three variants of the same idea, differing in how P(watch) is formed, plus the count of
airings that fall in slots the user is ever active in — a hard filter rather than a
weighting.
"""
import sys, numpy as np
sys.path.insert(0, "/Users/tshermatova/Downloads/week7-exam/src")
import lib, features

FEATS9 = ["exp_freq_ch", "exp_freq_slot", "exp_freq_both", "n_air_live", "exp_freq_rank"]

_AR = np.load(lib.W + "airings_raw.npy")
_ARS = np.array([lib.IIDX_ALL.get(int(s), -1) for s in _AR[:, 0]], dtype=np.int32)
_CH_IDS = np.unique(_AR[:, 1])
_CHI = {int(c): i for i, c in enumerate(_CH_IDS)}
N_CH = len(_CH_IDS)


def extra(hist_mask, CU, CI, air_lo, air_hi, ch_share=None):
    N = lib.N_ITEMS
    u = lib.UI[hist_mask].astype(np.int64)
    d = lib.DAY[hist_mask]
    h = lib.HOUR[hist_mask].astype(np.int64)
    slot = (((d + 4) % 7) * 24 + h).astype(np.int64)

    # P(user is watching at all in weekly slot s) — share of their events in that slot
    US = np.zeros((lib.N_USERS, 168), np.float32)
    np.add.at(US, (u, slot), 1.0)
    US = US / np.maximum(US.sum(1, keepdims=True), 1.0)

    # P(user is on channel c) — share of their seconds, from the full log
    if ch_share is None:
        ch_share = lib.CH_VIEWS
    UC = np.zeros((lib.N_USERS, N_CH), np.float32)
    for uu, v in ch_share.items():
        tot = sum(v.values()) or 1
        for c, sec in v.items():
            k = _CHI.get(int(c))
            if k is not None:
                UC[int(uu), k] = sec / tot

    # every airing of every candidate programme inside the target window
    i0 = np.searchsorted(_AR[:, 2], air_lo, "left")
    i1 = np.searchsorted(_AR[:, 2], air_hi, "right")
    ss, ast, ach = _ARS[i0:i1], _AR[i0:i1, 2], _AR[i0:i1, 1]
    ok = ss >= 0
    ss, ast, ach = ss[ok].astype(np.int64), ast[ok], ach[ok]
    aslot = ((((ast // 86400) + 4) % 7) * 24 + (ast // 3600) % 24).astype(np.int64)
    achi = np.array([_CHI[int(c)] for c in ach], dtype=np.int64)

    # Accumulate per (item, slot) and per (item, channel) airing counts once, then the
    # per-candidate sums are two matrix products instead of a loop over airings.
    A_slot = np.zeros((N, 168), np.float32)
    np.add.at(A_slot, (ss, aslot), 1.0)
    A_ch = np.zeros((N, N_CH), np.float32)
    np.add.at(A_ch, (ss, achi), 1.0)
    # the joint term needs slot and channel together, so keep the triples
    live_slot = (US > 0)

    CUi, CIi = CU.astype(np.int64), CI.astype(np.int64)

    def chunked(Umat, Amat, chunk=40000):
        out = np.zeros(len(CU), np.float32)
        for a in range(0, len(CU), chunk):
            b = min(a + chunk, len(CU))
            out[a:b] = (Umat[CUi[a:b]] * Amat[CIi[a:b]]).sum(1)
        return out

    exp_ch = chunked(UC, A_ch)
    exp_slot = chunked(US, A_slot)

    # joint: sum over airings of P(slot) * P(channel); done per airing, accumulated into
    # a (user, item) lookup only for the pairs we actually need
    want = {}
    for n_, (a, b) in enumerate(zip(CUi, CIi)):
        want.setdefault(int(b), []).append((int(a), n_))
    exp_both = np.zeros(len(CU), np.float32)
    order = np.argsort(ss, kind="stable")
    ss_s, aslot_s, achi_s = ss[order], aslot[order], achi[order]
    bnd = np.flatnonzero(np.r_[True, ss_s[1:] != ss_s[:-1], True])
    for k in range(len(bnd) - 1):
        it = int(ss_s[bnd[k]])
        rows = want.get(it)
        if not rows:
            continue
        sl = aslot_s[bnd[k]:bnd[k + 1]]
        ch = achi_s[bnd[k]:bnd[k + 1]]
        for a, n_ in rows:
            exp_both[n_] = float((US[a, sl] * UC[a, ch]).sum())

    n_live = chunked(live_slot.astype(np.float32), A_slot)

    rk = np.zeros(len(CU), np.float32)
    ordr = np.lexsort((-exp_both, CUi))
    _, i_0 = np.unique(CUi[ordr], return_index=True)
    rk[ordr] = np.arange(len(CU)) - np.repeat(i_0, np.diff(np.append(i_0, len(CU))))

    return np.column_stack([
        np.log1p(exp_ch), np.log1p(exp_slot), np.log1p(exp_both * 100.0),
        np.log1p(n_live), 1.0 / (1.0 + rk),
    ]).astype(np.float32)
