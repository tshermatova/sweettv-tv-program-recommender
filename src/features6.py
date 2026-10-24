"""Content affinity, broken out instead of collapsed into one similarity score.

88% of the 4 363 labelled programmes are films, and the schedule gives category, three
genre slots, year, director and actors with 97-100% coverage. All of that currently
reaches the ranker as a single `cbf` column — one aggregated similarity — so the model
cannot learn that a particular viewer watches action and not melodrama, follows one
director, or dislikes anything made before 2000.

Each feature here is the overlap between a profile built from the user's own history
and the attributes of the candidate programme. Everything is computed from the history
window; the schedule rows the attributes come from are static facts about a programme,
not behaviour, so using them for the target window leaks nothing.
"""
import os
import sys, numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib

FEATS6 = ["cat_aff", "genre_aff", "genre_top", "dir_aff", "act_aff",
          "year_gap", "year_aff", "item_year", "n_genres"]


def _split_actors(s):
    return [a.strip() for a in str(s).split(",") if a.strip()]


# ---- static attribute tables, built once -----------------------------------------
_CAT, _GEN, _DIR, _ACT, _YEAR = {}, {}, {}, {}, {}
for _s in lib.ITEMS_ALL:
    m = lib.SHOW_META.get(int(_s))
    i = lib.IIDX_ALL[int(_s)]
    if not m:
        _CAT[i] = ""; _GEN[i] = (); _DIR[i] = ""; _ACT[i] = (); _YEAR[i] = 0
        continue
    _CAT[i] = (m[1] or "").strip()
    _GEN[i] = tuple(g.strip() for g in m[2:5] if str(g).strip())
    _DIR[i] = (m[6] or "").strip() if len(m) > 6 else ""
    _ACT[i] = tuple(_split_actors(m[7])) if len(m) > 7 else ()
    try:
        _YEAR[i] = int(m[5])
    except (ValueError, TypeError):
        _YEAR[i] = 0

_CATS = sorted({v for v in _CAT.values() if v})
_CATI = {c: k for k, c in enumerate(_CATS)}
_GENS = sorted({g for v in _GEN.values() for g in v})
_GENI = {g: k for k, g in enumerate(_GENS)}


def extra(hist_mask, CU, CI):
    """Returns (len(CU), len(FEATS6)) float32 aligned with the cached CU/CI."""
    u = lib.UI[hist_mask].astype(np.int64)
    s = lib.SIDX[hist_mask].astype(np.int64)

    # ---- per-user profiles over watched attributes --------------------------
    ucat = np.zeros((lib.N_USERS, len(_CATS)), np.float32)
    ugen = np.zeros((lib.N_USERS, len(_GENS)), np.float32)
    udir = defaultdict(Counter)
    uact = defaultdict(Counter)
    uyear = defaultdict(list)
    for a, b in zip(u, s):
        c = _CAT.get(b, "")
        if c:
            ucat[a, _CATI[c]] += 1.0
        for g in _GEN.get(b, ()):
            ugen[a, _GENI[g]] += 1.0
        d = _DIR.get(b, "")
        if d:
            udir[int(a)][d] += 1
        for ac in _ACT.get(b, ()):
            uact[int(a)][ac] += 1
        y = _YEAR.get(b, 0)
        if y:
            uyear[int(a)].append(y)

    ntot = np.maximum(ucat.sum(1, keepdims=True), 1.0)
    ucat_n = ucat / ntot
    gtot = np.maximum(ugen.sum(1, keepdims=True), 1.0)
    ugen_n = ugen / gtot
    gtop = ugen.argmax(1)
    umean_year = {k: float(np.mean(v)) for k, v in uyear.items()}
    dtot = {k: sum(c.values()) for k, c in udir.items()}
    atot = {k: sum(c.values()) for k, c in uact.items()}

    n = len(CU)
    out = np.zeros((n, len(FEATS6)), np.float32)
    for r in range(n):
        a = int(CU[r]); b = int(CI[r])
        c = _CAT.get(b, "")
        if c:
            out[r, 0] = ucat_n[a, _CATI[c]]
        gs = _GEN.get(b, ())
        if gs:
            idx = [_GENI[g] for g in gs]
            out[r, 1] = ugen_n[a, idx].sum()
            out[r, 2] = 1.0 if gtop[a] in idx else 0.0
        d = _DIR.get(b, "")
        if d:
            out[r, 3] = udir[a].get(d, 0) / max(dtot.get(a, 1), 1)
        acs = _ACT.get(b, ())
        if acs:
            ua = uact.get(a)
            if ua:
                out[r, 4] = sum(ua.get(x, 0) for x in acs) / max(atot.get(a, 1), 1)
        y = _YEAR.get(b, 0)
        my = umean_year.get(a, 0.0)
        out[r, 5] = abs(y - my) / 50.0 if (y and my) else 1.0
        out[r, 6] = 1.0 / (1.0 + abs(y - my) / 5.0) if (y and my) else 0.0
        out[r, 7] = (y - 1950) / 70.0 if y else 0.0
        out[r, 8] = len(gs)
    return out
