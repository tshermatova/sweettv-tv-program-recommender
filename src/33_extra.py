"""Cache the features3 extras, and the window-aware channel-affinity replacements.

Two separate things live here because both depend on a time cutoff:
  * extras(tag)      -> the 11 FEATS3 columns for a cached window
  * ch_fix(tag)      -> honest ch_aff / ch_rank for columns 21 / 22 of the base X
"""
import sys, os, importlib, numpy as np
sys.path.insert(0, "/Users/tshermatova/Downloads/week7-exam/src")
import lib, features, features2, features3, features4

cache2 = importlib.import_module("30_cache2")
chv = importlib.import_module("32_chviews")
CACHE = lib.W + "ds2/"


# evaluation sets are not in WIN: they carry no labels and their own builder
EVALSETS = {
    "outer":  (lambda: (lib.B_AB, lib.B_TEST),      lambda: cache2.outer()),
    "outer2": (lambda: (cache2.B_EVAL2, lib.B_AB),  lambda: cache2.outer2()),
    "final":  (lambda: (lib.B_TEST, cache2.T_END),  lambda: cache2.final()),
}


def geom(tag):
    """(history cut, airing-window end) for a cached dataset tag."""
    if tag in EVALSETS:
        return EVALSETS[tag][0]()
    return cache2.WIN[tag]


def _base(tag):
    if tag in EVALSETS:
        return EVALSETS[tag][1]()
    return cache2.get(tag)


def extras(tag, honest_ch=True):
    f = CACHE + tag + ("_x3h.npy" if honest_ch else "_x3.npy")
    if os.path.exists(f):
        return np.load(f)
    _, CU, CI, _ = _base(tag)
    split, hi = geom(tag)
    share = chv.ch_share(split)[0] if honest_ch else None
    X = features3.extra(features.history_arrays(split), CU, CI, split, hi, ch_share=share)
    np.save(f, X)
    print(f"   extras {tag:<14} {X.shape}", flush=True)
    return X


def extras4(tag):
    """The 10 FEATS4 columns: completeness detail, repeat timing, target-window airings."""
    f = CACHE + tag + "_x4.npy"
    if os.path.exists(f):
        return np.load(f)
    _, CU, CI, _ = _base(tag)
    split, hi = geom(tag)
    X = features4.extra(features.history_arrays(split), CU, CI, split, hi)
    np.save(f, X)
    print(f"   extras4 {tag:<14} {X.shape}", flush=True)
    return X


def ch_fix(tag):
    """Recompute FEATS2 columns 21 (ch_aff) and 22 (ch_rank) with history-only channel views."""
    f = CACHE + tag + "_chfix.npy"
    if os.path.exists(f):
        return np.load(f)
    _, CU, CI, _ = _base(tag)
    split, hi = geom(tag)
    share, _ = chv.ch_share(split)
    main = features2.air_block(split, hi)["main"]
    aff = np.zeros(len(CU), np.float32)
    rnk = np.zeros(len(CU), np.float32)
    for n, (a, b) in enumerate(zip(CU, CI)):
        ch = main[b]
        v = share.get(int(a))
        if v and ch >= 0:
            tot = sum(v.values()) or 1
            x = v.get(int(ch), 0)
            aff[n] = x / tot
            srt = sorted(v.values(), reverse=True)
            rnk[n] = 1.0 / (1.0 + (srt.index(x) if x in srt else 99))
    out = np.column_stack([aff, rnk]).astype(np.float32)
    np.save(f, out)
    print(f"   ch_fix {tag:<14} {out.shape}", flush=True)
    return out


if __name__ == "__main__":
    for t in sys.argv[1:]:
        extras(t)
        ch_fix(t)
