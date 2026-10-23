"""Cache the features3 extras, and the window-aware channel-affinity replacements.

Two separate things live here because both depend on a time cutoff:
  * extras(tag)      -> the 11 FEATS3 columns for a cached window
  * ch_fix(tag)      -> honest ch_aff / ch_rank for columns 21 / 22 of the base X
"""
import sys, os, importlib, numpy as np
sys.path.insert(0, "/Users/tshermatova/Downloads/week7-exam/src")
import lib, features, features2, features3, features4, features5, features6, features7, features8, features9

cache2 = importlib.import_module("30_cache2")
chv = importlib.import_module("32_chviews")
CACHE = cache2.CACHE


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


def labels(tag, thr, min_n=1):
    """Recompute y at a different 'watched' threshold.

    The shipped label is FRAC >= 1.0 (the user's interval covered the whole programme),
    which leaves only ~3.5% positives. The feature matrix does not depend on the label,
    so a different threshold costs a set lookup, not a cache rebuild.
    """
    f = CACHE + f"{tag}_y{int(thr*100)}" + (f"n{min_n}" if min_n > 1 else "") + ".npy"
    if os.path.exists(f):
        return np.load(f)
    _, CU, CI, _ = _base(tag)
    split, hi = geom(tag)
    m = (lib.FRAC >= thr) & (lib.PST >= split) & (lib.PST < hi)
    key = lib.UI[m].astype(np.int64) * lib.N_ITEMS + lib.SIDX[m].astype(np.int64)
    uk, cnt = np.unique(key, return_counts=True)
    pos = set(uk[cnt >= min_n].tolist())
    y = np.fromiter((1 if (int(a) * lib.N_ITEMS + int(b)) in pos else 0
                     for a, b in zip(CU, CI)), np.int8, len(CU))
    np.save(f, y)
    print(f"   labels {tag:<14} thr={thr}  pos={y.mean():.4f}", flush=True)
    return y


def extras5(tag):
    """The 9 FEATS5 columns: whole-log time profiles, recent channel habit, item stickiness."""
    f = CACHE + tag + "_x5.npy"
    if os.path.exists(f):
        return np.load(f)
    _, CU, CI, _ = _base(tag)
    split, hi = geom(tag)
    X = features5.extra(features.history_arrays(split), CU, CI, split, hi)
    np.save(f, X)
    print(f"   extras5 {tag:<14} {X.shape}", flush=True)
    return X


def extras6(tag):
    """The 9 FEATS6 columns: category, genre, director, actor and year affinity."""
    f = CACHE + tag + "_x6.npy"
    if os.path.exists(f):
        return np.load(f)
    _, CU, CI, _ = _base(tag)
    split, _ = geom(tag)
    X = features6.extra(features.history_arrays(split), CU, CI)
    np.save(f, X)
    print(f"   extras6 {tag:<14} {X.shape}", flush=True)
    return X


def extras7(tag):
    """The 4 FEATS7 columns: the collaborative block refitted on recent history only."""
    f = CACHE + tag + "_x7.npy"
    if os.path.exists(f):
        return np.load(f)
    _, CU, CI, _ = _base(tag)
    split, _ = geom(tag)
    X = features7.extra(split, CU, CI)
    np.save(f, X)
    print(f"   extras7 {tag:<14} {X.shape}", flush=True)
    return X


def extras8(tag):
    """The 5 FEATS8 columns: Word2Vec proximity between the candidate and the history."""
    f = CACHE + tag + "_x8.npy"
    if os.path.exists(f):
        return np.load(f)
    _, CU, CI, _ = _base(tag)
    split, _ = geom(tag)
    X = features8.extra(split, CU, CI)
    np.save(f, X)
    print(f"   extras8 {tag:<14} {X.shape}", flush=True)
    return X


def labels_top5(tag, thr=None):
    """Label = the pair is in that user's five most frequent qualifying airings.

    This is the organisers' own target, applied to the training window: screen time
    >= 0.8 counts an airing, airings are counted per (user, show), and only the top five
    by that count are positives. Exactly the thing the model is scored on, instead of
    "watched anything at all", which is what it has been trained on until now.
    """
    thr = lib.TGT_THR if thr is None else thr
    f = CACHE + f"{tag}_ytop5_{int(thr*100)}.npy"
    if os.path.exists(f):
        return np.load(f)
    _, CU, CI, _ = _base(tag)
    split, hi = geom(tag)
    tgt = lib.target_top5(thr=thr, lo=split, hi=hi)
    pos = set()
    for u, items in tgt.items():
        for it in items:
            pos.add(u * lib.N_ITEMS + lib.IIDX_ALL[int(it)])
    y = np.fromiter(((1 if (int(a) * lib.N_ITEMS + int(b)) in pos else 0)
                     for a, b in zip(CU, CI)), np.int8, len(CU))
    np.save(f, y)
    print(f"   labels_top5 {tag:<14} pos={y.mean():.4f}", flush=True)
    return y


def extras9(tag):
    """The 5 FEATS9 columns: an explicit estimate of expected viewing frequency."""
    f = CACHE + tag + "_x9.npy"
    if os.path.exists(f):
        return np.load(f)
    _, CU, CI, _ = _base(tag)
    split, hi = geom(tag)
    X = features9.extra(features.history_arrays(split), CU, CI, split, hi)
    np.save(f, X)
    print(f"   extras9 {tag:<14} {X.shape}", flush=True)
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
