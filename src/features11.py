"""The programme's own broadcast history, which the feature set barely uses.

Everything airing-related in FEATS2 describes the TARGET window: how often the programme
airs there, on how many channels, in the first two weeks, in the last four. The training
weeks' schedule appears only as the denominator of `catch_rate` and as `air_tail`, the
three weeks immediately before the window.

That leaves the model unable to separate two very different candidates: a programme that
aired twenty times in weeks 11-30 and this user never watched it, and a premiere that
could not have been watched because it had not aired yet. Under the real target — the
frequency of qualifying airings — that distinction matters, because a premiere starts
everyone's count at zero while a long-running programme has an established pattern.
"""
import os
import sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib

FEATS11 = ["air_train", "air_train_4w", "days_since_air", "is_premiere",
           "air_growth", "air_span"]

_AR = np.load(lib.W + "airings_raw.npy")
_ARS = np.array([lib.IIDX_ALL.get(int(s), -1) for s in _AR[:, 0]], dtype=np.int32)


def _counts(lo, hi):
    i = np.searchsorted(_AR[:, 2], lo, "left")
    j = np.searchsorted(_AR[:, 2], hi, "right")
    s = _ARS[i:j]
    s = s[s >= 0].astype(np.int64)
    return np.bincount(s, minlength=lib.N_ITEMS).astype(np.float32)


def extra(CU, CI, air_lo, air_hi):
    """air_lo is the start of the window being predicted; everything before it is history."""
    N = lib.N_ITEMS
    train = _counts(0, air_lo)                       # the whole schedule before the window
    train_4w = _counts(air_lo - 28 * 86400, air_lo)
    future = _counts(air_lo, air_hi)

    # last airing before the window, and first airing ever, in days relative to air_lo
    j = np.searchsorted(_AR[:, 2], air_lo, "right")
    s_h = _ARS[:j]
    t_h = _AR[:j, 2]
    ok = s_h >= 0
    s_h, t_h = s_h[ok].astype(np.int64), t_h[ok]
    last = np.zeros(N, np.int64)
    np.maximum.at(last, s_h, t_h)
    first = np.full(N, 0, np.int64)
    tmp = np.full(N, 2**62, np.int64)
    np.minimum.at(tmp, s_h, t_h)
    first = np.where(tmp == 2**62, 0, tmp)

    days_since = np.where(last > 0, (air_lo - last) / 86400.0, 999.0).astype(np.float32)
    premiere = ((train == 0) & (future > 0)).astype(np.float32)
    growth = future / np.maximum(train, 1.0)
    span = np.where(first > 0, (last - first) / 86400.0, 0.0).astype(np.float32)

    CIi = CI.astype(np.int64)
    return np.column_stack([
        np.log1p(train[CIi]),
        np.log1p(train_4w[CIi]),
        np.log1p(days_since[CIi]),
        premiere[CIi],
        np.log1p(growth[CIi]),
        np.log1p(span[CIi]),
    ]).astype(np.float32)
