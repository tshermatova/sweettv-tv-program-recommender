"""Window-aware channel affinity.

`work/ch_views.pkl` (built by src/01_join.py) accumulates seconds per (user, channel)
over the WHOLE view log, weeks 11-30. For the final model that is legitimate — the test
window is weeks 31-42 and the log does not reach it. For the calibrated hold-out it is
not: history is cut at B_AB but the channel feature still sees the target window
2020-05-18..07-27, so `ch_aff` / `ch_rank` are optimistic on validation and the ranker
tunes their weight against a signal that is weaker at test time.

This script re-streams the log into per-day (user, channel, seconds) totals, so channel
affinity can be recomputed with any cutoff.
"""
import os
import sys, os, csv, time, calendar, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib

OUT = lib.W + "ch_day.npz"
_t0 = time.time()


def ts_log(s):   # 'YYYY-MM-DD HH:MM:SS', as in dataset11-30.csv
    return calendar.timegm((int(s[0:4]), int(s[5:7]), int(s[8:10]),
                            int(s[11:13]), int(s[14:16]), int(s[17:19]), 0, 0, 0))


def build():
    agg = {}
    n = 0
    with open(lib.D + "dataset11-30.csv", newline="") as f:
        r = csv.reader(f); next(r)
        for row in r:
            n += 1
            ui = lib.UIDX.get(row[0])
            if ui is None:
                continue
            vs = ts_log(row[2]); ve = ts_log(row[3])
            k = (ui, int(row[1]), vs // 86400)
            agg[k] = agg.get(k, 0) + (ve - vs)
    A = np.array([(u, c, d, s) for (u, c, d), s in agg.items()], dtype=np.int64)
    A = A[np.argsort(A[:, 2], kind="stable")]           # sorted by day, for cutoffs
    np.savez_compressed(OUT, A=A)
    print(f"{n:,} view rows -> {len(A):,} (user, channel, day) totals [{time.time()-_t0:.0f}s]")
    return A


def load():
    if not os.path.exists(OUT):
        return build()
    return np.load(OUT)["A"]


_A = None


def ch_share(cut):
    """user_idx -> {channel: share of that user's seconds}, over days strictly before `cut`."""
    global _A
    if _A is None:
        _A = load()
    j = np.searchsorted(_A[:, 2], cut // 86400, "left")
    B = _A[:j]
    out = {}
    tot = {}
    for u, c, _, s in B:
        d = out.setdefault(int(u), {})
        d[int(c)] = d.get(int(c), 0) + int(s)
        tot[int(u)] = tot.get(int(u), 0) + int(s)
    return out, tot


if __name__ == "__main__":
    A = build()
    for cut, name in [(lib.B_AB, "B_AB"), (lib.B_TEST, "B_TEST")]:
        j = np.searchsorted(A[:, 2], cut // 86400, "left")
        print(f"  rows before {name}: {j:,} / {len(A):,}")
