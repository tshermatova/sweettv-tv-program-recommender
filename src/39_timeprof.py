"""When does a user actually watch television? Built from the WHOLE log.

`features2` estimates each user's hour-of-day and day-of-week profile from
`history_arrays`, i.e. from matched (user, programme) events only — 888 573 of them
before B_AB. But 89.2% of the broadcast grid carries no `tv_show_id`, so a view that
landed on unlabelled airtime contributes nothing to the profile, even though it says
exactly as much about when this person watches. The full log has 5.8M view rows.

This streams the log once into per (user, day, hour) second totals, so an honest
profile can be cut at any date. Channel affinity already uses the whole log and is the
single most used feature in the model; the time profiles were left on the 1/6 diet.
"""
import os
import sys, os, csv, time, calendar, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib

OUT = lib.W + "time_prof.npz"
_t0 = time.time()


def ts_log(s):   # 'YYYY-MM-DD HH:MM:SS'
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
            k = (ui, vs // 86400, (vs // 3600) % 24)
            agg[k] = agg.get(k, 0) + (ve - vs)
    A = np.array([(u, d, h, s) for (u, d, h), s in agg.items()], dtype=np.int64)
    A = A[np.argsort(A[:, 1], kind="stable")]           # by day, for cutoffs
    np.savez_compressed(OUT, A=A)
    print(f"{n:,} view rows -> {len(A):,} (user, day, hour) totals [{time.time()-_t0:.0f}s]")
    return A


_A = None


def load():
    global _A
    if _A is None:
        _A = np.load(OUT)["A"] if os.path.exists(OUT) else build()
    return _A


def profiles(cut):
    """(hour profile 24, dow profile 7, slot profile 168), L2-normalised, days < cut."""
    A = load()
    j = np.searchsorted(A[:, 1], cut // 86400, "left")
    u, d, h, s = A[:j, 0], A[:j, 1], A[:j, 2], A[:j, 3].astype(np.float32)
    dow = (d + 4) % 7
    H = np.zeros((lib.N_USERS, 24), np.float32); np.add.at(H, (u, h), s)
    D = np.zeros((lib.N_USERS, 7), np.float32);  np.add.at(D, (u, dow), s)
    S = np.zeros((lib.N_USERS, 168), np.float32); np.add.at(S, (u, dow * 24 + h), s)

    def nrm(M):
        k = np.linalg.norm(M, axis=1, keepdims=True); k[k == 0] = 1
        return M / k
    return nrm(H), nrm(D), nrm(S)


if __name__ == "__main__":
    A = build()
    H, D, S = profiles(lib.B_AB)
    print(f"  users with a non-empty profile: {(H.sum(1) > 0).sum():,} / {lib.N_USERS:,}")
