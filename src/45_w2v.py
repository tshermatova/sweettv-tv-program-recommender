"""Word2Vec over viewing sequences: item embeddings that encode watch ORDER.

iALS already factorises the user-item matrix, but it only knows *that* a pair occurred.
Word2Vec reads each user's viewing history as a sentence, so two programmes end up close
when they are watched near each other in time — a different relation from co-occurrence
anywhere in a four-month window.

In the OTTO write-up this was one of the strongest candidate sources (recall@20 = 0.51).
Candidate generation is not our bottleneck — measured: widening the pool raises the MAP@5
ceiling by 0.005 while the model sits 0.34 below the existing ceiling — so these are used
as FEATURES instead: how close is the candidate to what this user has been watching.

Training is streamed and the model holds only a vocabulary of vectors (4 363 x 64), so it
runs comfortably beside a LightGBM job that has the RAM.
"""
import os
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib, features
from gensim.models import Word2Vec

DIM = 64
_t0 = time.time()


def sequences(split):
    """One sentence per user: their watched programmes, in time order."""
    hm = features.history_arrays(split)
    u = lib.UI[hm].astype(np.int64)
    s = lib.SIDX[hm].astype(np.int64)
    t = lib.PST[hm]
    o = np.lexsort((t, u))
    u, s = u[o], s[o]
    out, cur, buf = [], -1, []
    for a, b in zip(u, s):
        if a != cur:
            if len(buf) > 1:
                out.append(buf)
            cur, buf = a, []
        buf.append(str(b))
    if len(buf) > 1:
        out.append(buf)
    return out


def train(split, tag):
    f = lib.W + f"w2v_{tag}.npy"
    if os.path.exists(f):
        return np.load(f)
    seqs = sequences(split)
    n_tok = sum(len(s) for s in seqs)
    print(f"  {tag}: {len(seqs):,} последовательностей, {n_tok:,} событий "
          f"[{time.time()-_t0:.0f}s]", flush=True)
    m = Word2Vec(seqs, vector_size=DIM, window=5, min_count=3, sg=1, negative=5,
                 epochs=5, workers=2, seed=lib.SEED)
    E = np.zeros((lib.N_ITEMS, DIM), np.float32)
    for k in m.wv.index_to_key:
        E[int(k)] = m.wv[k]
    n = np.linalg.norm(E, axis=1, keepdims=True); n[n == 0] = 1
    E = E / n
    np.save(f, E)
    print(f"  {tag}: обучено, покрытие {100*(np.abs(E).sum(1) > 0).mean():.1f}% предметов "
          f"[{time.time()-_t0:.0f}s]", flush=True)
    return E


if __name__ == "__main__":
    import importlib
    c2 = importlib.import_module("30_cache2")
    jobs = [("outer", lib.B_AB), ("outer2", c2.B_EVAL2), ("final", lib.B_TEST)]
    for t in ["s0413_lAB", "s0330_l0504", "s0427_lAB"] + c2.EVAL2_WIN + c2.FINAL_WIN:
        jobs.append((t, c2.WIN[t][0]))
    seen = set()
    for tag, split in jobs:
        if split in seen:                       # windows that share a history cut share
            continue                            # their embedding; train it once
        seen.add(split)
        train(split, f"s{split}")
    print(f"готово [{time.time()-_t0:.0f}s]")
