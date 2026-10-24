"""Build notebook.ipynb.

The source of every cell is stored as a list of lines, and nbformat requires each line
to keep its trailing newline — `split('\n')` drops them and the whole cell collapses
onto one line, which raises SyntaxError the moment anyone runs it. `splitlines(True)`
keeps them.
"""
import json

def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}

def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": t.splitlines(keepends=True)}

cells = [
md("""# sweet.tv — TV programme recommender

End-to-end walkthrough of the submitted solution: the interval join, the organisers'
target rule, the five required methods, the LambdaRank re-ranker that was shipped, and
the two-user comparison.

Designated Kaggle submission: `sub25_top5_s0.csv`, ref 57573516 —
**public 0.45440 / private 0.44205**.

Repository: https://github.com/tshermatova/sweettv-tv-program-recommender

Run with `OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2`; without them BLAS oversubscribes
on the small ALS solves and the machine slows by roughly a factor of ten. The heavy
steps are cached under `work/`, so a second run is fast.
"""),

md("""## 1. The data, and the one fact that shapes everything

The view log covers weeks 11-30 and never reaches the test window, so no viewing
behaviour from weeks 31-42 exists anywhere — there is nothing to leak. The schedule for
the test weeks is supplied and is used only to know which programme airs when.

**Only 10.8% of schedule rows carry a `tv_show_id`.** The other 89.2% is unlabelled
airtime: it can be watched but can never be recommended.
"""),
code('''import sys, os
sys.path.insert(0, "src")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import lib

print(f"join events         {len(lib.UI):,}")
print(f"users               {lib.N_USERS:,}")
print(f"labelled programmes {lib.N_ITEMS:,}")
print(f"test users          {len(lib.TEST_UI):,}")
print()
print(f"history threshold   FRAC >= {lib.FEAT_THR}")
print(f"target threshold    FRAC >= {lib.TGT_THR}  (the organisers' rule)")
'''),

md("""## 2. The join

A view matches a programme when they share a channel and their intervals overlap.
`frac = overlap / programme duration` cannot exceed 1.0, because the overlap is capped
by the programme itself. An event enters the history at `frac >= 0.20`; that threshold
was swept and 0.10 and 0.35 are both worse on both hold-outs.
"""),
code('''for q in (50, 75, 90, 95, 99):
    print(f"  p{q}: {np.percentile(lib.FRAC, q):.3f}")
print(f"  max: {lib.FRAC.max():.3f}")
print()
for t in (0.2, 0.5, 0.8, 1.0):
    m = lib.FRAC >= t
    print(f"  frac >= {t}: {m.sum():>9,} events ({100*m.mean():.2f}%)")
'''),

md("""## 3. The target, exactly as the organisers define it

> all entries with tv_show_id = 0 were removed; for each user and each TV show user
> screen time (time user watched the show divided by total duration of the show) was
> calculated. Only TV shows with high screen time (>= 0.8) were taken into account; for
> each user high-screen time TV shows frequency was calculated; for each user **strictly
> 5 most frequent** TV shows were selected DESC - order matters!

The relevant set is **exactly five programmes**, not every programme watched. An earlier
version of this work used `frac >= 1.0` and treated all qualifying programmes as
relevant, which read 0.5586 on the hold-out while Kaggle returned 0.43499. Correcting the
definition closed that gap to 0.003.
"""),
code('''GT = {u: set(v) for u, v in lib.target_top5(thr=0.80, lo=lib.B_AB, hi=lib.B_TEST).items()}
sizes = [len(GT[u]) for u in lib.TEST_UI if GT.get(u)]
print(f"test users with a target: {len(sizes)}/1260")
print(f"median |R| = {np.median(sizes):.0f}, exactly five for {100*np.mean([s == 5 for s in sizes]):.1f}%")
'''),

md("""## 4. MAP@5

Verified against the worked examples in the course slides: for relevant {A,B,C},
`A,B,D,C,E` scores 0.9167 and `D,A,E,B,C` scores 0.5333. The denominator is
`min(5, |relevant|)` and a repeated prediction counts once.
"""),
code('''R = {"A", "B", "C"}
print(f"A,B,D,C,E -> {lib.apk(R, ['A','B','D','C','E'], 5):.4f}  (slides: 0.917)")
print(f"D,A,E,B,C -> {lib.apk(R, ['D','A','E','B','C'], 5):.4f}  (slides: 0.533)")
'''),

md("""## 5. Candidates

Each user's own watched programmes plus the eighty most popular: about 139 per user.

Measured against the true target, perfect ordering of these candidates would score
0.8181 and the model scores 0.4395. **Ranking, not retrieval, is the binding
constraint** — adding every programme airing on each user's three main channels raises
the ceiling to 0.8325 and the model gains nothing (-0.0004).
"""),
code('''import importlib
ex = importlib.import_module("33_extra")
_, CU, CI, _ = ex._base("outer")
pool = {}
for a, b in zip(CU, CI):
    pool.setdefault(int(a), set()).add(int(lib.ITEMS_ALL[b]))
tu = [u for u in lib.TEST_UI if GT.get(u)]
rec = np.mean([len(GT[u] & pool.get(u, set())) / len(GT[u]) for u in tu])
cap = np.mean([min(len(GT[u] & pool.get(u, set())), 5) / min(len(GT[u]), 5) for u in tu])
print(f"candidates per user  {len(CU)/1260:.0f}")
print(f"recall               {rec:.4f}")
print(f"MAP@5 ceiling        {cap:.4f}")
'''),

md("""## 6. The five required methods

Popularity is the reference every method is measured against; association rules,
personalised PageRank, content-based filtering and iALS are fitted on the history window
in `src/methods.py` and reach the ranker as four score columns.

Full table: `src/55_method_table.py`, output in `work/log_methods.txt`.
"""),
code('''print(open("work/log_methods.txt").read())
'''),

md("""## 7. The shipped model

`LGBMRanker`, objective lambdarank, 41 features, three training windows, 1600 trees at
lr 0.015, rank-averaged over five seeds. The label is the organisers' own rule applied to
the training window.

That label is the single change that moved the score: +0.0079 on the hold-out and
**+0.00706 on Kaggle private**. Builder:
`src/36_final_feats3.py --big --seeds 5 --top5 --per-seed`.
"""),
code('''lab = importlib.import_module("31_lab")
import features2, features3
print(f"features: {len(features2.FEATS2)} base + {len(features3.FEATS3)} habit-shape "
      f"= {len(features2.FEATS2) + len(features3.FEATS3)}")
print()
for k, v in lab.PARAMS.items():
    print(f"  {k:<22} {v}")
'''),

md("""## 8. What was tried and rejected

Eleven directions were measured against the corrected target; ten were neutral or worse.
The table is in `report/section8.md` and the raw logs in `work/log_*.txt`. Two bound what
further work could achieve:

* expanding the pool with every programme on the user's three main channels raises the
  ceiling from 0.7974 to 0.8325 and changes the score by **-0.0004**;
* the top-20 of association rules adds **zero** new candidates, PageRank zero,
  content-based four, iALS about zero — on this catalogue those methods propose what
  popularity already proposed.
"""),

md("""## 9. The two users

`src/37_two_users.py` prints each method's top five with titles and HIT/miss against the
organisers' target; the full output is in `report/section7_two_users.txt`.

The finding that drives the discussion: the user with **one** recorded view is predicted
three times better than the user with 3 733, because for a sparse user the five most
frequent programmes simply are the mass-audience bulletins.
"""),
code('''print(open("report/section7_two_users.txt").read()[:1800])
'''),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11.4"}},
      "nbformat": 4, "nbformat_minor": 5}
with open("notebook.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"notebook.ipynb: {len(cells)} cells")
