# Session state — read this first after a restart

Project root: `/Users/tshermatova/Downloads/week7-exam/`
Exam: sweet.tv TV programme recommender, MAP@5, deadline **24 Oct 2026 19:00 MSK**.
Spec lives in the conversation as `exam_quiz_GSB.md`; consultation transcript as a PDF.

## Where things stand

Best Kaggle result so far — `sub09_lambdarank.csv`, ref 57495736:
**public 0.43135 / private 0.41924**. The original 2020 winner scored 0.42912, so the
public score is already past it. 9 submissions made; the daily cap is 99, not 5.

Hold-out progression (calibrated hold-out, see below):
popularity 0.40040 -> own repeats 0.48054 -> LR hybrid 0.51141 -> LambdaRank 0.53003 ->
**3 training windows 0.54464 (best)**.

## Decisions already settled (do not re-litigate)

| question | answer | evidence |
| --- | --- | --- |
| join rule | same channel + interval overlap >= 20% of programme duration | `src/02_explore.py` distributions |
| `tv_show_id = 0` | never an item; still counts for channel affinity | 89.2% of the grid |
| hold-out | features < 2020-05-18, target >= that date, target = `frac >= 1.0` | `src/08_calibrate2.py`: this reproduces Kaggle's ordering of 5 probe submissions exactly |
| repeats allowed? | yes — repeats are the single strongest signal | probe 0.41510 vs popularity 0.29788 |
| cold start | there is none: all 1 260 test users appear in the log | `src/00_probe.py` |
| best model | LGBMRanker, lambdarank, 30 features, **3 training windows**, binary labels | `src/25_graded.py` |

Tried and rejected, with numbers (all belong in the report's limitations):
restricting candidates to programmes airing in weeks 31-42 (no effect at all);
weighting by watch seconds (-0.12 Kaggle); time decay (+0.004, noise);
recovering `tv_show_id` from `tv_show_title` (coverage 11.1% -> 12.2% only);
wider candidate pools (oracle +0.04, model -0.003); graded labels (-0.003);
extended 9-signal collaborative block (-0.001); pointwise GBM and within-user
normalisation (both below plain LR).

Diagnostic that sets the ceiling: oracle inside the candidate pool = **0.85963**,
model = 0.530. Ranking, not candidate generation, is the binding constraint.

## File map

```
src/lib.py          data layer, calibrated hold-out, MAP@5          <- start here
src/methods.py      popularity, iALS, CBF, association rules, PPR
src/features2.py    the 30 features (FEATS2)
src/collab2.py      extended collaborative block (tested, rejected)
src/01_join.py      interval join -> work/matches.npy
src/25_graded.py    best configuration (3 windows, binary labels)
src/23_final_lgb.py final submission builder
src/26_two_users.py section 7 two-user comparison (written, not yet run)
work/               caches: matches.npy, airings*.npy, ds/*.npz (201 MB)
data/               competition CSVs, 974 MB, gitignored
submissions/        all 9 Kaggle files
```

Always run python with `OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2` — without it,
BLAS oversubscribes on the tiny ALS solves and load average hits 300+ (a 10x slowdown).

## What is left to do

1. Final model on the winning 3-window configuration -> submit (expect ~0.435 public).
2. `src/26_two_users.py` — section 7, two users, every method's Top-5 with titles.
3. Section 8 business analysis. **This is what `/arena` was installed for.**
4. `notebook.ipynb` — optional but worth it for the grader.
5. Report PDF: GitHub link, screenshot of the returned Kaggle score, method comparison,
   validation table, business analysis.
6. Push to https://github.com/tshermatova/sweettv-tv-program-recommender
   (SSH key is passphrase-locked; the user must `ssh-add --apple-use-keychain` once).
7. **Revoke the Kaggle token** at https://www.kaggle.com/settings/api before submitting —
   it was pasted into the session as a screenshot and the session export is submitted.

## Gate 0 discipline

Every number in the report must come from a run in this repository. No invented
experiments, no claimed verification that did not happen. Fabrication is an automatic
fail of the exam.
