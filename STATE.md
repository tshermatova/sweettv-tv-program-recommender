# Session state — read this first after a restart

Project root: `/Users/tshermatova/Downloads/week7-exam/`
Exam: sweet.tv TV programme recommender, MAP@5, deadline **24 Oct 2026 19:00 MSK**.
Spec lives in the conversation as `exam_quiz_GSB.md`; consultation transcript as a PDF.

## Where things stand

Best Kaggle result — `sub11_feats3_3win.csv`, **ref 57514552**, re-submitted
2026-10-22 07:07 with the description
"Shermatova Tansuluu - FINAL. LambdaRank, 41 features, 3 training windows, 5-seed
rank average". **public 0.44994 / private 0.43415**. It is the most recent, the best,
and the only one carrying the full name the instructor asked for, so the screenshot
for the PDF must show THIS row (ref 57514552), not the earlier 57499460 whose
description was just the file name. 13 submissions made; the cap is 99 per day.

Verified against the frozen leaderboard itself on 2026-10-22 (screenshots in the
session, both tabs): team **Magic City**, 1st place, public **0.44297**, private
**0.42912**, 12 entries. Our 0.44994 / 0.43415 is above first place on BOTH columns.
State it in the report with these caveats: the board is frozen so we never appear on
it; we had late-submission feedback over 11 attempts against the public half; the
private half we saw only twice, so the private margin is not public-score overfitting.
(An earlier note in this file compared our PUBLIC score to that private figure, which
was wrong -- always compare like with like.)

Submission history (public / private):
0.29788/0.29442 popularity -> 0.41510/0.40302 own repeats -> 0.42449/0.40028 LR hybrid
-> 0.42846/0.41388 LR 17 feats -> 0.43135/0.41924 LambdaRank -> 0.44008/0.42787 3 windows
-> **0.44994/0.43415 41 features**.

Hold-out progression (calibrated hold-out, see below):
popularity 0.40040 -> own repeats 0.48054 -> LR hybrid 0.51141 -> LambdaRank 0.53003 ->
3 training windows 0.54464 (0.54336 when re-run, see noise below) -> **41 features 0.55785**.

## Measured noise floor — read before believing any delta

Seven seeds of the identical 3-window / 30-feature configuration:
min 0.53975, max 0.54481, mean 0.54234, **sd 0.00178**. Anything under ~0.005 on the
hold-out is one draw from this distribution, not a result.

Rank-averaging those 7 seeds scored 0.54289 — **+0.0006 over the mean, below the best
single fit**. Seed ensembling buys insurance against a bad draw, not accuracy.

`lambdarank_truncation_level` swept over 5/8/10/15/20/30/50: 0.53901, 0.53824, 0.54468,
0.54119, 0.54245, 0.54336, 0.54318. Non-monotonic, whole range inside the seed spread.
**No effect. Do not revisit.**

**The hold-out overestimates gains by roughly 1.5x.** FEATS3 was +0.0145 on the hold-out
and +0.00986 public / +0.00628 private on Kaggle. Read future hold-out deltas accordingly.

## Second feature block (FEATS4): built, measured, REJECTED

`src/features4.py` added 10 more signals in the direction that had just worked —
watch completeness in detail (frac_last, frac_p75), repeat timing (gap_mean, overdue
= time since last view in units of that pair's own typical gap), current-streak,
and target-window airing shape (next_air_d, air_weeks, air_regular).

Hold-out, 3 seeds each: 30 base 0.54198 +- 0.00110, 41 (+FEATS3) 0.55662 +- 0.00321,
51 (+FEATS3+FEATS4) 0.55933 +- 0.00117, 40 (base+FEATS4 only) 0.54967 +- 0.00085.
So FEATS4 looked worth +0.0027 over FEATS3.

**It did not transfer.** `sub12_feats4_3win.csv` (ref 57514526) scored public 0.44472
vs sub11's 0.44994 — **worse by 0.0052** — with private identical to 5 decimal places
(0.43420 vs 0.43415). Do not revisit: this is the third time a sub-0.003 hold-out gain
evaporated, and the first time it did so with three seeds behind it. Treat 3 seeds as
the minimum, and treat anything under ~0.005 on the hold-out as unproven regardless of
how many seeds support it.

Pruning the FEATS3 columns whose single-seed leave-one-out delta was positive
(streak, frac_max, own7, last_week) changed nothing: 41 full 0.55662 vs 41 pruned
0.55701; 51 full 0.55933 vs 51 pruned 0.55869. The single-seed ablation that said
`streak` was hurting by +0.0033 was noise.

Window count on 41 features, ONE seed each (so not reliable): 1 win 0.54349,
2 0.55025, 3 0.55785, 4 0.55770, 5 0.56039, 6 0.55938. Rises to three, then flat.
Five windows is nominally best by 1.4 sigma, i.e. unproven.

## The 11 features that produced the gain (`src/features3.py`)

Habit shape rather than habit size: own7, own14, frac_mean, frac_max, streak (longest
run of consecutive calendar weeks), last_week, slot_conc, n_slots, slot_aff (user's
weekly dow x hour profile against the programme's airings in the target window),
ch_air_aff (user's channel shares against the channels the programme airs on), share14.
In the final fit `ch_air_aff` is the 2nd most used feature of 41, `frac_mean` 5th,
`slot_aff` 10th. Which of them carry the gain is NOT yet measured — the leave-one-out
ablation (`src/31_lab.py ablate`) has not been run.

## A leak in the hold-out, found 2026-10-22

`work/ch_views.pkl` (src/01_join.py line 48) sums seconds per (user, channel) over the
WHOLE view log, weeks 11-30. In the FINAL regime that is honest: the test window is
weeks 31-42 and the log never reaches it. In the HOLD-OUT regime it is not: history is
cut at B_AB but the channel feature still sees the target window. 2,126,441 of 3,219,961
(user, channel, day) rows lie before B_AB, so **34% of the channel signal on validation
came from the future**.

`src/32_chviews.py` rebuilds per-day totals so affinity can be cut at any date;
`src/33_extra.py ch_fix` recomputes the two affected columns. Honest hold-out:
30 features 0.51645, 41 features 0.53313 — the FEATS3 gain is +0.0167 there, i.e. it
does not depend on the leak. Use the honest regime to SELECT between configurations;
ship the ordinary one, which is legitimate at final-fit time.

## Decisions already settled (do not re-litigate)

| question | answer | evidence |
| --- | --- | --- |
| join rule | same channel + interval overlap >= 20% of programme duration | `src/02_explore.py` distributions |
| `tv_show_id = 0` | never an item; still counts for channel affinity | 89.2% of the grid |
| hold-out | features < 2020-05-18, target >= that date, target = `frac >= 1.0` | `src/08_calibrate2.py`: this reproduces Kaggle's ordering of 5 probe submissions exactly |
| repeats allowed? | yes — repeats are the single strongest signal | probe 0.41510 vs popularity 0.29788 |
| cold start | there is none: all 1 260 test users appear in the log | `src/00_probe.py` |
| best model | LGBMRanker, lambdarank, **41 features**, 3 training windows, binary labels, 5-seed rank average | `src/36_final_feats3.py` -> 0.44994 public |

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
src/features3.py    the 11 added features (FEATS3) — the 2026-10-22 gain
src/collab2.py      extended collaborative block (tested, rejected)
src/01_join.py      interval join -> work/matches.npy
src/30_cache2.py    disk cache of FEATS2 datasets, hold-out AND final geometry
src/31_lab.py       experiment harness: baseline/trunc/feats3/chfix/seeds/windows3/ablate
src/32_chviews.py   per-day channel totals, so affinity can be cut at any date
src/33_extra.py     caches FEATS3 columns and the honest ch_aff/ch_rank replacements
src/36_final_feats3.py  CURRENT final model + submission builder
src/26_two_users.py section 7 two-user comparison (written, not yet run; it still
                    shows only the four base methods — the final ranker's Top-5 is
                    missing and `mdl=... if False else None` is dead code)
work/ds2/           FEATS2 caches, hold-out + final windows
work/               caches: matches.npy, airings*.npy, ch_day.npz, ds/*.npz
data/               competition CSVs, 974 MB, gitignored
submissions/        all 11 Kaggle files
```

Always run python with `OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2` — without it,
BLAS oversubscribes on the tiny ALS solves and load average hits 300+ (a 10x slowdown).

## What is left to do

1. ~~Final model~~ DONE: `src/36_final_feats3.py`, 0.44994 public / 0.43415 private.
   **The instructor grades the MOST RECENT submission, not the best one**
   (consultation 09:01-09:29: "I'm considering your recent submission is your final
   submission"). `sub11_feats3_3win.csv` is currently both. If any later experiment
   scores worse, re-submit `sub11_feats3_3win.csv` last so it is the most recent again,
   and take the screenshot only after that.
   Not yet run after the FEATS3 result: `31_lab.py seeds3 windows3 ablate`, and the
   honest-regime (chfix) re-check of whichever window geometry wins.
2. `src/26_two_users.py` — section 7, two users, every method's Top-5 with titles.
3. Section 8 business analysis. **This is what `/arena` was installed for.**
4. `notebook.ipynb` — optional but worth it for the grader.
5. Report PDF: GitHub link, screenshot of the returned Kaggle score, method comparison,
   validation table, business analysis.
6. ~~Push to GitHub~~ DONE: https://github.com/tshermatova/sweettv-tv-program-recommender
   (public; SSH works now — the key was added to the GitHub account).
7. **Revoke the Kaggle token** at https://www.kaggle.com/settings/api before submitting —
   it was pasted into the session as a screenshot and the session export is submitted.

## Gate 0 discipline

Every number in the report must come from a run in this repository. No invented
experiments, no claimed verification that did not happen. Fabrication is an automatic
fail of the exam.


## Model plan for the `/arena` run (agreed with the user before the restart)

The skill launches every sub-agent as `subagent_type: general-purpose` and sets no
model, so without an explicit choice they all inherit the parent (Opus 5) — 91 Opus
calls for one report section. Use `--quick` (16 agents, 91 calls) and pass `model`
per phase on each Agent call:

| phase | calls | model | why |
| --- | ---: | --- | --- |
| spawn (16 solutions) | 16 | sonnet | writing argued prose from numbers already supplied |
| attack / defend | 60 | sonnet | finding holes in a peer's text and answering them |
| judge | 15 | opus | the judge eliminates competitors; a bad call corrupts the bracket |

Not 100 agents: 595 calls for one section, and strategy-card diversity saturates well
before that. Not Haiku anywhere: the rubric has five weighted criteria with score
anchors plus a correctness check, and a weak judge defaults everyone to 7.

**What the arena must be given, and what it must not be believed about.**
Sub-agents cannot see the conversation or the data — they know only `.arena/task.md`.
Write every verified number into that brief: final 0.44008 public / 0.42787 private,
hold-out 0.54464, oracle-in-pool 0.85963, popularity baseline 0.40040 hold-out /
0.29788 Kaggle, 10.8% labelled airtime, median 67 relevant programmes per user in the
hold-out window, and the full rejected-hypotheses table above. Any number the winning
solution adds beyond that brief is invented and must not reach the PDF — check each
claim against the logs in `work/` before using it. Gate 0 is an automatic fail.
