# sweet.tv TV programme recommender — consolidated brief

Self-contained as of 2026-10-23 23:30. Every number here comes from a run in this
repository or from a Kaggle submission; nothing is estimated or recalled.

## The task

Kaggle competition `sweettv-tv-program-recommender` (closed 2020-11-21, late submissions
still scored). For each of **1 260 test users**, predict **5 programme ids** they will
watch during weeks 31-42. Metric **MAP@5**, denominator `min(5, |relevant|)`, verified
against the course slides' worked examples (relevant {A,B,C}: `A,B,D,C,E` → 0.9167,
`D,A,E,B,C` → 0.5333).

**Grading**: the instructor confirmed on 2026-10-23 that he takes the **highest private
score** over all submissions — not the public score the written spec names, and not the
most recent submission.

## CORRECTION, 2026-10-24: the target was wrong for the first two days

The competition data page defines the answer file:

    all entries with tv_show_id = 0 were removed
    for each user and each TV show user screen time (time user watched the show divided
    by total duration of the show) was calculated. Only TV shows with high screen time
    (>= 0.8) were taken into account.
    for each user high-screen time TV shows frequency was calculated
    for each user strictly 5 most frequent TV shows were selected DESC - order matters!
    Private/public test data is split BY USERS (50/50) - not by time.

Everything measured before 2026-10-24 00:50 used `frac >= 1.00` and treated **every**
qualifying programme as relevant (median 11 per user). The truth is **exactly five**, the
five most frequent, at threshold **0.8**. 84.7% of the true answers were inside our set,
together with roughly six extra per user that cannot score.

| | hold-out | Kaggle private | gap |
| --- | ---: | ---: | ---: |
| old target | 0.5586 | 0.43499 | 0.12 |
| **true target (`lib.target_top5`)** | **0.4316** | 0.43499 | **0.003** |

The hold-out was never broken; the scale was. It now predicts private to 0.003.

Training labels aligned to the same rule, 3 seeds, 41 features, 1600 trees:

| label | MAP@5 (true target) |
| --- | ---: |
| every programme, frac >= 1.0 (what sub13 ships) | 0.43156 +- 0.00169 |
| every programme, frac >= 0.8 | 0.43101 +- 0.00037 |
| **top-5 by frequency, frac >= 0.8** | **0.43949 +- 0.00095** |

The threshold was never the issue. The **top-5 restriction** is worth +0.0079, eight times
the seed spread.

**The pool, re-measured under the true target:** recall 0.8181, ceiling 0.8181, 1.9% of
users have no correct answer in the pool, and **10.79% of candidates never air in weeks
31-42**, so their frequency is necessarily zero. Note that coverage now costs
1 − 0.8181 = **0.182**, against 0.104 under the old target: candidate generation matters
nearly twice as much as it appeared to.

**Which earlier rejections survive.** Those confirmed by a Kaggle submission do, because
the private score is the true target: FEATS4, FEATS5, FEATS6, FEATS5+6, the 0.50 label,
the switching rule, both rank fusions, pool 160. Those rejected on the hold-out alone were
measured on the wrong scale and are uncertain again: truncation_level, num_leaves,
rank_xendcg, lambdarank_norm, sigmoid, DART, FEAT_THR, the absolute-seconds rule, window
weights, window count, negative downsampling, the binary classifier, the alignment
variants, "N complete views", FEATS7. One is void rather than uncertain: group weight =
1/min(5, positives), since |R| = 5 for 90.9% of users leaves nothing to normalise.

**How it was found:** searching for solutions to THIS competition, which had not been done
in two days of studying H&M and OTTO write-ups.
[Marrrik/sweettv-recsys](https://github.com/Marrrik/sweettv-recsys) reports public 0.45497
/ private 0.45015 and states the 0.8 rule. Its approach also differs from ours in two ways
worth copying: candidates are restricted to programmes **airing in the target window**, and
it carries an explicit **habit x schedule** source — which is an estimate of the expected
number of qualifying views, i.e. of the target quantity itself. `src/features9.py`
implements that as five columns; a tree splits on axes and does not multiply them, so
supplying the factors separately was never the same as supplying the product.

## Data

| file | rows | role |
| --- | ---: | --- |
| `dataset11-30.csv` | 5 821 987 | view log, weeks 11-30 only |
| `export_arh_11-20`, `21-30` | 2 148 629 | schedule, training weeks |
| `export_arh_31-42` | 1 295 378 | schedule, test weeks (grid only, no viewing) |

View log columns: `user_id, vsetv_id (channel), start_time, stop_time, duraton`.
Schedule: channel, title, start, duration, `tv_show_id`, category, 3 genres, year,
director, actors.

Facts that shape everything:

- **Only 10.8% of schedule rows carry a `tv_show_id`.** The rest is unlabelled airtime
  and cannot become an item.
- The view log stops at week 30, so **no test-window viewing exists anywhere** — there
  is nothing to leak.
- 4 363 distinct labelled programmes appear in the join; **88% are category "Фильмы"**.
  Metadata coverage: category 100%, genre-1 99.3%, director 97.7%, actors 97.9%.
- Episodes do **not** split across ids: only 78 titles span more than one `tv_show_id`
  (160 of 4 363 ids), and those are same-named films, not serial episodes.
- Median test-user history is **330 events**. Only 2 users of 1 260 have zero history,
  8 have fewer than 10, 32 fewer than 50. **There is no cold-start problem.**
- Test users are **not** a random sample: median 196 history events before B_AB against
  125 for the other 3 573 active users; 77 vs 56 distinct programmes.

## Pipeline

1. **Join** view intervals to the schedule: same channel, interval overlap, keep events
   where `frac = overlap / programme_duration >= 0.20`. `frac` cannot exceed 1.0 — the
   overlap is capped by the programme's own duration.
2. **Candidates** per user: own watched items + the 80 globally most popular → ~138/user.
3. **Features**: 41 columns (see below).
4. **Model**: `LGBMRanker`, objective lambdarank, 1600 trees, lr 0.015, 31 leaves,
   3 training windows, binary labels `frac >= 1.0`, rank-average of 5 seeds.

Training windows for the shipped model are the hold-out's winning geometry shifted
forward 70 days: history < 06-08 / 06-22 / 07-06, label horizons 5, 5, 3 weeks.

## Best result

`sub13_feats3_1600.csv`, ref 57522216: **public 0.45080 / private 0.43499**.
Frozen leaderboard first place (Magic City): public 0.44297 / private 0.42912.

Submission history, private: 0.29442 popularity → 0.40302 own repeats → 0.41388 LR
→ 0.41924 LambdaRank → 0.42787 three windows → **0.43499 (1600 trees)**.

## Measured diagnostics

| quantity | value | meaning |
| --- | ---: | --- |
| candidate recall (all relevant) | 0.4161 | 58% of what users watch never enters the pool |
| MAP@5 ceiling given the pool | 0.8959 | perfect ranking of current candidates |
| model on hold-out A | 0.5586 | **the gap to the ceiling is 0.34 — ranking is the bottleneck** |
| users with no relevant item in pool | 2.0% | |
| users with <5 relevant in pool, \|R\|>=5 | 14.1% | |
| seed-to-seed sd (identical config) | 0.00178 | anything under ~0.005 on one hold-out is noise |
| per-user AP sd | 0.3024 | AP is near-binary: 17.1% score 0, 6.4% score 1 |
| implied public/private sampling spread | 0.0170 | with ~630 users per half |
| observed public − private | +0.0035 to +0.0242 | **positive for all 11 models including the leader's — the public half is systematically easier** |

## The two hold-outs

- **A**: history < 2020-05-18, target 05-18..07-27 (10 weeks), 1 209/1 260 users have
  truth, median 11 relevant.
- **B**: history < 2020-04-13, target 04-13..05-18 (5 weeks), 1 178 users, median 8.
  Its training windows carry only 2-3 weeks of history.

**Known defect**: `work/ch_views.pkl` sums channel-seconds over the whole log, so on
hold-out A the channel features see 34% of their signal from the target window. Honest
recomputation drops A from 0.5634 to 0.5165 but preserves every ordering tested.

**Known limitation**: the hold-outs are reliable for structural questions and unreliable
below ~0.003. Label threshold 0.50 → A −0.016, Kaggle private −0.011 (agree). FEATS5
→ A +0.005, Kaggle private −0.001 (disagree).

## Features (41 = 30 base + 11)

Base 30: own-history counts/days/weeks/share/rank, recency, tenure, 28-day trend,
regularity, catch rate, popularity (users/events/recent/rank), airings in the target
window (total, days, channels, first 2 weeks, last 4 weeks, tail), channel affinity and
rank, hour and day-of-week affinity, and four collaborative scores (association rules,
personalised PageRank, content similarity, iALS).

Added 11 (`features3.py`, the one block that transferred): own7, own14, frac_mean,
frac_max, streak, last_week, slot_conc, n_slots, slot_aff, ch_air_aff, share14.

Leave-one-out says two of them carry the gain: **ch_air_aff** (−0.0058) and
**frac_mean** (−0.0043). The others are inside noise. Feature importance agrees:
`ch_aff` and `ch_air_aff` are first and second of 41, `frac_mean` third.

## Everything tried, with numbers

**Accepted (2):**

| change | hold-out A | hold-out B | Kaggle private |
| --- | ---: | ---: | ---: |
| FEATS3 habit-shape features | +0.0146 | +0.0039 | **+0.0063** |
| 1600 trees / lr 0.015 | +0.0019 | +0.0024 | **+0.0008** |

**Rejected on Kaggle after winning on validation (4):**

| change | hold-out A | Kaggle private |
| --- | ---: | ---: |
| FEATS4 repeat timing (51 feats) | +0.0027 | 0.0000 |
| FEATS5 whole-log time profiles (50) | +0.0052 | −0.0010 |
| FEATS6 content affinity (50) | +0.0029 | −0.0004 |
| FEATS5+FEATS6 (59) | +0.0059 | −0.0003 |

**Rejected on validation (14):**

- `lambdarank_truncation_level` 5/8/10/15/20/30/50 — non-monotonic inside the seed spread
- `num_leaves` 15 and 127 — both worse (127: A −0.0035, B −0.0085)
- `rank_xendcg` — A −0.0052, B −0.0020
- `lambdarank_norm=false` — A −0.0071, B −0.0032
- `sigmoid` 0.5 and 2.0 — both worse, inverted U at the default
- DART, drop_rate 0.1 — A −0.0065, at **11.6x** the cost of GBDT
- training-label threshold 0.8 / 0.5 / 0.2 — monotone down, −0.005 / −0.016 / −0.035
- `FEAT_THR` 0.10 and 0.35 — inverted U, 0.20 is the peak
- absolute-seconds watched rule, 300 s and 600 s — A −0.009 and −0.015
- window weights 1:1:2 and 1:0.5:2 — the second reads +0.0019 on A and −0.0021 on B
- window count 1→7 — rises to 3, then flat
- negative downsampling 40 and 20 per user — monotone down
- binary classifier instead of ranker — A −0.0044
- upweighting (×2, ×3) or dropping low-activity users — A flat or worse, B monotone down
- group weight = 1/min(5, positives) — A −0.0074, B −0.0047
- recent-window collaborative block (FEATS7) — A −0.0040
- extended 9-signal collaborative block — −0.001
- graded labels, time decay, within-user normalisation, pointwise GBM — all at or below baseline

**Rejected by direct measurement, no model fitted (2):**

- **Co-visitation candidates.** Adding the top-20 co-viewed items per history item raises
  recall 0.4161 → 0.4250 (count/confidence) or 0.4445 (lift, pool 140 → 551) and the
  MAP@5 ceiling only 0.8959 → 0.9009. The ceiling was never the constraint.
- **Switching rule.** Replacing the ranker with popularity for thin-history users:
  threshold 50 events (32 users) changes private by −0.00009; threshold 100 (111 users)
  costs −0.0055. The ranker beats popularity from ~50 events upward.

**Rejected by submission (3):**

- Rank fusion of six LightGBM variants (RRF, k=0 and k=60): private 0.43419 / 0.43422.
  The six models share **5.86 of 5 items** on average — 39.8% of users get an identical
  set from all six, 72.6% of items appear in all six lists. There is nothing to fuse.
- LightGBM + CatBoost rank fusion: hold-out +0.0015 / +0.0030, private **0.43405**.
  This was the most genuinely different model (4.72/5 overlap with sub13) and still lost.
- Candidate pool 80 → 160: A −0.0008, B +0.0014.

**Running at the time of writing:** label requiring N complete views (N=1/2/3), and
Word2Vec proximity features (41 → 46 columns).

**Untried:** cascade re-ranking (a second model over the first model's top 20).

## What the research of winning solutions said, and why it did not transfer

[H&M 1st place](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/writeups/senkin13-30crmnsia-1st-place-solution):
100 candidates, 6 training weeks, LightGBM **classifier**, ensemble of 5 LightGBM +
7 CatBoost. States that user-item interaction features mattered most and image/text
features did not — which our FEATS6 result independently reproduces.

[OTTO, nicolaivicol](https://github.com/nicolaivicol/otto-recommender): seven candidate
sources (co-visitation ×5, Word2Vec ×2, cluster popularity) reaching 172 candidates and
recall@20 = 0.64; ~100 features; LightGBM lambdarank, depth 4, 150 trees, lr 0.25;
negatives downsampled 1:40 capped at 100 per session.

[OTTO, cdeotte](https://github.com/cdeotte/Kaggle-OTTO-Comp): co-visitation on GPU,
XGBoost picks 20 of 100, trained on 3.5 weeks and inferred on 4.5.

**Why these recipes failed here.** Their bottleneck is retrieval: OTTO's recall@20 is
0.64 and the ceiling binds. Ours is ranking: the ceiling is 0.896 and the model sits at
0.559. Their datasets are 40M rows, so downsampling is a memory necessity; ours is 1.4M
and every negative is affordable. Their sessions are short and anonymous, so cold start
dominates; our median user has 330 events and cold start does not exist. Six of the
rejected directions above are theirs, and each failed for one of those three reasons.

## What is not explained

A classmate reportedly has private **0.437**, another **0.45**. Our 0.43499 is 0.002 and
0.015 away. The gap is not explained by anything measured here:

- the target definition is right — loosening it costs 0.011 on private, confirmed by
  submission, not only by validation;
- candidate generation is not the constraint — measured, ceiling 0.896;
- the algorithm is exhausted — seven parameters swept, one win worth 0.0008;
- ensembling cannot help — our models agree on 5.86 of 5 items;
- the public/private gap is systematic for everyone, including the leaderboard winner,
  so it is not an artefact of our tuning.

A private of 0.45 would be 0.021 above the competition's own winner. Either there is an
approach not found in the survey above, or the figure is a public score read as private.
**The single most useful thing to learn from whoever holds it: how many features, which
model, and which column they are quoting.**

## Environment

anaconda python 3.11 (`/Users/tshermatova/anaconda3/bin/python3`), numpy 1.24.3,
lightgbm 4.7.0, catboost 1.2.10, gensim 4.3.0. MacBook Air M1, 8 GB, **fanless**.
Always run with `OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2`; without it BLAS
oversubscribes on the small ALS solves. One heavy LightGBM job at a time plus one light
job is what this machine supports.

=== SECTION 7 OUTPUT (the two users) ===
==============================================================================================
SECTION 7 — two users, every method's Top-5 (scored against the organisers' top-5 target)
==============================================================================================

Why these two. There is no cold-start case to show: all 1 260 test users appear in
the view log, so no user has an empty history. The informative contrast is therefore
volume of evidence. These are the extremes of the eligible range, which makes them
the hardest and easiest cases the ranker faces. Median history is 212 events;
1118 of 1 260 test users have both a history and something to predict.

  HEAVY  user_id=13571199046623737884   history  3733 events over  445 distinct programmes;  relevant in the hold-out window: 5
  LIGHT  user_id=14401191285650834662   history     1 events over    1 distinct programmes;  relevant in the hold-out window: 5

fitting the final configuration on the hold-out geometry [2s]
done [156s]

==============================================================================================
HEAVY  user_id=13571199046623737884   candidate pool 450   actually watched in the hold-out window: 5 programmes
==============================================================================================

  what this user watched most in the training window:
      538x  id=2400480   ТСН. Неделя                                          [Инфо]
      244x  id=20088     Завтрак с 1+1'. Информационно-развлекательная програ [Развлечения]
      222x  id=240081    Факты. День                                          [Инфо]
      150x  id=2400508   Сегодня. Итоги с Олегом Панютой                      [Инфо]
      124x  id=12001682  М/с 'Губка Боб Квадратные Штаны                      [Детям]
      123x  id=2400467   Викна-Новости                                        [Инфо]

  popularity (baseline)   (1/5 hit)
     1. [miss] id=2400480   ТСН. Неделя                                          [Инфо]
     2. [HIT ] id=240081    Факты. День                                          [Инфо]
     3. [miss] id=2400467   Викна-Новости                                        [Инфо]
     4. [miss] id=2400508   Сегодня. Итоги с Олегом Панютой                      [Инфо]
     5. [miss] id=6500479   Мир наизнанку. Бразилия', 36 эп                      [Познавательное]

  association rules   (1/5 hit)
     1. [miss] id=2400480   ТСН. Неделя                                          [Инфо]
     2. [HIT ] id=240081    Факты. День                                          [Инфо]
     3. [miss] id=2400467   Викна-Новости                                        [Инфо]
     4. [miss] id=2400508   Сегодня. Итоги с Олегом Панютой                      [Инфо]
     5. [miss] id=6500479   Мир наизнанку. Бразилия', 36 эп                      [Познавательное]

  PageRank (PPR)   (1/5 hit)
     1. [miss] id=2400480   ТСН. Неделя                                          [Инфо]
     2. [HIT ] id=240081    Факты. День                                          [Инфо]
     3. [miss] id=2400467   Викна-Новости                                        [Инфо]
     4. [miss] id=2400508   Сегодня. Итоги с Олегом Панютой                      [Инфо]
     5. [miss] id=6500479   Мир наизнанку. Бразилия', 36 эп                      [Познавательное]

  content-based (CBF)   (1/5 hit)
     1. [miss] id=2400480   ТСН. Неделя                                          [Инфо]
     2. [miss] id=2500261   Подробности недели                                   [Инфо]
     3. [miss] id=5900347   Деньги                                               [Инфо]
     4. [HIT ] id=240081    Факты. День                                          [Инфо]
     5. [miss] id=2400467   Викна-Новости                                        [Инфо]

  matrix factorization (iALS)   (0/5 hit)
     1. [miss] id=10003014  Т/с 'Отдел 44', 71 с                                 [Сериалы]
     2. [miss] id=300358    Скетч-шоу 'На троих', 7 сезон                        [Развлечения]
     3. [miss] id=1000589   Т/с 'Отчаянные домохозяйки', 4 сезон, 17 с           [Сериалы]
     4. [miss] id=2500430   Затерянный мир                                       [Познавательное]
     5. [miss] id=400426    Меняю жену                                           [Развлечения]

  FINAL — LambdaRank, 41 features   (1/5 hit)
     1. [HIT ] id=240081    Факты. День                                          [Инфо]
     2. [miss] id=2400480   ТСН. Неделя                                          [Инфо]
     3. [miss] id=2400508   Сегодня. Итоги с Олегом Панютой                      [Инфо]
     4. [miss] id=12001682  М/с 'Губка Боб Квадратные Штаны                      [Детям]
     5. [miss] id=6500479   Мир наизнанку. Бразилия', 36 эп                      [Познавательное]

==============================================================================================
LIGHT  user_id=14401191285650834662   candidate pool 81   actually watched in the hold-out window: 5 programmes
==============================================================================================

  what this user watched most in the training window:
        1x  id=90071788  Х/ф 'Призрак                                         [Фильмы]

  popularity (baseline)   (3/5 hit)
     1. [miss] id=2400480   ТСН. Неделя                                          [Инфо]
     2. [HIT ] id=240081    Факты. День                                          [Инфо]
     3. [HIT ] id=2400467   Викна-Новости                                        [Инфо]
     4. [HIT ] id=2400508   Сегодня. Итоги с Олегом Панютой                      [Инфо]
     5. [miss] id=6500479   Мир наизнанку. Бразилия', 36 эп                      [Познавательное]

  association rules   (3/5 hit)
     1. [miss] id=2400480   ТСН. Неделя                                          [Инфо]
     2. [HIT ] id=240081    Факты. День                                          [Инфо]
     3. [HIT ] id=2400467   Викна-Новости                                        [Инфо]
     4. [HIT ] id=2400508   Сегодня. Итоги с Олегом Панютой                      [Инфо]
     5. [miss] id=6500479   Мир наизнанку. Бразилия', 36 эп                      [Познавательное]

  PageRank (PPR)   (3/5 hit)
     1. [miss] id=90071788  Х/ф 'Призрак                                         [Фильмы]
     2. [miss] id=2400480   ТСН. Неделя                                          [Инфо]
     3. [HIT ] id=240081    Факты. День                                          [Инфо]
     4. [HIT ] id=2400467   Викна-Новости                                        [Инфо]
     5. [HIT ] id=2400508   Сегодня. Итоги с Олегом Панютой                      [Инфо]

  content-based (CBF)   (0/5 hit)
     1. [miss] id=90071788  Х/ф 'Призрак                                         [Фильмы]
     2. [miss] id=10003159  Т/с 'Ничто не случается дважды', 1 и 2 с             [Сериалы]
     3. [miss] id=10003178  Премьера. Т/с 'Агенты справедливости', 7 сезон, 9 с  [Сериалы]
     4. [miss] id=10001657  Т/с 'Касл', 5 сезон, 3 с. 'Секрет в моих надёжных ру [Сериалы]
     5. [miss] id=10003102  Т/с 'Опер по вызову', 4 сезон, 1 и 2 с               [Сериалы]

  matrix factorization (iALS)   (0/5 hit)
     1. [miss] id=90077687  Х/ф 'Праведник (Великий уравнитель)                  [Фильмы]
     2. [miss] id=90077217  Х/ф 'Геракл: Начало легенды                          [Фильмы]
     3. [miss] id=1000589   Т/с 'Отчаянные домохозяйки', 4 сезон, 17 с           [Сериалы]
     4. [miss] id=90076295  Х/ф 'Призрачный патруль                              [Фильмы]
     5. [miss] id=90064639  Х/ф 'Пророк                                          [Фильмы]

  FINAL — LambdaRank, 41 features   (3/5 hit)
     1. [HIT ] id=240081    Факты. День                                          [Инфо]
     2. [HIT ] id=2400508   Сегодня. Итоги с Олегом Панютой                      [Инфо]
     3. [miss] id=2400480   ТСН. Неделя                                          [Инфо]
     4. [miss] id=300358    Скетч-шоу 'На троих', 7 сезон                        [Развлечения]
     5. [HIT ] id=2400467   Викна-Новости                                        [Инфо]

written to /Users/tshermatova/Downloads/week7-exam/work/two_users.txt  [156s]


=== NUMBERS MEASURED AFTER THE TARGET WAS CORRECTED (the only ones that count) ===

All on the calibrated hold-out, true target (each user's five most frequent airings at
screen time >= 0.8), 3 seeds, 41 features, 1600 trees unless stated.

  popularity baseline                         0.27387
  own repeats weighted by airings             0.37324
  SHIPPED: 41 features, top-5 label           0.43949 +- 0.00095
  + expected-frequency features (honest)      0.43711   (-0.0024)
  + graded label, rank 5..1                   0.46457 vs 0.46667 on the leaky 46-set
  + graded label, airing count                0.46497 vs 0.46667 on the leaky 46-set
  + broadcast-history features                0.43819   (-0.0013)
  + candidates restricted to airing items     0.46524 vs 0.46667 on the leaky 46-set
  + pool expanded by top-3 channels           0.43907   (-0.0004), ceiling 0.7974 -> 0.8325
  2400 trees / lr 0.010                       0.44026
  800 trees / lr 0.030                        0.44060
  num_leaves 63                               0.43928
  num_leaves 15                               0.43826
  cascade re-ranking the top 20               0.43315   (-0.0064)

  Kaggle, designated submission sub25_top5_s0: public 0.45440 / private 0.44205
  Frozen leaderboard first place (Magic City): public 0.44297 / private 0.42912
  The aligned label was worth +0.0079 hold-out and +0.00706 Kaggle private.
  The leaky expected-frequency feature read +0.0272 hold-out and -0.011 Kaggle private.

  Candidate pool: 139 per user. Recall of the true answers 0.8181, MAP@5 ceiling 0.8181.
  17.1% of correct answers are not candidates; 175 of those are premieres.
  Top-20 of association rules adds 0 new candidates, PageRank 0, CBF 4, iALS ~0.
  Seed-to-seed sd 0.00178. Public is systematically above private for every model.

=== WHAT SECTION 8 MUST ANSWER (from the exam specification, verbatim) ===

"Which recommendation strategy would you select for this platform, and why?"

Using the same two users, compare each method's five ids against what those users
actually watched, and discuss:
 - whether the lists reflect familiar preferences or offer relevant new programmes;
 - how and why the methods differ - what association rules capture that matrix
   factorization misses, what content attributes contribute where co-view signal is
   thin, how PageRank's graph diffusion reshapes the ranking, and where the popularity
   baseline already does well;
 - why your final strategy is preferable, considering validation performance,
   computational cost at this scale, and what the lists imply for the user experience;
 - limitations and failure cases your results actually support, beginning with the fact
   that only 10.8% of airtime carries an id.

Two cautions. An unfamiliar programme is NOT automatically a useful discovery - it may
simply be a poor recommendation, and nothing here reveals whether a user enjoyed it. And
absence from the log does not establish that a user has never seen a programme: the log
records this platform's views only. Qualify "never", "always" and "better" to the
evidence and scope examined.

"No live A/B test, online experiment or revenue claim is required - and none may be
claimed." Every assertion must follow from the supplied data and the implementation.

=== YOUR TASK ===

Write Section 8 as it will appear in the submitted PDF. English. Aim for 900-1400 words.
Write for the grader: a recommender-systems lecturer who has the specification in front
of them and will check claims against the repository.

Hard rules:
 - Use ONLY numbers that appear above. Any figure not in this brief is fabricated and
   will be removed; a fabricated number is an automatic fail of the exam.
 - Do not claim any online experiment, A/B test or revenue effect.
 - Do not pad. Every paragraph must make a claim the data supports.
 - Say what is uncomfortable where the data says it, rather than smoothing it.
