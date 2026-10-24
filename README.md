# sweet.tv — TV programme recommender

Final exam, GSB. Kaggle community competition
[`sweettv-tv-program-recommender`](https://www.kaggle.com/competitions/sweettv-tv-program-recommender):
for each of 1 260 users, predict the five programmes they will watch most often in
weeks 31-42 of 2020. Metric MAP@5.

## Result

**`submissions/sub25_top5_s0.csv`** — Kaggle ref 57573516,
**public 0.45440 / private 0.44205**.

For scale, the frozen leaderboard's first place (team Magic City) scored public 0.44297
/ private 0.42912.

## The model

`LGBMRanker`, objective lambdarank, over a candidate pool of each user's own watched
programmes plus the 80 most popular — about 139 candidates per user.

* 41 features: 30 describing volume, popularity, airings, channel and time-of-day
  affinity and four collaborative scores (association rules, personalised PageRank,
  content-based similarity, iALS), plus 11 describing the *shape* of a viewing habit
  (`src/features3.py`).
* three training windows, the hold-out's geometry shifted forward 70 days
* 1600 trees, lr 0.015, 31 leaves; rank-average of five seeds
* **label = the organisers' own rule**: each user's five most frequent airings at screen
  time >= 0.8. Aligning the label to the target was worth +0.0079 on the hold-out and
  +0.00706 on Kaggle private — the single change that moved the result.

Build it with:

```
OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
python3 src/36_final_feats3.py --big --seeds 5 --top5 --per-seed --name sub25_top5
```

## Reproducing

**The competition data is not in this repository and must not be**: it is 890 MB, and
the competition rules forbid redistributing it. Download it from Kaggle with your own
account (the rules must be accepted first, or the download returns 403). Everything else
— code, the final submission file, the notebook, the report — is here.

All paths are derived from each file's own location, so the repository runs from any
checkout.

```
pip install lightgbm numpy scipy scikit-learn
kaggle competitions download -c sweettv-tv-program-recommender -p data/ && unzip -d data/ data/*.zip
python3 src/01_join.py          # interval join -> work/matches.npy
python3 src/03b_airings_raw.py  # airing grid
python3 src/30_cache2.py        # feature caches
python3 src/36_final_feats3.py --big --seeds 5 --top5 --per-seed --name sub25_top5
```

Always set `OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2`; without it BLAS oversubscribes
on the small ALS solves and the machine slows by about a factor of ten.

## Where to look

| file | what |
| --- | --- |
| `notebook.ipynb` | end-to-end walkthrough |
| `report/section8.md` | the business and algorithmic analysis |
| `report/section7_two_users.txt` | two users, every method's top five, with titles |
| `BRIEF.md` | every measurement, every direction tried, and why the winners' recipes did not transfer |
| `STATE.md` | working notes, including the mistakes and what they cost |
| `src/lib.py` | data layer, MAP@5, and `target_top5` — the organisers' target rule |
| `src/31_lab.py` | the experiment harness |

## Two things worth reading before trusting any number here

**The target definition was wrong for the first two days.** The hold-out counted every
programme with a complete view as relevant (median 11 per user) instead of the five most
frequent at screen time >= 0.8. It read 0.5586 while Kaggle returned 0.43499. Correcting
it closed the gap to 0.003, and only measurements made afterwards are used in the report.

**One feature block leaked and was discarded.** An expected-frequency feature built
channel preferences from the whole log rather than from history alone, so on the hold-out
it saw the window it was predicting: +0.0272 there, **-0.011** on Kaggle private. Fixed
in `src/features9.py`; the honest version is worth -0.0024 and is not shipped.
