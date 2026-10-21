# sweet.tv — TV Program Recommender (final exam quiz)

Tansuluu Shermatova · LLM4Rec, HSE University · instructor: Seungmin Jin

Kaggle community competition
[`sweettv-tv-program-recommender`](https://www.kaggle.com/competitions/sweettv-tv-program-recommender),
metric **MAP@5**, 1 260 test users, five ranked programme ids each.

## What this repository contains

```
src/          implementation, numbered in execution order
submissions/  every file submitted to Kaggle, including the designated final one
README.md     this file
```

The competition data is **not** committed (890 MB, and redistribution is not ours to make).
`src/00_probe.py` onwards expect it unzipped in `data/`.

## Reproducing

```bash
pip install kaggle numpy pandas scipy scikit-learn
kaggle competitions download -c sweettv-tv-program-recommender -p data/ && unzip -d data/ data/*.zip
python src/00_probe.py        # schema checks
python src/01_join.py         # interval join  -> work/matches.npy
python src/03b_airings_raw.py # labelled airing index
python src/14_gbm.py          # method comparison on the hold-out
python src/16_final.py        # final submission file
```

Environment used for every number reported: macOS 13.2 (Darwin 22.3), Python 3.11.4,
numpy 1.24.3, scipy 1.10.1, scikit-learn 1.3.0, 8 cores / 8 GB RAM.
Random seed `20261021` throughout (`lib.SEED`).

## Pipeline in one paragraph

A viewing is matched to a programme when it is on the **same channel** and the view
interval overlaps the airing interval; the overlap must cover at least **20%** of the
programme's own duration to count as watched. Schedule rows with `tv_show_id = 0`
(89.2% of the grid) are unlabelled airtime: they never become items, but they still
count towards a user's channel affinity. Models are fitted on weeks 11-30 only;
weeks 31-42 contribute the broadcast grid and nothing else.
