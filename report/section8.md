# Section 8 — Which recommendation strategy to select, and why

## The strategy

**A LightGBM LambdaRank re-ranker over a candidate pool of the user's own watched
programmes plus the eighty most popular ones, trained on three overlapping time windows
with a label that copies the organisers' own definition of a correct answer.**

On the calibrated hold-out it scores MAP@5 = 0.43949 against 0.27387 for a popularity
baseline. On Kaggle it returns public 0.45440 / private 0.44205, above the frozen
leaderboard's first place (public 0.44297 / private 0.42912).

Everything below is measured on this data. No online experiment was run and none is
claimed.

## What the two users show

Two users, both with a full five-programme target, chosen as the extremes of evidence:
one with 3 733 viewing events over 445 distinct programmes, one with **a single event**.

| method | heavy user (3 733 events) | light user (1 event) |
| --- | ---: | ---: |
| popularity | 1/5 | 3/5 |
| association rules | 1/5 | 3/5 |
| personalised PageRank | 1/5 | 3/5 |
| content-based | 1/5 | 0/5 |
| matrix factorisation (iALS) | 0/5 | 0/5 |
| **final ranker** | **1/5** | **3/5** |

The first thing to say about this table is that it inverts the expectation. **The user
with one recorded view is predicted three times better than the user with 3 733.** That
is not a defect of the model; it follows from what the task actually asks. The answer is
each user's *five most frequent* programmes at screen time ≥ 0.8. For someone with no
history, those five are simply the mass-audience news bulletins, and popularity names
them. For someone who watched 445 distinct programmes, the five that happen to top their
frequency list are a fine-grained ordering problem inside a large personal repertoire,
and every method we have gets one of them.

The practical reading is uncomfortable but useful: **on this platform, personalisation
earns the least precisely where there is most to personalise from.**

## How the methods differ, and where they do not

**Popularity, association rules and personalised PageRank produced byte-identical
lists for the heavy user** — the same five ids in the same order. This is not a bug. Of
4 363 labelled programmes, 88% are films, yet the heavy viewers of this service watch
recurring news bulletins, which dominate the co-view matrix. Two programmes are "watched
together" mostly because almost everyone watches both, so the association matrix is a
popularity matrix with extra steps, and PageRank's diffusion over that graph returns to
the same high-degree nodes. We measured this directly: adding the top-20 items of
association rules or PageRank to the candidate pool adds **zero new candidates**, and all
four collaborative methods together add four.

**Content-based filtering is the only method that proposes genuinely different things,
and it scores zero.** For the light user it recommended films and series resembling the
one film they had watched. Those are defensible recommendations by any editorial
standard — and they earn nothing, because the metric counts *frequency of complete
viewings*, and nobody watches a particular film five times. Content similarity answers
"what else is like this", and the target asks "what will recur".

**Matrix factorisation is the method that disagrees most, and it is wrong both times.**
It is the only one that put series and entertainment ahead of news for the heavy user,
and films for the light one. iALS learns latent taste from a user-item matrix whose mass
is in habitual viewing; asked to extrapolate twelve weeks forward, it extrapolates taste
when the correct answer is habit.

**Where popularity already does well is the sparse tail**, and it does well there not by
understanding the user but because the target for a sparse user *is* the population mode.

**What the final ranker adds** is not a different set of candidates — it is ordering.
For the heavy user it finds the same single correct programme as popularity but places it
**first instead of second**, which is what MAP@5 pays for. For the light user it finds
three, with two of them in the top two positions. Across all users this ordering effect is
worth 0.43949 against 0.27387: the ranker is not discovering different content, it is
arranging the same content better.

## Why this strategy and not another

**Validation.** The hold-out splits history at 2020-05-18 and predicts the ten weeks
after it, scoring against the organisers' rule exactly. Of the eleven alternatives
tested after the target definition was corrected, ten were neutral or worse: expected-
frequency features (−0.0024), graded labels (−0.0021 and −0.0017), broadcast-history
features (−0.0013), restricting candidates to programmes that actually air (−0.0014),
expanding the pool with every programme on the user's top three channels (−0.0004),
four capacity settings (within ±0.0011), and a two-stage cascade re-ranking the top 20
(−0.0064). Exactly one change helped: aligning the training label to the organisers'
definition, worth +0.0079 on the hold-out and +0.00706 on Kaggle private.

**Cost.** One fit is about six minutes on an 8 GB laptop with no GPU; the five-seed
shipped model takes forty. Candidate generation is 139 items per user, so scoring all
1 260 test users is 174 092 rows. This matters for the choice: a service refreshing
recommendations nightly can run this on commodity hardware, and the heavier things we
tried — CatBoost YetiRank at 11.6× the cost, DART at the same multiple — were not better.

**User experience.** The honest description of what this system does is that it
**predicts habit**. The single strongest signal is repetition: a model that recommends
only programmes the user has already watched, weighted by how often they air, scores
0.37324 on the hold-out against the full model's 0.43949. That is 85% of the final
result from repeats alone. A viewer shown these five would mostly see the news they
already watch. Whether that is a good product is a question this data cannot answer —
the log records what people watched, not whether they were pleased — but it is what the
metric rewards, and a service adopting it should adopt it knowing that.

## Limitations, in order of how much they constrain the result

**Only 10.8% of the broadcast grid carries a programme id.** In weeks 11-20, 115 613 of
1 074 235 schedule rows have `tv_show_id != 0`. Everything else is airtime that can be
watched but can never be recommended, and a view landing there contributes nothing to any
item-level signal. The whole system is built on a tenth of the schedule.

**Ranking, not retrieval, is the binding constraint — measured.** With the current pool,
perfect ordering of the candidates we already have would score 0.8181; the model scores
0.4395. We raised the ceiling to 0.8325 by adding every programme airing on each user's
three main channels, and the model gained **nothing** (−0.0004). It cannot exploit
candidates it already fails to order.

**17.1% of correct answers are not candidates at all**, and 175 of them are premieres —
programmes that had never aired before the target window. Candidates come from the user's
own history and from the popular list, and both require having aired, so a premiere is
unreachable by construction. No feature fixes a missing candidate.

**The validation target was wrong for two days of this work.** It counted every
programme with a complete view as relevant (median 11 per user) instead of the five most
frequent at screen time ≥ 0.8. Under that definition the hold-out read 0.5586 while
Kaggle returned 0.43499 — a gap of 0.12 — and five successive feature blocks that won on
the hold-out lost on Kaggle. Correcting the target closed the gap to 0.003. Every
conclusion in this report is from after the correction; the earlier measurements are
recorded in the repository but are not evidence.

**One feature block leaked and had to be discarded.** An expected-frequency feature built
channel preferences from the whole log rather than from history alone, so on the hold-out
it could see the window it was predicting. It read +0.0272 and scored **−0.011** on
Kaggle private. The honest version is worth −0.0024.

**What cannot be concluded from this data.** The log records this service only, so a
programme absent from a user's history may simply have been watched elsewhere — "never
seen" is not something these records can establish. The log also records viewing, not
satisfaction: a completed programme is not an enjoyed one, and an unfamiliar
recommendation is not automatically a useful discovery — it may just be wrong. Both
users' content-based lists were unfamiliar and both scored zero.
