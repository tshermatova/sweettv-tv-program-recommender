# Section 8 — Which recommendation strategy to select, and why

## What this section can and cannot show

Two users cannot establish that one method is generally better than another, and nothing
below shows that any viewer enjoyed anything: the log records viewing, not satisfaction.
No online experiment was run and none is claimed. Every number comes from the hold-out or
from a returned Kaggle score, both listed in the repository.

## The target drives everything that follows

The organisers' answer file is built like this: keep views whose screen time — watched
duration divided by programme duration — is at least 0.8; count how often each programme
recurs at that level; take **strictly the five most frequent**, in descending order.

Three consequences shape every result in this section.

**The relevant set is closed at five.** `|R| = 5` for 90.9% of users, so the metric's
denominator `min(5, |R|)` is fixed. There is no long tail to recall: the task is to name
a small, closed set of habitual, fully-watched programmes.

**Order is paid for.** The specification's own worked example scores the same three hits
at 0.9167 or 0.5333 depending only on where they sit — a difference of 0.38 from
reordering alone.

**Frequency, not affinity, is the currency.** A programme someone watches constantly but
never finishes is not relevant. A film someone loves and watches once is not relevant
either. This is why a defensible recommendation can score zero, as one does below.

## The two users

Both have a full five-programme target. They are the extremes of evidence among the 1 118
test users who have both a history and something to predict: one with 3 733 viewing
events over 445 distinct programmes, one with **a single event**. Neither is a cold-start
case — all 1 260 test users appear in the log.

| method | heavy: 3 733 events | light: 1 event |
| --- | ---: | ---: |
| popularity (baseline) | 1/5 | 3/5 |
| association rules | 1/5 | 3/5 |
| personalised PageRank | 1/5 | 3/5 |
| content-based | 1/5 | 0/5 |
| matrix factorisation (iALS) | 0/5 | 0/5 |
| **final ranker** | **1/5** | **3/5** |

**The user with one recorded view is predicted three times better than the user with
3 733.** That inversion is not a defect; it follows from the target. For a sparse viewer
the five most frequent programmes simply *are* the mass-audience bulletins, and
popularity names them. For a viewer with 445 programmes in their repertoire, naming which
five top their frequency list is a fine-grained ordering problem, and every method here
finds one of them.

The uncomfortable practical reading: **on this platform personalisation earns least
exactly where there is most to personalise from.**

## How the methods differ, and where they do not

**Popularity, association rules and PageRank returned byte-identical lists for the heavy
user** — the same five ids in the same order, hitting the same one (240081, *Факты.
День*) at rank 2. The mechanism is appointment viewing. The heavy user's own history is
538 views of *ТСН. Неделя*, 244 of *Завтрак с 1+1*, 222 of *Факты. День*: daily news on a
fixed schedule. Two such programmes are "watched together" mostly because nearly everyone
watches both, so the co-view matrix is a popularity matrix with extra steps, and
PageRank's diffusion over that graph returns to the same high-degree nodes.

This was measured directly rather than inferred: adding the top-20 items of association
rules to the candidate pool adds **zero** new candidates, PageRank **zero**,
content-based four, iALS about zero. On this catalogue those methods do not propose
anything popularity has not already proposed. Association rules would capture
complementary consumption — what you watch *because* you watched something else — and
here there is almost none to capture at the item level.

**Content-based filtering is the only method that proposes genuinely different things,
and for the light user it scores zero.** Given one watched film, *Призрак*, it returned
five films and series of the same kind. By any editorial standard those are reasonable
recommendations. They earn nothing, because the target counts complete viewings that
*recur*, and a particular film is not watched five times. Content attributes contribute
where the co-view signal is thin — and what they contribute is the wrong kind of
relevance for this metric.

**Matrix factorisation disagrees most and is wrong both times.** It alone put series and
entertainment ahead of news for the heavy user, and films for the light one. iALS learns
latent taste from a matrix whose mass is habitual viewing; asked to extrapolate twelve
weeks forward it extrapolates taste, when the correct answer is habit. It is the clearest
illustration in this comparison that taste and habit are different objects.

**Popularity already does well in the sparse tail**, and not by understanding anyone: for
a sparse user the target *is* the population mode.

**What the final ranker adds is ordering, not content.** For the heavy user it finds the
same single correct programme as popularity and places it **first instead of second**,
which is exactly what MAP@5 pays for. For the light user it finds three, two of them in
the top two positions. Across all users that ordering effect is worth 0.43949 against
0.27387 for popularity: the ranker is not discovering different programmes, it is
arranging the same ones better.

## Why this strategy

**Validation.** Of the eleven alternatives tested after the target definition was
corrected, ten were neutral or worse: expected-frequency features (−0.0024), two graded
label encodings (−0.0021 and −0.0017), broadcast-history features (−0.0013), restricting
candidates to programmes that actually air (−0.0014), expanding the pool with every
programme on the user's three main channels (−0.0004), four capacity settings (all within
±0.0011), and a two-stage cascade re-ranking the top twenty (−0.0064). Exactly one change
helped: aligning the training label to the organisers' own rule, worth +0.0079 on the
hold-out and **+0.00706 on Kaggle private**.

**Cost.** One fit is about six minutes on an 8 GB laptop with no GPU; the shipped
five-seed model takes forty. Scoring all 1 260 users means 174 092 candidate rows. A
service refreshing recommendations nightly can run this on commodity hardware — and the
heavier alternatives were not better: CatBoost YetiRank and DART both cost about 11.6×
and both lost.

**User experience.** The honest description is that this system **predicts habit**.
Recommending only programmes the user has already watched, weighted by how often they
air, scores 0.37471 against the full model's 0.43949 — 85% of the result from repeats
alone. A viewer shown these five would mostly see the news they already watch. Whether
that makes a good product is not answerable from this data; it is what the metric
rewards, and a service adopting it should adopt it knowing that.

## Limitations

**Only 10.8% of the broadcast grid carries a programme id.** In weeks 11-20, 115 613 of
1 074 235 schedule rows have `tv_show_id != 0`. The rest is airtime that can be watched
but never recommended. Every method compared here is choosing within a tenth of the
schedule, by construction and not by a modelling choice that could be tuned away.

**Ranking, not retrieval, is the binding constraint — measured.** Perfect ordering of the
candidates we already have would score 0.8181; the model scores 0.43949. Raising the
ceiling to 0.8325 by adding every programme on each user's three main channels changed
the score by −0.0004. The model cannot exploit candidates it already fails to order.

**17.1% of correct answers are not candidates at all**, and 175 of them are premieres —
programmes that had never aired before the target window. Candidates come from the user's
history and from the popular list, and both require having aired, so a premiere is
unreachable by construction.

**The validation target was wrong for two days of this work.** It counted every programme
with a complete view as relevant (median 11 per user) rather than the five most frequent
at screen time ≥ 0.8. Under that definition the hold-out read 0.5586 while Kaggle returned
0.43499; correcting it closed the gap to 0.003. Only measurements made afterwards are used
here.

**One feature block leaked and was discarded.** An expected-frequency feature built
channel preferences from the whole log instead of from history alone, so on the hold-out
it could see the window it was predicting: +0.0272 there, **−0.011** on Kaggle private.
The honest version is worth −0.0024 and is not shipped.

**What this data cannot establish.** The log records this service only, so a programme
absent from a user's history may have been watched elsewhere — "never seen" is not
something these records can support. An unfamiliar recommendation is not automatically a
useful discovery; it may simply be wrong. Both content-based lists in the table above were
unfamiliar, and both scored zero.
