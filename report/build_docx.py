"""Build the exam report as .docx, in the same IEEE-ish shape as the previous homework.

Every number written here is copied from a run in this repository or from a returned
Kaggle score. The sources are named in the text so the grader can check them.
"""
import re
import sys
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

ROOT = "/Users/tshermatova/Downloads/week7-exam/"
doc = Document()

# ---- page and base style ---------------------------------------------------------
for s in doc.sections:
    s.left_margin = s.right_margin = Inches(0.9)
    s.top_margin = s.bottom_margin = Inches(0.8)
st = doc.styles["Normal"]
st.font.name = "Calibri"
st.font.size = Pt(10.5)
st.paragraph_format.space_after = Pt(6)


def H(text, level=1):
    p = doc.add_heading(text, level=level)
    for r in p.runs:
        r.font.color.rgb = RGBColor(0, 0, 0)
    return p


def P(text, bold_lead=None, italic=False):
    p = doc.add_paragraph()
    if bold_lead:
        r = p.add_run(bold_lead)
        r.bold = True
    r = p.add_run(text)
    r.italic = italic
    return p


def bullets(items):
    for it in items:
        doc.add_paragraph(it, style="List Bullet")


def table(caption, rows, widths=None, header=True):
    cap = doc.add_paragraph()
    r = cap.add_run(caption)
    r.bold = True
    r.font.size = Pt(9.5)
    t = doc.add_table(rows=0, cols=len(rows[0]))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        cells = t.add_row().cells
        for c, val in zip(cells, row):
            c.text = ""
            par = c.paragraphs[0]
            par.paragraph_format.space_after = Pt(2)
            run = par.add_run(str(val))
            run.font.size = Pt(9)
            if header and i == 0:
                run.bold = True
    if widths:
        for row in t.rows:
            for c, w in zip(row.cells, widths):
                c.width = Inches(w)
    doc.add_paragraph()
    return t


# ---- title block -----------------------------------------------------------------
t = doc.add_paragraph()
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("Predicting Habitual Viewing on sweet.tv: a LambdaRank Re-ranker "
              "over Habit and Schedule Features")
r.bold = True
r.font.size = Pt(15)

for line in ["Student: Tansuluu Shermatova | Team: Individual | Email: tnshermatova@edu.hse.ru",
             "Date: 2026-10-24 | Final Exam Quiz (GSB): TV Program Recommendation on sweet.tv",
             "Repository: https://github.com/tshermatova/sweettv-tv-program-recommender"]:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(line)
    run.font.size = Pt(9.5)

P("I predict, for each of 1 260 users, the five TV programmes they will watch most often "
  "at screen time ≥ 0.8 during weeks 31–42 of 2020. Five methods are implemented over a "
  "shared candidate pool — popularity, association rules, personalised PageRank, "
  "content-based filtering and implicit matrix factorisation — and a LightGBM LambdaRank "
  "model re-ranks their signals. On a hold-out that splits history at 2020-05-18 and is "
  "scored against the organisers' own target rule, the re-ranker reaches MAP@5 = 0.43949 "
  "against 0.27387 for the popularity baseline. The submitted file returns Kaggle public "
  "0.45440 / private 0.44205, above the frozen leaderboard's first place (0.44297 / "
  "0.42912). The decisive change was not a model or a feature but the training label: "
  "copying the organisers' definition of a correct answer was worth +0.00706 on the "
  "private score, while eleven subsequent feature and capacity changes were neutral or "
  "worse.", bold_lead="Abstract — ")

P("TV recommendation, MAP@5, learning to rank, LambdaRank, implicit feedback, "
  "personalised PageRank, association rules, candidate generation, target leakage.",
  bold_lead="Index Terms — ")

# ---- 1 ---------------------------------------------------------------------------
H("1. Introduction", 1)
P("Given a view log of 5 821 987 rows covering weeks 11–30 and the broadcast schedule for "
  "weeks 11–42, predict five programme ids per user for weeks 31–42. The answer file is "
  "built by the organisers as follows: discard schedule rows with tv_show_id = 0; for each "
  "user and programme compute screen time (watched duration / programme duration) and keep "
  "airings with screen time ≥ 0.8; count how often each programme recurs at that level; take "
  "strictly the five most frequent, in descending order. The metric is MAP@5 with "
  "denominator min(5, |relevant|).", bold_lead="Problem statement. ")
P("The target is a frequency, not an affinity. A programme watched constantly but never "
  "finished is not relevant, and a film watched once and loved is not relevant either. "
  "That single property explains most of what follows, including why four of the five "
  "required methods land within 0.04 of a popularity baseline.", bold_lead="Motivation. ")
P("The user with one recorded view in the whole log is predicted three times better "
  "(3/5 hits) than a user with 3 733 views (1/5). For a sparse viewer the five most "
  "frequent programmes simply are the mass-audience news bulletins; for a viewer with 445 "
  "programmes in their repertoire, naming which five top the frequency list is a much "
  "harder ordering problem.", bold_lead="Concrete example. ")

# ---- 2 ---------------------------------------------------------------------------
H("2. Related Work", 1)
P("Two published competition solutions informed the architecture. The winner of H&M "
  "Personalized Fashion Recommendations used 100 candidates per user, six training weeks "
  "and an ensemble of LightGBM and CatBoost re-rankers, and reported that user–item "
  "interaction features mattered most while image and text features did not. Solutions to "
  "the OTTO competition built 100–172 candidates from co-visitation matrices and Word2Vec "
  "and re-ranked them with gradient-boosted trees, tuning candidate sources against "
  "measured recall.")
P("Three of their recipes were tested here and none transferred, for reasons specific to "
  "this data: negative downsampling (they have 40M rows and must; we have 1.4M and need "
  "not), a binary classifier instead of a ranker, and co-visitation as a candidate source. "
  "On this catalogue the top-20 of association rules adds zero new candidates to the pool "
  "and personalised PageRank adds zero, because both are dominated by the same popular "
  "news bulletins the popularity source already supplies.", bold_lead="Alternatives considered. ")

# ---- 3 ---------------------------------------------------------------------------
H("3. Method", 1)
P("A view is matched to a programme when both share a channel and their intervals overlap. "
  "frac = overlap / programme duration, which cannot exceed 1.0 because the overlap is "
  "capped by the programme itself. An event enters a user's history when frac ≥ 0.20. That "
  "threshold was swept and 0.10 and 0.35 were both worse, as was an absolute-seconds "
  "rule by 0.009 and 0.015 \u2014 five minutes of a fifteen-minute bulletin is a third of it, "
  "five minutes of a two-hour film is the titles. Those sweeps were run before the target "
  "definition was corrected (Section 5), so they are indicative rather than conclusive; "
  "the threshold was left at 0.20 and not re-swept afterwards. src/01_join.py, src/lib.py.",
  bold_lead="Join rule and watched threshold. ")
P("89.2% of schedule rows carry tv_show_id = 0. They never become items and never appear "
  "in any candidate list. They are not discarded entirely: the seconds spent on them still "
  "count towards a user's channel affinity, which is computed from the raw log rather than "
  "from matched events.", bold_lead="Unlabelled airtime. ")
P("History is cut at 2020-05-18, the week 20/21 boundary; the target is the ten weeks after "
  "it, scored by the organisers' rule. Temporal order is preserved and no model sees any "
  "event at or after its own cut. A second hold-out (target 04-13 to 05-18) was built as a "
  "cross-check. Models are fitted on weeks 11–30 only; weeks 31–42 contribute the broadcast "
  "grid — which programme airs on which channel at which time — and nothing else. No "
  "viewing behaviour from the test window exists in the supplied data, so there is nothing "
  "there to leak.", bold_lead="Validation split. ")
P("Each user's own watched programmes plus the 80 globally most popular, about 138 "
  "candidates per user, 174 092 rows for the 1 260 test users. The pool is identical for "
  "every method compared in this report. Recommending an already-watched programme is "
  "allowed and is in fact the strongest single signal: repeats alone, weighted by how often "
  "the programme airs in the target window, score 0.37471 against the full model's 0.43949.",
  bold_lead="Candidate pool and repeats. ")
P("There is none. All 1 260 test users appear in the view log; only two have zero matched "
  "events, eight have fewer than ten and the median is 330. For those two the model falls "
  "back to the five most popular programmes, which is also the fallback used to pad any "
  "list shorter than five — in practice padding never fires, because the popularity source "
  "guarantees at least 80 candidates per user.", bold_lead="Cold start. ")
P("Each method produces a score per (user, candidate) pair; the five highest-scoring "
  "candidates for that user, in descending score, are its answer. Scores are not comparable "
  "between methods and are never mixed: the four collaborative methods enter the final "
  "model as four feature columns, not as a blended score. Ties break by candidate order, "
  "which is deterministic.", bold_lead="How each method produces exactly five ids. ")
P("LGBMRanker, objective lambdarank, metric MAP@5, 1600 trees, learning rate 0.015, 31 "
  "leaves, min_child_samples 50, subsample 0.8 with frequency 1, colsample_bytree 0.8, "
  "seeds 20261021 + 1000i for i = 0..4, rank-averaged. Three training windows with history "
  "cut at 06-08, 06-22 and 07-06 — the hold-out's winning geometry shifted forward by 70 "
  "days so the shipped model sees the same window shapes, only later. The training label is "
  "the organisers' rule applied to each window: that user's five most frequent airings at "
  "screen time ≥ 0.8.", bold_lead="Model and hyperparameters. ")
P("anaconda Python 3.11.4, numpy 1.24.3, lightgbm 4.7.0, catboost 1.2.10, gensim 4.3.0, on "
  "a MacBook Air M1 with 8 GB and no GPU. One fit takes about six minutes; the five-seed "
  "shipped model takes forty. All runs use OMP_NUM_THREADS=2 and VECLIB_MAXIMUM_THREADS=2; "
  "without them BLAS oversubscribes on the small ALS solves and the machine slows by about "
  "a factor of ten. Builder: src/36_final_feats3.py --big --seeds 5 --top5.",
  bold_lead="Environment and runtime. ")

# ---- 4 ---------------------------------------------------------------------------
H("4. Experiments", 1)

table("Table 1. Data, join, target and candidate pool.",
      [["Quantity", "Value"],
       ["View log rows; users; labelled programmes", "5 821 987; 4 914; 4 363"],
       ["Schedule rows with tv_show_id ≠ 0 (weeks 11–20)", "115 613 of 1 074 235 (10.8%)"],
       ["Join events at frac ≥ 0.20; at frac ≥ 0.80", "3 847 696; 570 588"],
       ["Test users; with a target in the hold-out window", "1 260; 1 236 (median |R| = 5)"],
       ["Candidates per user; rows scored", "138; 174 092"],
       ["Candidate recall of true answers; MAP@5 ceiling", "0.8181; 0.8181"],
       ["Correct answers not in the pool; of those, premieres", "17.1%; 175"],
       ["Seed-to-seed sd of an identical configuration", "0.00178"]],
      widths=[4.2, 2.6])

table("Table 2. Methods on the hold-out, scored against the organisers' target. "
      "Each method ranks the same 138 candidates per user.",
      [["Method", "MAP@5"],
       ["Content-based filtering", "0.24107"],
       ["Popularity (required baseline)", "0.27387"],
       ["Association rules", "0.27538"],
       ["Matrix factorisation (iALS, 64 factors)", "0.27931"],
       ["Personalised PageRank (α = 0.85)", "0.27982"],
       ["Own repeats weighted by airings", "0.37471"],
       ["Final: LambdaRank over 41 features", "0.43949 ± 0.00095"]],
      widths=[4.2, 2.6])

table("Table 3. Kaggle submissions. The designated file is sub25_top5_s0.",
      [["Submission", "Public", "Private"],
       ["Popularity baseline", "0.29788", "0.29442"],
       ["Own repeats", "0.41510", "0.40302"],
       ["LambdaRank, 30 features", "0.43135", "0.41924"],
       ["LambdaRank, 41 features, 3 windows", "0.44008", "0.42787"],
       ["+ 1600 trees", "0.45080", "0.43499"],
       ["+ organisers' label (designated)", "0.45440", "0.44205"],
       ["Frozen leaderboard, 1st place (Magic City)", "0.44297", "0.42912"]],
      widths=[3.6, 1.6, 1.6])

table("Table 4a. Heavy user 13571199046623737884: 3 733 events over 445 programmes, "
      "pool of 450, five relevant programmes.",
      [["Method", "Top-5 (id — title [category])", "Hits"],
       ["Popularity", "2400480 ТСН. Неделя [Инфо]; 240081 Факты. День [Инфо] ✓; "
        "2400467 Викна-Новости [Инфо]; 2400508 Сегодня. Итоги [Инфо]; "
        "6500479 Мир наизнанку [Познав.]", "1/5"],
       ["Association rules", "identical list, identical order", "1/5"],
       ["Personalised PageRank", "identical list, identical order", "1/5"],
       ["Content-based", "2400480; 2500261 Подробности недели; 5900347 Деньги; "
        "240081 ✓; 2400467", "1/5"],
       ["Matrix factorisation", "10003014 Т/с 'Отдел 44'; 300358 На троих; 1000589 Отчаянные "
        "домохозяйки; 2500430 Затерянный мир; 400426 Меняю жену", "0/5"],
       ["Final ranker", "240081 ✓ at rank 1; 2400480; 2400508; 12001682 Губка Боб; 6500479", "1/5"]],
      widths=[1.5, 4.3, 0.7])

table("Table 4b. Light user 14401191285650834662: one event (90071788 Х/ф 'Призрак'), "
      "pool of 81, five relevant programmes.",
      [["Method", "Top-5 (id — title [category])", "Hits"],
       ["Popularity", "2400480; 240081 ✓; 2400467 ✓; 2400508 ✓; 6500479", "3/5"],
       ["Association rules", "identical list, identical order", "3/5"],
       ["Personalised PageRank", "90071788 Призрак; 2400480; 240081 ✓; 2400467 ✓; 2400508 ✓", "3/5"],
       ["Content-based", "90071788; 10003159 Ничто не случается дважды; 10003178 Агенты "
        "справедливости; 10001657 Касл; 10003102 Опер по вызову", "0/5"],
       ["Matrix factorisation", "90077687 Праведник; 90077217 Геракл; 1000589; 90076295 Призрачный "
        "патруль; 90064639 Пророк", "0/5"],
       ["Final ranker", "240081 ✓; 2400508 ✓; 2400480; 300358 На троих; 2400467 ✓", "3/5"]],
      widths=[1.5, 4.3, 0.7])

table("Table 5. Everything tested after the target definition was corrected. "
      "Hold-out deltas against the shipped 0.43949.",
      [["Change", "Δ hold-out", "Verdict"],
       ["Training label = organisers' top-5 rule", "+0.0079", "accepted, +0.00706 private"],
       ["Expected-frequency features (honest)", "−0.0024", "rejected"],
       ["Graded label, rank 5..1 within the five", "−0.0021", "rejected"],
       ["Graded label, airing count", "−0.0017", "rejected"],
       ["Broadcast-history features", "−0.0013", "rejected"],
       ["Candidates restricted to airing programmes", "−0.0014", "rejected"],
       ["Pool + all programmes on top-3 channels", "−0.0004", "rejected (ceiling +0.035)"],
       ["2400 trees / 800 trees / 63 leaves / 15 leaves", "within ±0.0011", "no effect"],
       ["Cascade re-ranking the top 20", "−0.0064", "rejected"]],
      widths=[3.5, 1.3, 2.0])

H("Designated submission", 2)
P("sub25_top5_s0.csv, Kaggle ref 57573516. The screenshot below is the Submissions tab: "
  "the left number is the Private Score (0.44205) and the right is the Public Score "
  "(0.45440).")
doc.add_picture(ROOT + "report/img/kaggle_submission.png", width=Inches(6.5))
cap = doc.add_paragraph()
cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = cap.add_run("Figure 1. Returned Kaggle scores for the designated submission: "
                "private 0.44205, public 0.45440.")
r.font.size = Pt(9)
r.italic = True

# ---- 5: the business analysis ----------------------------------------------------
H("5. Discussion", 1)
src = open(ROOT + "report/section8.md", encoding="utf-8").read()
for block in src.split("\n\n"):
    b = block.strip()
    if not b or b.startswith("# "):
        continue
    if b.startswith("## "):
        H(b[3:].strip(), 2)
        continue
    if b.startswith("|"):
        rows = [[c.strip() for c in ln.strip().strip("|").split("|")]
                for ln in b.split("\n") if ln.strip().startswith("|")]
        rows = [r_ for r_ in rows if not all(set(c) <= set("-: ") for c in r_)]
        rows = [[re.sub(r"\*\*(.+?)\*\*", r"\1", c).replace("`", "") for c in r_]
                for r_ in rows]
        if rows:
            table("Table 6. The two users, hits out of five.", rows, widths=[3.0, 1.9, 1.9])
        continue
    b = b.replace("\n", " ")
    lead = None
    if b.startswith("**"):
        end = b.find("**", 2)
        if end > 0:
            lead, b = b[2:end], b[end + 2:]
    # strip the remaining markdown: bold, italics and code spans otherwise render as
    # literal asterisks and backticks in Word
    b = re.sub(r"\*\*(.+?)\*\*", r"\1", b)
    b = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"\1", b)
    b = b.replace("`", "")
    P(b, bold_lead=(lead + " ") if lead else None)

# ---- 6 ---------------------------------------------------------------------------
H("6. AI Usage Disclosure", 1)
P("The implementation, the experiments and the drafting of this report were carried out "
  "with Claude Code (Claude Opus 5) over three working sessions, in the manner the course "
  "permits. The complete session export is submitted alongside this PDF. Every number "
  "reported here is produced by a script in the repository; none is quoted from the "
  "assistant's recollection. Two errors found during the work are documented rather than "
  "removed, because they bound how much the earlier measurements can be trusted: the "
  "validation target was defined incorrectly for the first two days, and one feature block "
  "leaked future channel activity into the hold-out. Both are described in Section 5 and "
  "in BRIEF.md in the repository.")

H("References", 1)
for i, ref in enumerate([
    "sweet.tv — TV program recommender. Kaggle community competition. "
    "https://www.kaggle.com/competitions/sweettv-tv-program-recommender",
    "senkin13 and h4211819. 1st place solution, H&M Personalized Fashion Recommendations, "
    "Kaggle, 2022.",
    "C. Deotte. Kaggle-OTTO-Comp, 3rd place solution. https://github.com/cdeotte/Kaggle-OTTO-Comp",
    "N. Vicol. otto-recommender: candidate sources with measured recall. "
    "https://github.com/nicolaivicol/otto-recommender",
    "Q. Wu et al. Adapting boosting for information retrieval measures. "
    "Information Retrieval 13(3), 2010 (LambdaMART).",
    "Y. Hu, Y. Koren, C. Volinsky. Collaborative filtering for implicit feedback datasets. "
    "ICDM 2008 (iALS).",
], 1):
    p = doc.add_paragraph(f"[{i}] {ref}")
    p.paragraph_format.space_after = Pt(3)
    for r_ in p.runs:
        r_.font.size = Pt(9)

out = ROOT + "report/shermatova_tansuluu_exam_report.docx"
doc.save(out)
print("saved:", out)
