"""Which truncation strategy should all three methods use?

The input has to be cut down, because both BERT models stop at 512 tokens. The
question is which part of the advertisement to keep.

The structural argument came first, and it is model-free. Measured on the train
split, the share of each region that contains administrative boilerplate —
application instructions, URLs, email addresses, collective agreement text,
recruiter disclaimers:

    first 100 words    12.2%
    middle             49.8%
    last 100 words     84.3%

Swedish job advertisements are front-loaded: title and duties first,
requirements in the middle, "så ansöker du" and the company blurb last. So
head+tail spends half its budget on the least informative region, and at the
median length of 424 words it throws away 53% of the text from the middle,
which is where the requirements sit.

That reasoning points to head-only. This script is the confirmation, not the
decision: it runs the same TF-IDF model over several strategies and budgets so
the structural argument can be checked against a number.

Note on honesty. Choosing the input by one method's validation score would tune
the shared input on TF-IDF's behaviour, which is the kind of circularity the
label-selection protocol was written to avoid. So the decision rests on the
measurement above. If this script disagrees, the structural reason wins and
both numbers get reported.
"""

import json
import time

from sklearn.metrics import f1_score

from dataset import load_split
from inspect_data import save_json
from paths import PROJECT_DIR, RESULTS_DIR
from train_tfidf import build_model

PERIOD = "2023_2026"
BUDGET = 200          # fixed: the largest budget both tokenizers fit at 95%+

# Which 200 words to keep. "none" is the ceiling if length were no constraint.
#
# The head/tail rows from the earlier sweep are not repeated: head beat
# head_tail by 17.5 points and the trend was monotonic in how much tail was
# kept. What is open is whether a stronger gate makes 200 words carry more,
# since the real missing signal sits in words 200-300 rather than in the tail.
SETTINGS = [
    ("head", BUDGET),              # first 200 words, the current default
    ("gated", BUDGET),             # narrow gate: how-to-apply text removed
    ("gated_wide", BUDGET),        # wide gate: conditions, benefits, values too
    ("gated_wide_tail", BUDGET),   # wide gate, then 150 from the start and 50 from the end
    ("skip_intro", BUDGET),        # start at the job-section heading, skip the employer intro
    ("skip_intro_wide", BUDGET),   # the same with six more headings
    ("opening_job", BUDGET),       # first 30 words, then the job section
    ("opening_job_wide", BUDGET),  # the same with six more headings
    ("head", 250),                 # reference: what more head buys
    ("head", 300),                 # reference, but mBERT only fits 20.8% here
    ("none", 0),                   # ceiling
]

# Fixed, so only the input changes between rows. These are the settings that
# won the TF-IDF search on head+tail at 200 words.
NGRAM = (1, 1)
MIN_DF = 1
C_VALUE = 0.5


def score(strategy, budget):
    """Fit on train and score on validation for one input setting."""
    truncate = strategy != "none"

    train_texts, train_labels = load_split(
        PROJECT_DIR, PERIOD, "train", budget=budget,
        truncate=truncate, strategy=strategy,
    )
    val_texts, val_labels = load_split(
        PROJECT_DIR, PERIOD, "val", budget=budget,
        truncate=truncate, strategy=strategy,
    )

    vectorizer, classifier = build_model(NGRAM, MIN_DF, C_VALUE)

    train_features = vectorizer.fit_transform(train_texts)
    classifier.fit(train_features, train_labels)

    predictions = classifier.predict(vectorizer.transform(val_texts))
    macro = f1_score(val_labels, predictions, average="macro", zero_division=0)

    median_words = sorted(len(t.split()) for t in train_texts)[len(train_texts) // 2]

    return macro, median_words, len(vectorizer.vocabulary_)


def main():
    print(f"TF-IDF fixed at ngram {NGRAM}, min_df {MIN_DF}, C {C_VALUE}")
    print("Only the input text changes between rows.")
    print()
    print(f"  {'strategy':12}{'budget':>8}{'median words':>14}"
          f"{'vocab':>10}{'val macro F1':>14}{'secs':>7}")

    rows = []

    for strategy, budget in SETTINGS:
        started = time.perf_counter()
        macro, median_words, vocab = score(strategy, budget)
        seconds = time.perf_counter() - started

        rows.append({
            "strategy": strategy,
            "budget": budget,
            "median_words": median_words,
            "vocab_size": vocab,
            "val_macro_f1": macro,
            "seconds": round(seconds, 1),
        })

        shown = "-" if strategy == "none" else str(budget)
        print(f"  {strategy:12}{shown:>8}{median_words:>14,}"
              f"{vocab:>10,}{macro:>14.4f}{seconds:>7.0f}")

    best = max(rows, key=lambda r: r["val_macro_f1"])

    print()
    print(f"highest validation macro F1: {best['strategy']} at budget "
          f"{best['budget']} -> {best['val_macro_f1']:.4f}")

    # The comparison that actually matters: same budget, different strategy.
    pairs = {(r["strategy"], r["budget"]): r["val_macro_f1"] for r in rows}

    for budget in (200, 300):
        head = pairs.get(("head", budget))
        both = pairs.get(("head_tail", budget))
        if head is not None and both is not None:
            print(f"  at {budget} words: head {head:.4f} vs head_tail {both:.4f}"
                  f"   difference {head - both:+.4f}")

    save_json({"period": PERIOD, "ngram": list(NGRAM), "min_df": MIN_DF,
               "C": C_VALUE, "rows": rows, "best": best},
              RESULTS_DIR / f"truncation_{PERIOD}.json")

    print()
    print("Saved:", RESULTS_DIR / f"truncation_{PERIOD}.json")
    print("The test set has not been touched.")


if __name__ == "__main__":
    main()
