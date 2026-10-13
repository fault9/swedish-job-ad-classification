"""Method 1: TF-IDF features with logistic regression.

This is the baseline the two BERT models get compared against. It is also the
fastest of the three to run, so it is a good place to find mistakes in the data
loading and the scoring before spending GPU time.

How it works, briefly. TF-IDF turns each advertisement into a long list of
numbers, one per word in the vocabulary. A word scores high if it appears often
in this advertisement and rarely in the others, which is what "inverse document
frequency" means. Logistic regression then learns a weight per word per class.
There is no understanding of word order or context: "inte en lärare" and "en
lärare inte" look identical to it. That limitation is the whole reason the
transformers might do better.

Two rules this script follows strictly:

  1. The vectorizer is fitted on the TRAINING texts only. Fitting it on
     train+val would let validation words influence the features, and the
     validation score would then be optimistic.

  2. The test splits are never loaded here. This script only ever sees train
     and validation. Testing happens once, later, after the configuration is
     fixed.

The hyperparameter search is deliberately small: fourteen configurations, fixed
before looking at any result. Validation holds 253 examples per label, so
trying hundreds of settings against it would start fitting the validation set
rather than choosing between genuinely different models.

Note on the grid, because it was changed. A first run used C values of 1, 4 and
16 and found that lower C was always better, with the winner sitting on the
edge of the grid. An edge winner means the search never bracketed the best
value, so the range moved down to 0.1 to 2.0. This is written down rather than
quietly fixed, because moving a grid after seeing results is only honest if you
say so. The move was driven by the shape of the trend, not by picking whichever
single number happened to score highest. The script now prints a warning if the
winner lands on an edge again.
"""

import json
import time
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from dataset import load_split
from evaluate import report
from inspect_data import save_json

from paths import PROJECT_DIR

PERIOD = "2023_2026"

SEED = 42

# The grid, written out so the budget is exactly what it says it is.
#
# C runs from 0.1 to 2.0. The first run used 1, 4 and 16 and found lower was
# always better, with the winner on the edge of the grid, so the range moved
# down.
#
# Bigrams get only three slots rather than half the grid. They scored 0.51
# against 0.57 for unigrams at every setting in the first run: with 1,170
# examples per class and 372,694 bigram features they are too sparse to help.
# Keeping a few means the write-up can say so with evidence instead of
# dropping them silently.
C_VALUES = [0.1, 0.25, 0.5, 1.0, 2.0]

GRID = (
    [((1, 1), 2, c) for c in C_VALUES]        # unigrams, the main line
    + [((1, 1), 1, c) for c in (0.25, 0.5, 1.0)]   # does keeping rare words help?
    + [((1, 1), 3, c) for c in (0.25, 0.5, 1.0)]   # does cutting more help?
    + [((1, 2), 2, c) for c in (0.25, 0.5, 1.0)]   # bigrams, for the comparison
)


def build_model(ngram_range, min_df, c_value):
    """Make a fresh vectorizer and classifier for one configuration.

    sublinear_tf=True uses 1+log(count) instead of the raw count, which stops a
    word repeated twenty times in one advertisement from dominating.

    lowercase=False keeps the Swedish capitalisation, because the project
    requires preserving case. It costs vocabulary size but it is the same text
    the BERT models get, which is the point.

    class_weight is left alone: the classes are already equal in size.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=True,
        lowercase=False,
        strip_accents=None,     # keep å, ä, ö as themselves
    )

    # No n_jobs: it has no effect in scikit-learn 1.8 and later, and passing it
    # only produces a FutureWarning on every single fit.
    classifier = LogisticRegression(
        C=c_value,
        max_iter=2000,          # enough for this to converge
        random_state=SEED,
    )

    return vectorizer, classifier


def fit_and_score(train_texts, train_labels, val_texts, val_labels,
                  ngram_range, min_df, c_value):
    """Train one configuration and return its validation macro F1."""
    vectorizer, classifier = build_model(ngram_range, min_df, c_value)

    # fit_transform on train, transform on val. Never fit on val.
    train_features = vectorizer.fit_transform(train_texts)
    val_features = vectorizer.transform(val_texts)

    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(val_features)

    macro = f1_score(val_labels, predictions, average="macro", zero_division=0)

    return macro, vectorizer, classifier, predictions


def main():
    project_dir = PROJECT_DIR

    print("loading splits")
    train_texts, train_labels = load_split(project_dir, PERIOD, "train")
    val_texts, val_labels = load_split(project_dir, PERIOD, "val")
    print(f"  train {len(train_texts):,}   val {len(val_texts):,}")
    print()

    trials = []

    print(f"searching {len(GRID)} configurations, validation macro F1 each")
    print(f"  {'ngram':8}{'min_df':>8}{'C':>7}{'macro F1':>11}{'vocab':>10}{'secs':>7}")

    for ngram_range, min_df, c_value in GRID:
        started = time.perf_counter()
        macro, vectorizer, classifier, _ = fit_and_score(
            train_texts, train_labels, val_texts, val_labels,
            ngram_range, min_df, c_value,
        )
        seconds = time.perf_counter() - started

        trials.append({
            "ngram_range": list(ngram_range),
            "min_df": min_df,
            "C": c_value,
            "val_macro_f1": macro,
            "vocab_size": len(vectorizer.vocabulary_),
            "seconds": round(seconds, 1),
        })

        print(f"  {str(ngram_range):8}{min_df:>8}{c_value:>7}"
              f"{macro:>11.4f}{len(vectorizer.vocabulary_):>10,}{seconds:>7.0f}")

    best = max(trials, key=lambda t: t["val_macro_f1"])

    # If the winner sits at either end of the C range, the search did not
    # actually bracket the best value and the grid should be moved.
    if best["C"] in (min(C_VALUES), max(C_VALUES)):
        print()
        print(f"  WARNING: best C is {best['C']}, an edge of the grid "
              f"{C_VALUES}. Widen the range in that direction and re-run.")

    print()
    print(f"best: ngram {tuple(best['ngram_range'])}  min_df {best['min_df']}  "
          f"C {best['C']}  ->  validation macro F1 {best['val_macro_f1']:.4f}")
    print()

    # Refit the winner so we can print its full scorecard on validation.
    _, _, _, predictions = fit_and_score(
        train_texts, train_labels, val_texts, val_labels,
        tuple(best["ngram_range"]), best["min_df"], best["C"],
    )

    results = report(val_labels, predictions, "best configuration, VALIDATION set")

    out = {
        "method": "tfidf_logreg",
        "period": PERIOD,
        "seed": SEED,
        "n_trials": len(trials),
        "trials": trials,
        "best": best,
        "validation": {k: results[k] for k in
                       ("n", "accuracy", "macro_f1", "majority_baseline",
                        "families", "confusion_matrix", "labels")},
    }

    out_path = project_dir / "reports" / f"tfidf_val_{PERIOD}.json"
    save_json(out, out_path)

    print()
    print("Saved:", out_path)
    print("The test set has not been touched.")


if __name__ == "__main__":
    main()
