"""Method 1 measured properly: TF-IDF across five employer-disjoint folds.

train_tfidf.py chose the settings on the frozen validation split. This script
does not choose anything. It takes those settings, runs them on all five folds,
and reports how good the method actually is.

Why not just use the single split. Its test set holds 253 advertisements per
label from 317 employers, which gives a 95% interval of about plus or minus 3.8
points on macro F1. Two methods would have to differ by roughly 5 points before
the difference could be called established. Across five folds every
advertisement takes a turn in the test set, so the estimate rests on 1,600 per
label instead of 253.

Two numbers come out, and they answer different questions.

  per-fold spread   Five scores, one per fold. Their range shows how much the
                    answer depends on which employers happened to be held out.

  pooled interval   All 14,400 predictions joined together, with a confidence
                    interval from resampling employers. This is the number to
                    report, and the one the BERT runs get compared against.

The predictions are saved per fold. Three later analyses need them and none is
possible from scores alone: whether the methods fail on the same
advertisements, whether errors sit in the off-type occupations inside a label,
and whether errors cluster by employer.

NOTHING IS TUNED HERE. If you find yourself editing the settings below after
seeing a score, stop: that is what the frozen validation split was for.
"""

import json
import time
from collections import Counter

from sklearn.metrics import f1_score

from dataset import load_fold
from evaluate import (bootstrap_macro_f1, family_errors, macro_f1,
                      report, save_predictions)
from inspect_data import save_json
from paths import PROJECT_DIR, REPORTS_DIR
from train_tfidf import build_model

PERIOD = "2023_2026"
N_FOLDS = 5

# Chosen on the frozen validation split by train_tfidf.py, which scored 0.7691.
# Fixed here on purpose.
NGRAM = (1, 1)
MIN_DF = 2
C_VALUE = 0.1
LOWERCASE = True


def run_fold(fold_number):
    """Train on four folds, predict the fifth, return predictions and the score."""
    data = load_fold(PROJECT_DIR, PERIOD, fold_number)

    vectorizer, classifier = build_model(NGRAM, MIN_DF, C_VALUE, LOWERCASE)

    # Fit the vectorizer on the training folds only. Fitting it on all five
    # would let the test fold's vocabulary shape the features.
    train_features = vectorizer.fit_transform(data["train_texts"])
    classifier.fit(train_features, data["train_labels"])

    predictions = list(classifier.predict(vectorizer.transform(data["test_texts"])))

    return data, predictions, len(vectorizer.vocabulary_)


def main():
    print(f"TF-IDF across {N_FOLDS} folds")
    print(f"  settings from the frozen split: ngram {NGRAM}, min_df {MIN_DF}, "
          f"C {C_VALUE}, lowercase {LOWERCASE}")
    print()

    all_units = []
    all_true = []
    all_predicted = []
    fold_rows = []

    print(f"  {'fold':6}{'train':>9}{'test':>8}{'vocab':>10}"
          f"{'macro F1':>11}{'secs':>7}")

    for fold_number in range(N_FOLDS):
        started = time.perf_counter()
        data, predictions, vocab = run_fold(fold_number)
        seconds = time.perf_counter() - started

        score = macro_f1(data["test_labels"], predictions)

        all_units += data["test_units"]
        all_true += data["test_labels"]
        all_predicted += predictions

        fold_rows.append({
            "fold": fold_number,
            "n_train": len(data["train_texts"]),
            "n_test": len(data["test_texts"]),
            "vocab_size": vocab,
            "macro_f1": score,
            "seconds": round(seconds, 1),
        })

        print(f"  {fold_number:<6}{len(data['train_texts']):>9,}"
              f"{len(data['test_texts']):>8,}{vocab:>10,}"
              f"{score:>11.4f}{seconds:>7.0f}")

        save_predictions(
            data["test_units"], data["test_labels"], predictions,
            REPORTS_DIR / f"predictions_tfidf_fold{fold_number}_{PERIOD}.json",
            method="tfidf_logreg", split=f"fold{fold_number}",
        )

    scores = [row["macro_f1"] for row in fold_rows]
    mean = sum(scores) / len(scores)

    print()
    print(f"  per-fold mean {mean:.4f}   lowest {min(scores):.4f}   "
          f"highest {max(scores):.4f}   spread {max(scores) - min(scores):.4f}")
    print()

    # The pooled result: every advertisement predicted exactly once.
    print("=" * 62)
    results = report(all_true, all_predicted,
                     f"POOLED over {N_FOLDS} folds, every advertisement once")

    print()
    print("  bootstrapping the interval by resampling employers ...")
    interval = bootstrap_macro_f1(all_units, all_true, all_predicted)

    print(f"  macro F1   {interval['macro_f1']:.4f}")
    print(f"  95% CI     [{interval['ci_low']:.4f}, {interval['ci_high']:.4f}]"
          f"   width {interval['ci_high'] - interval['ci_low']:.4f}")
    print(f"  from       {interval['n_advertisements']:,} advertisements, "
          f"{interval['n_employers']:,} employers")

    save_json(
        {
            "method": "tfidf_logreg",
            "period": PERIOD,
            "n_folds": N_FOLDS,
            "settings": {"ngram_range": list(NGRAM), "min_df": MIN_DF,
                         "C": C_VALUE, "lowercase": LOWERCASE,
                         "chosen_on": "frozen validation split"},
            "folds": fold_rows,
            "per_fold_mean": mean,
            "per_fold_spread": max(scores) - min(scores),
            "pooled": {k: results[k] for k in
                       ("n", "accuracy", "macro_f1", "majority_baseline",
                        "families", "confusion_matrix", "labels")},
            "pooled_interval": interval,
        },
        REPORTS_DIR / f"tfidf_cv_{PERIOD}.json",
    )

    print()
    print("Saved:", REPORTS_DIR / f"tfidf_cv_{PERIOD}.json")
    print(f"Saved: {N_FOLDS} per-fold prediction files")


if __name__ == "__main__":
    main()
