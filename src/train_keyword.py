"""The floor: one keyword per label, no learning.

This is on purpose the stupidest classifier that could work. For each of the
nine labels it finds the single word that best identifies that label in the
training data, and then classifies an advertisement by which of those nine
words appears in it.

Why have it. The majority baseline is 11.1% with nine equal classes, which is
too weak to be informative: any method beating it has learned something, but
that tells you nothing about how much. The keyword baseline answers a sharper
question — how much of this task is just spotting one word?

    majority baseline   0.111    guess the most common class
    keyword baseline    ?        this script
    TF-IDF              0.779    weighted counts of every word
    KB-BERT, mBERT      ?        context and word order

If the keyword rule gets close to TF-IDF, then almost the whole task is
lexical, and that is a finding rather than a disappointment.

IT ALSO CATCHES BROKEN PIPELINES. Earlier in this project TF-IDF scored 0.580
because the input truncation was keeping the wrong half of each advertisement.
A nine-word lookup table had scored 0.604 on a comparable setup, so the trained
model was losing to nine words. Nobody noticed, because there was no floor to
compare against. A trained model that cannot beat this script means something
is wrong with the data, not the model.

Two rules had to be decided, and both are stated here rather than buried:

  several keywords match   take the label whose keyword has the highest
                           training precision, because that is the keyword we
                           have most reason to trust
  no keyword matches       fall back to the label with the most training
                           examples, which with balanced classes is just the
                           first label alphabetically
"""

import re
from collections import Counter, defaultdict

from dataset import INPUT_STRATEGY, load_fold
from evaluate import bootstrap_macro_f1, macro_f1, report, save_predictions
from inspect_data import save_json
from paths import PROJECT_DIR, RESULTS_DIR

PERIOD = "2023_2026"
N_FOLDS = 5

MIN_RECALL = 0.25   # the word must appear in at least this share of the label
MIN_LENGTH = 4      # ignore very short words like "och", "att", "för"

WORD = re.compile(r"[a-zåäöéüA-ZÅÄÖÉÜ]{%d,}" % MIN_LENGTH)


def words_in(text):
    """The set of distinct words in one text, lowercased."""
    return {word.lower() for word in WORD.findall(text)}


def count_documents(texts, labels):
    """Count, for every word, how many documents of each label contain it.

    Returns (per_word, per_label) where per_word[word][label] is a document
    count, and per_label[label] is how many documents that label has.
    """
    per_word = defaultdict(Counter)
    per_label = Counter()

    for text, label in zip(texts, labels):
        per_label[label] += 1
        for word in words_in(text):
            per_word[word][label] += 1

    return per_word, per_label


def pick_keywords(texts, labels):
    """Find the best single word for each label. Fitted on training data only.

    For every word and every label we have two numbers:

      recall     of this label's advertisements, how many contain the word
      precision  of all advertisements containing the word, how many are this
                 label

    We want high precision, because that is what makes the word a usable
    signal. But precision alone would pick a word appearing in three
    advertisements with perfect precision and useless coverage, so a word has
    to reach MIN_RECALL before it is considered.

    Returns {label: (word, precision, recall)}.
    """
    per_word, per_label = count_documents(texts, labels)

    best = {}

    for label in sorted(per_label):
        best_word = None
        best_precision = -1.0
        best_recall = 0.0

        for word, counts in per_word.items():
            in_label = counts[label]
            if in_label == 0:
                continue

            recall = in_label / per_label[label]
            if recall < MIN_RECALL:
                continue

            precision = in_label / sum(counts.values())

            # Highest precision wins; recall breaks ties, so of two equally
            # precise words we keep the one covering more of the label.
            if (precision, recall) > (best_precision, best_recall):
                best_word = word
                best_precision = precision
                best_recall = recall

        best[label] = (best_word, best_precision, best_recall)

    return best


def predict(texts, keywords, fallback):
    """Classify each text by which of the nine keywords it contains."""
    # Sort once, most trustworthy keyword first, so ties resolve consistently.
    ordered = sorted(keywords.items(), key=lambda item: -item[1][1])

    predictions = []

    for text in texts:
        found = words_in(text)
        chosen = fallback
        for label, (word, _precision, _recall) in ordered:
            if word is not None and word in found:
                chosen = label
                break
        predictions.append(chosen)

    return predictions


def main():
    print(f"Keyword baseline across {N_FOLDS} folds")
    print(f"  a word must appear in at least {MIN_RECALL:.0%} of a label's "
          f"training advertisements to be eligible")
    print()

    all_units = []
    all_true = []
    all_predicted = []
    fold_rows = []

    for fold_number in range(N_FOLDS):
        data = load_fold(PROJECT_DIR, PERIOD, fold_number)

        # Fitted on the training folds only, like every other method here.
        keywords = pick_keywords(data["train_texts"], data["train_labels"])
        fallback = Counter(data["train_labels"]).most_common(1)[0][0]

        predictions = predict(data["test_texts"], keywords, fallback)
        score = macro_f1(data["test_labels"], predictions)

        all_units += data["test_units"]
        all_true += data["test_labels"]
        all_predicted += predictions

        fold_rows.append({
            "fold": fold_number,
            "macro_f1": score,
            "keywords": {label: {"word": word, "precision": precision,
                                 "recall": recall}
                         for label, (word, precision, recall) in keywords.items()},
        })

        print(f"  fold {fold_number}: macro F1 {score:.4f}")
        if fold_number == 0:
            print("    the words it chose:")
            for label, (word, precision, recall) in sorted(keywords.items()):
                print(f"      {label}  {word!r:28} precision {precision:.2f}"
                      f"  recall {recall:.2f}")

        save_predictions(
            data["test_units"], data["test_labels"], predictions,
            RESULTS_DIR / f"predictions_keyword_fold{fold_number}_{PERIOD}.json",
            method="keyword", split=f"fold{fold_number}",
        )

    scores = [row["macro_f1"] for row in fold_rows]

    print()
    print(f"  per-fold mean {sum(scores)/len(scores):.4f}   "
          f"lowest {min(scores):.4f}   highest {max(scores):.4f}")
    print()

    print("=" * 62)
    results = report(all_true, all_predicted,
                     f"POOLED over {N_FOLDS} folds, keyword baseline")

    interval = bootstrap_macro_f1(all_units, all_true, all_predicted)

    print()
    print(f"  macro F1   {interval['macro_f1']:.4f}")
    print(f"  95% CI     [{interval['ci_low']:.4f}, {interval['ci_high']:.4f}]")
    print()
    print(f"  for comparison, TF-IDF across the same folds: 0.7786")

    save_json(
        {
            "method": "keyword",
            "input": INPUT_STRATEGY,
            "period": PERIOD,
            "n_folds": N_FOLDS,
            "min_recall": MIN_RECALL,
            "min_word_length": MIN_LENGTH,
            "folds": fold_rows,
            "per_fold_mean": sum(scores) / len(scores),
            "pooled": {k: results[k] for k in
                       ("n", "accuracy", "macro_f1", "majority_baseline",
                        "families", "confusion_matrix", "labels")},
            "pooled_interval": interval,
        },
        RESULTS_DIR / f"keyword_cv_{PERIOD}.json",
    )

    print()
    print("Saved:", RESULTS_DIR / f"keyword_cv_{PERIOD}.json")


if __name__ == "__main__":
    main()
