"""Count which methods get each advertisement wrong (SQ2).

If all three methods fail on the same advertisements, the limit lies in the
data or the labels rather than in any one method. If each method fails on
different advertisements, the methods genuinely differ.

The script reads the saved prediction files and trains nothing. The BERT
results use seed 1.
"""

import json
from collections import Counter

from paths import RESULTS_DIR

PERIOD = "2023_2026"
N_FOLDS = 5

FILES = {
    "tfidf": "predictions_tfidf_fold{}_" + PERIOD + ".json",
    "kbbert": "predictions_kbbert_seed1_fold{}_" + PERIOD + ".json",
    "mbert": "predictions_mbert_seed1_fold{}_" + PERIOD + ".json",
}


def load_rows(file_pattern):
    """Load the full prediction row for one method.

    Args:
        file_pattern: File name with {} where the fold number goes.

    Returns:
        A dict mapping each ad_id to its prediction row.
    """
    rows = {}

    # Each ad sits in exactly one fold, so the five files together hold
    # every ad once.
    for fold in range(N_FOLDS):
        path = RESULTS_DIR / file_pattern.format(fold)
        with path.open(encoding="utf-8") as file:
            for row in json.load(file)["predictions"]:
                rows[row["ad_id"]] = row

    return rows


def count_errors(rows, ad_ids):
    """Count which methods got each advertisement wrong.

    Args:
        rows: Prediction rows per method, from load_rows.
        ad_ids: The advertisements to go through.

    Returns:
        The combinations Counter, the errors Counter, and the list of
        ad_ids that all three methods got wrong.
    """
    combinations = Counter()
    errors = Counter()
    shared = []

    for ad_id in ad_ids:
        wrong = []
        for method in FILES:
            if not rows[method][ad_id]["correct"]:
                wrong.append(method)
                errors[method] += 1

        if len(wrong) == 3:
            shared.append(ad_id)

        if wrong:
            combinations[" + ".join(wrong)] += 1
        else:
            combinations["none, all right"] += 1

    return combinations, errors, shared


def count_pairs(rows, shared, method):
    """Count which true label was predicted as which, for one method.

    Args:
        rows: Prediction rows per method, from load_rows.
        shared: The ad_ids that all three methods got wrong.
        method: Which method's predictions to count, for example "kbbert".

    Returns:
        A Counter mapping "true -> predicted" to a number of ads.
    """
    pairs = Counter()

    for ad_id in shared:
        row = rows[method][ad_id]
        pairs[f"{row['true']} -> {row['predicted']}"] += 1

    return pairs


def print_table(counts, total, title, top=None):
    """Print a Counter as a table with counts and shares.

    Args:
        counts: The Counter to print.
        total: The number the shares are a fraction of.
        title: The heading of the first column.
        top: How many rows to print. None prints all of them.
    """
    print(f"  {title:28}{'ads':>8}{'share':>8}")
    for name, count in counts.most_common(top):
        print(f"  {name:28}{count:>8,}{count / total:>8.1%}")
    print()

def count_same_answer(rows, shared):
    """Count the shared errors where all three methods gave the same label.

    Args:
        rows: Prediction rows per method, from load_rows.
        shared: The ad_ids that all three methods got wrong.

    Returns:
        The number of shared errors where all three predictions match.
        
    """
    same = 0

    for ad_id in shared:
        answers = []
        for method in FILES:
            answers.append(rows[method][ad_id]["predicted"])

        if len(set(answers))==1:
            same+=1

    return same
                        
        



def main():
    """Print the error combinations, the chance check and the label pairs."""
    rows = {}
    for method, file_pattern in FILES.items():
        rows[method] = load_rows(file_pattern)

    ad_ids = list(rows["tfidf"])
    print(f"advertisements: {len(ad_ids):,}")
    print()

    combinations, errors, shared = count_errors(rows, ad_ids)
    print_table(combinations, len(ad_ids), "who got it wrong")

    # If the methods failed independently, the chance that all three fail on
    # the same ad would be their three error rates multiplied together.
    chance = 1.0
    for method in FILES:
        error_rate = errors[method] / len(ad_ids)
        print(f"  {method:8} wrong on {error_rate:.1%} of ads")
        chance = chance * error_rate

    expected = chance * len(ad_ids)

    print()
    print(f"  all three wrong, observed:           {len(shared):>6,}")
    print(f"  all three wrong, expected by chance: {expected:>6,.0f}")
    print(f"  observed is {len(shared) / expected:.0f} times what chance gives")
    print()

    pairs = count_pairs(rows, shared, "kbbert")
    print("  shared errors, by kbbert's prediction")
    print_table(pairs, len(shared), "true -> predicted", top=10)

    same = count_same_answer(rows, shared)
    print(f"  all three gave the same wrong label: {same:,} of {len(shared):,} "
          f"({same / len(shared):.0%})")
    
if __name__ == "__main__":
    main()
