"""Compare the methods pairwise on the same advertisements (H1 and H2).

For each BERT seed, KB-BERT and mBERT are compared with TF-IDF (H1), and
KB-BERT with mBERT (H2). The script reads the saved prediction files and
trains nothing.
"""

from evaluate import compare_methods
from files import load_json, save_json
from paths import PERIOD, RESULTS_DIR

N_FOLDS = 5
SEEDS = [1, 2, 3]


def load_rows(name):
    """Load the prediction rows of one run from all folds.

    Args:
        name: The run name in the file names, for example "kbbert_seed1".

    Returns:
        A dict mapping each ad_id to its prediction row.
    """
    rows = {}

    for fold in range(N_FOLDS):
        path = RESULTS_DIR / f"predictions_{name}_fold{fold}_{PERIOD}.json"
        for row in load_json(path)["predictions"]:
            rows[row["ad_id"]] = row

    return rows


def main():
    """Run the nine comparisons, print them and save them."""
    names = ["tfidf"] + [f"{method}_seed{seed}"
                         for method in ["kbbert", "mbert"] for seed in SEEDS]
    rows = {name: load_rows(name) for name in names}

    # Every run must be lined up on the same advertisements, in the same order.
    ad_ids = list(rows["tfidf"])
    units = [rows["tfidf"][ad_id] for ad_id in ad_ids]
    true_labels = [unit["true"] for unit in units]
    predictions = {name: [rows[name][ad_id]["predicted"] for ad_id in ad_ids]
                   for name in names}

    comparisons = []

    for seed in SEEDS:
        pairs = [(f"kbbert_seed{seed}", "tfidf"),
                 (f"mbert_seed{seed}", "tfidf"),
                 (f"kbbert_seed{seed}", f"mbert_seed{seed}")]

        for name_a, name_b in pairs:
            result = compare_methods(units, true_labels, predictions[name_a],
                                     predictions[name_b], name_a, name_b)
            comparisons.append(result)

            print(f"{name_a:13} - {name_b:13}"
                  f" {result['macro_f1_difference'] * 100:+.2f}"
                  f" [{result['ci_low'] * 100:+.2f}, {result['ci_high'] * 100:+.2f}]"
                  f"  McNemar p {result['mcnemar_p']:.2g}"
                  f"  established {result['established']}", flush=True)

    path = RESULTS_DIR / f"comparisons_{PERIOD}.json"
    save_json({"period": PERIOD, "comparisons": comparisons}, path)
    print("Saved:", path)


if __name__ == "__main__":
    main()
