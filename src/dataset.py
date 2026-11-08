"""Load the frozen splits and folds as model input.

Every method loads its data here, so they all read exactly the same text.
"""

from files import load_json
from make_splits import check_alignment
from paths import CANDIDATES_FILE, FOLDS_FILE, SPLITS_FILE
from truncate import INPUT_STRATEGY, WORD_BUDGET, truncate_text


def load_split(name, budget=WORD_BUDGET, truncate=True,
               strategy=INPUT_STRATEGY):
    """Load the texts and labels of one split.

    Args:
        name: "train", "val" or "test".
        budget: How many words to keep per text.
        truncate: False returns the whole texts.
        strategy: Which words to keep, see truncate.py.

    Returns:
        Two matching lists: texts and labels.
    """
    candidates = load_json(CANDIDATES_FILE)
    splits = load_json(SPLITS_FILE)

    # Stop if the candidate file has changed since the splits were frozen.
    check_alignment(splits, candidates)

    units = splits["splits"][name]

    texts, labels = [], []

    for unit in units:
        text = candidates[unit["position"]]["text"]
        text = truncate_text(text, budget, strategy if truncate else "none")
        texts.append(text)
        labels.append(unit["label"])

    return texts, labels


def load_units(name):
    """Return the ad_id, cluster, employer, label and position of each ad in a split."""
    splits = load_json(SPLITS_FILE)

    return splits["splits"][name]


def load_fold(fold_number, budget=WORD_BUDGET, strategy=INPUT_STRATEGY):
    """Load one fold as test data and the other four as training data.

    Args:
        fold_number: Which fold to test on, 0 to 4.
        budget: How many words to keep per text.
        strategy: Which words to keep, see truncate.py.

    Returns:
        A dict with test_texts, test_labels, test_units, train_texts,
        train_labels and n_folds.
    """
    candidates = load_json(CANDIDATES_FILE)
    saved = load_json(FOLDS_FILE)

    if len(candidates) != saved["candidate_count"]:
        raise ValueError(
            f"Candidate file has {len(candidates):,} records but the folds were "
            f"built from {saved['candidate_count']:,}. Rebuild the folds."
        )

    def to_texts(units):
        """Return the truncated text of each unit, after checking it is the right advertisement."""
        texts = []
        for unit in units:
            # Check that the position still points at the same ad.
            actual = candidates[unit["position"]]["source_ad_ids"][0]
            if actual != unit["ad_id"]:
                raise ValueError(
                    f"Position {unit['position']} should be advertisement "
                    f"{unit['ad_id']} but the candidate file has {actual}."
                )
            texts.append(truncate_text(candidates[unit["position"]]["text"],
                                       budget, strategy))
        return texts

    test_units = saved["folds"][str(fold_number)]

    train_units = []
    for number, units in saved["folds"].items():
        if number != str(fold_number):
            train_units += units

    return {
        "test_texts": to_texts(test_units),
        "test_labels": [u["label"] for u in test_units],
        "test_units": test_units,
        "train_texts": to_texts(train_units),
        "train_labels": [u["label"] for u in train_units],
        "n_folds": saved["n_folds"],
    }
