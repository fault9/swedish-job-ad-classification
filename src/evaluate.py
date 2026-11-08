"""Score predictions the same way for every method.

All methods go through these functions, so their scores are comparable. The
headline metric is macro F1. Confidence intervals and paired comparisons
resample employers, because advertisements from one employer share a template
and their errors are not independent.
"""

import random
from collections import Counter, defaultdict

from scipy.stats import chi2
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from files import save_json

# Used only for error analysis, never as a label or a feature.
FAMILIES = {
    "2511": "IT", "2512": "IT", "2514": "IT",
    "2341": "Education", "2330": "Education", "5312": "Education",
    "2221": "Healthcare", "5321": "Healthcare", "5323": "Healthcare",
}


def family_errors(true_labels, predicted_labels):
    """Split the errors into within-family and across-family errors.

    Args:
        true_labels: The true label of each advertisement.
        predicted_labels: The predicted label of each advertisement.

    Returns:
        A dict with the counts and the two error rates.
    """
    total = len(true_labels)
    within = 0
    across = 0

    for true_label, predicted in zip(true_labels, predicted_labels):
        if true_label == predicted:
            continue
        if FAMILIES[true_label] == FAMILIES.get(predicted):
            within += 1
        else:
            across += 1

    return {
        "n": total,
        "correct": total - within - across,
        "within_family_errors": within,
        "across_family_errors": across,
        "within_family_rate": within / total,
        "across_family_rate": across / total,
    }


def majority_baseline(true_labels):
    """Return the accuracy of always guessing the most common label.

    Args:
        true_labels: The true label of each advertisement.

    Returns:
        The share of advertisements with the most common label.
    """
    most_common = Counter(true_labels).most_common(1)[0][1]

    return most_common / len(true_labels)


def macro_f1(true_labels, predicted_labels):
    """Return macro F1, with every label weighted equally.

    A label that is never predicted counts as zero.

    Args:
        true_labels: The true label of each advertisement.
        predicted_labels: The predicted label of each advertisement.

    Returns:
        The macro F1 score.
    """
    labels = sorted(set(true_labels))

    return f1_score(true_labels, predicted_labels, labels=labels,
                    average="macro", zero_division=0)


def group_by_employer(units):
    """Group advertisement positions by employer.

    Args:
        units: One dict per advertisement, each with an "employer" key.

    Returns:
        A dict mapping each employer to the positions of its advertisements.
    """
    groups = defaultdict(list)

    for position, unit in enumerate(units):
        groups[unit["employer"]].append(position)

    return dict(groups)


def draw_employers(groups, generator):
    """Draw employers with replacement and return their positions.

    As many employers are drawn as there are in total, so each draw is about
    the size of the real test set.

    Args:
        groups: Positions per employer, from group_by_employer.
        generator: A random.Random instance.

    Returns:
        The positions of all advertisements from the drawn employers.
    """
    names = sorted(groups)
    positions = []

    for _ in names:
        picked = generator.choice(names)
        positions += groups[picked]

    return positions


def bootstrap_macro_f1(units, true_labels, predicted_labels,
                       n_rounds=2000, seed=11):
    """Return macro F1 with a 95% confidence interval from resampling employers.

    Args:
        units: One dict per advertisement, each with an "employer" key.
        true_labels: The true label of each advertisement.
        predicted_labels: The predicted label of each advertisement.
        n_rounds: How many resamples to draw.
        seed: Seed for the random generator.

    Returns:
        A dict with the macro F1, the interval and the sample sizes.
    """
    groups = group_by_employer(units)
    generator = random.Random(seed)

    scores = []

    for _ in range(n_rounds):
        positions = draw_employers(groups, generator)
        drawn_true = [true_labels[p] for p in positions]
        drawn_predicted = [predicted_labels[p] for p in positions]
        scores.append(macro_f1(drawn_true, drawn_predicted))

    # The middle 95% of the scores is the interval.
    scores.sort()
    low = scores[int(0.025 * n_rounds)]
    high = scores[int(0.975 * n_rounds)]

    return {
        "macro_f1": macro_f1(true_labels, predicted_labels),
        "ci_low": low,
        "ci_high": high,
        "n_employers": len(groups),
        "n_advertisements": len(units),
        "n_rounds": n_rounds,
        "seed": seed,
    }


def count_disagreements(true_labels, predictions_a, predictions_b):
    """Count the advertisements where exactly one of two methods is right.

    Args:
        true_labels: The true label of each advertisement.
        predictions_a: The predictions of method A.
        predictions_b: The predictions of method B.

    Returns:
        Two counts: only A right, and only B right.
    """
    only_a = 0
    only_b = 0

    for true_label, a, b in zip(true_labels, predictions_a, predictions_b):
        if a == true_label and b != true_label:
            only_a += 1
        elif b == true_label and a != true_label:
            only_b += 1

    return only_a, only_b


def mcnemar_p_value(only_a, only_b):
    """Return the McNemar p-value, with continuity correction.

    Args:
        only_a: Advertisements only method A got right.
        only_b: Advertisements only method B got right.

    Returns:
        The p-value, or nan if there are fewer than 10 disagreements.
    """
    total = only_a + only_b

    if total < 10:
        return float("nan")

    statistic = (abs(only_a - only_b) - 1) ** 2 / total

    return chi2.sf(statistic, 1)


def compare_methods(units, true_labels, predictions_a, predictions_b,
                    name_a="A", name_b="B", n_rounds=2000, seed=12):
    """Compare two methods on the same advertisements.

    The difference in macro F1 gets a 95% interval from resampling employers.
    Both methods are scored on the same resample each round. The difference is
    established only if the interval does not include zero.

    Args:
        units: One dict per advertisement, each with an "employer" key.
        true_labels: The true label of each advertisement.
        predictions_a: The predictions of method A.
        predictions_b: The predictions of method B.
        name_a: Name of method A.
        name_b: Name of method B.
        n_rounds: How many resamples to draw.
        seed: Seed for the random generator.

    Returns:
        A dict with the disagreement counts, the McNemar p-value, the
        difference in macro F1 (A minus B) and its interval.
    """
    only_a, only_b = count_disagreements(true_labels, predictions_a, predictions_b)

    groups = group_by_employer(units)
    generator = random.Random(seed)

    differences = []

    for _ in range(n_rounds):
        positions = draw_employers(groups, generator)
        drawn_true = [true_labels[p] for p in positions]
        score_a = macro_f1(drawn_true, [predictions_a[p] for p in positions])
        score_b = macro_f1(drawn_true, [predictions_b[p] for p in positions])
        differences.append(score_a - score_b)

    differences.sort()
    low = differences[int(0.025 * n_rounds)]
    high = differences[int(0.975 * n_rounds)]

    observed = (macro_f1(true_labels, predictions_a)
                - macro_f1(true_labels, predictions_b))

    return {
        "name_a": name_a,
        "name_b": name_b,
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "mcnemar_p": mcnemar_p_value(only_a, only_b),
        "macro_f1_difference": observed,
        "ci_low": low,
        "ci_high": high,
        "established": low > 0 or high < 0,
    }


def save_predictions(units, true_labels, predicted_labels, path, method, split):
    """Save one row per advertisement with its true and predicted label.

    The error overlap and the paired comparisons are computed from these files.

    Args:
        units: One dict per advertisement, from the split or fold.
        true_labels: The true label of each advertisement.
        predicted_labels: The predicted label of each advertisement.
        path: Where to save the file.
        method: Name of the method, for example "kbbert".
        split: Name of the split, for example "fold0".

    Returns:
        The path the file was saved to.
    """
    rows = [
        {
            "ad_id": unit["ad_id"],
            "cluster": unit["cluster"],
            "employer": unit["employer"],
            "true": true_label,
            "predicted": predicted,
            "correct": true_label == predicted,
        }
        for unit, true_label, predicted in zip(units, true_labels, predicted_labels)
    ]

    save_json({"method": method, "split": split, "n": len(rows),
               "predictions": rows}, path)

    return path


def report(true_labels, predicted_labels, title="results"):
    """Print the scores and return them as a dict.

    Args:
        true_labels: The true label of each advertisement.
        predicted_labels: The predicted label of each advertisement.
        title: Heading for the printout.

    Returns:
        A dict with accuracy, macro F1, family errors, per-class scores and
        the confusion matrix.
    """
    labels = sorted(set(true_labels))

    macro = macro_f1(true_labels, predicted_labels)
    accuracy = sum(t == p for t, p in zip(true_labels, predicted_labels)) / len(true_labels)
    families = family_errors(true_labels, predicted_labels)

    print(f"=== {title} ===")
    print(f"  n                  {len(true_labels):,}")
    print(f"  accuracy           {accuracy:.3f}")
    print(f"  macro F1           {macro:.3f}")
    print(f"  majority baseline  {majority_baseline(true_labels):.3f}")
    print()
    print(f"  within-family errors   {families['within_family_errors']:>5,}"
          f"  ({families['within_family_rate']:.1%})")
    print(f"  across-family errors   {families['across_family_errors']:>5,}"
          f"  ({families['across_family_rate']:.1%})")
    print()

    print(classification_report(true_labels, predicted_labels, labels=labels,
                                digits=3, zero_division=0))

    print("  confusion matrix, rows are true labels:")
    matrix = confusion_matrix(true_labels, predicted_labels, labels=labels)
    print("          " + "".join(f"{l:>7}" for l in labels))
    for label, row in zip(labels, matrix):
        print(f"  {label:7}" + "".join(f"{n:>7}" for n in row))

    return {
        "n": len(true_labels),
        "accuracy": accuracy,
        "macro_f1": macro,
        "majority_baseline": majority_baseline(true_labels),
        "families": families,
        "per_class": classification_report(true_labels, predicted_labels,
                                           labels=labels, output_dict=True,
                                           zero_division=0),
        "confusion_matrix": matrix.tolist(),
        "labels": labels,
    }
