"""Score predictions the same way for every method.

All three methods go through these functions, so the numbers are comparable by
construction rather than by me remembering to compute them the same way.

What gets reported, and why each one is here:

  macro F1          The headline. Macro means every label counts equally, which
                    matters because the nine classes are equal in size here but
                    the real world is not, and because a method could otherwise
                    look good by being right on the easy labels only.

  per-class F1      Shows which occupations a method struggles with. Expect the
                    IT labels to be hardest.

  confusion matrix  Shows WHAT a method confuses, not just how often.

  within-family vs  The point of the 3x3 design. Confusing 2511 with 2512 is a
  across-family     fine distinction between two software roles. Confusing 2511
                    with 5321 is an undersköterska mistaken for an IT architect.
                    Both count as one error in accuracy, but they are not the
                    same kind of failure, and a good method should make mostly
                    the first kind.

  majority baseline With nine equal classes this is 1/9 = 11.1%. It is here so
                    that any score has something to be compared against.
"""

import json
import random
from collections import Counter, defaultdict

from scipy.stats import chi2
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Which family each occupation belongs to. Used only for error analysis, never
# as a label the models predict.
FAMILIES = {
    "2511": "IT", "2512": "IT", "2514": "IT",
    "2341": "Education", "2330": "Education", "5312": "Education",
    "2221": "Healthcare", "5321": "Healthcare", "5323": "Healthcare",
}

NAMES = {
    "2511": "Systemanalytiker/IT-arkitekter",
    "2512": "Mjukvaru-/systemutvecklare",
    "2514": "Systemtestare/testledare",
    "2341": "Grundskollärare",
    "2330": "Gymnasielärare",
    "5312": "Elevassistenter",
    "2221": "Grundutbildade sjuksköterskor",
    "5321": "Undersköterskor, hemtjänst",
    "5323": "Undersköterskor, vårdavdelning",
}


def family_errors(true_labels, predicted_labels):
    """Split the mistakes into fine confusions and gross ones.

    A within-family error means the method picked the wrong occupation but the
    right broad area. An across-family error means it was not even close.

    Returns the counts and the two rates as shares of all predictions.
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
    """Accuracy you get by always guessing the most common label.

    With nine equal classes this is about 11.1%. Any method that cannot beat it
    has learned nothing.
    """
    most_common = Counter(true_labels).most_common(1)[0][1]

    return most_common / len(true_labels)


def macro_f1(true_labels, predicted_labels):
    """Macro F1 for one set of predictions.

    "Macro" means each of the nine labels counts the same, no matter how many
    advertisements it has. The label list is taken from the true labels and
    passed in explicitly, so a label that never gets predicted still counts as
    a zero instead of quietly disappearing.
    """
    labels = sorted(set(true_labels))

    return f1_score(true_labels, predicted_labels, labels=labels,
                    average="macro", zero_division=0)


def group_by_employer(units):
    """Return {employer: [positions]}, one entry per employer.

    Same shape as nest_counts and group_positions_by_band: walk the list once
    and collect positions under a key. The positions are indexes into the
    true-label and prediction lists, so once we have picked some employers we
    know exactly which predictions belong to them.
    """
    groups = defaultdict(list)

    for position, unit in enumerate(units):
        groups[unit["employer"]].append(position)

    # Plain dict on the way out: a defaultdict would silently invent an
    # employer if anything later looked one up that is not there.
    return dict(groups)


def draw_employers(groups, generator):
    """Pick employers at random, with replacement, and return their positions.

    "With replacement" means the same employer can be picked more than once,
    and some are not picked at all. That is what makes it a bootstrap: each
    draw is a slightly different version of the same test set.

    We pick as many employers as there are in total, so each draw is about the
    same size as the real test set.
    """
    names = sorted(groups)
    positions = []

    for _ in names:
        picked = generator.choice(names)
        positions += groups[picked]

    return positions


def bootstrap_macro_f1(units, true_labels, predicted_labels,
                       n_rounds=2000, seed=11):
    """Macro F1 with a 95% confidence interval, by resampling employers.

    What a confidence interval is for. A single score like 0.769 looks exact,
    but it came from one particular test set. If we had drawn a different test
    set we would have got a slightly different number. The interval says how
    much that number could reasonably move, so you know whether a gap between
    two methods is real or just noise.

    Why we resample EMPLOYERS and not advertisements. The test set has 2,277
    advertisements but only 432 employers, and label 5321 has just 22. Several
    advertisements from the same employer are written by the same person in the
    same style, so they are not 2,277 independent facts. Resampling
    advertisements would pretend they were, and the interval would come out far
    too narrow.

    How it works, step by step:

      1. Group the test advertisements by employer.
      2. Pick 432 employers at random, with replacement.
      3. Score the advertisements those employers brought.
      4. Do that 2,000 times, which gives 2,000 slightly different scores.
      5. Sort them and cut off the lowest 2.5% and the highest 2.5%. What is
         left is the middle 95%, and that is the interval.
    """
    groups = group_by_employer(units)
    generator = random.Random(seed)

    scores = []

    for _ in range(n_rounds):
        positions = draw_employers(groups, generator)
        drawn_true = [true_labels[p] for p in positions]
        drawn_predicted = [predicted_labels[p] for p in positions]
        scores.append(macro_f1(drawn_true, drawn_predicted))

    scores.sort()
    # Cut 2.5% off each end. What remains is the middle 95%.
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
    """Count the advertisements where exactly one of the two methods is right.

    Returns (only_a, only_b). Cases where both are right, or both wrong, are
    ignored on purpose: they say nothing about which method is better.
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
    """Probability of seeing this lopsided a result if the methods were equal.

    This is McNemar's test. The idea is simple: if two methods were equally
    good, then among the advertisements where exactly one of them is right, it
    should be method A about half the time. If A wins 80 and B wins 40, that is
    lopsided, and the test says how surprising that is.

    A small p-value means the difference is unlikely to be chance. It says
    nothing about how BIG the difference is, which is why the bootstrap
    interval is reported next to it.

    Returns "not a number" when there are too few disagreements to test.
    """
    total = only_a + only_b

    if total < 10:
        return float("nan")

    # The -1 is a continuity correction, standard for this test.
    statistic = (abs(only_a - only_b) - 1) ** 2 / total

    # sf is the survival function: the chance of a value this large or larger.
    # One degree of freedom, because there is one comparison.
    return chi2.sf(statistic, 1)


def compare_methods(units, true_labels, predictions_a, predictions_b,
                    name_a="A", name_b="B", n_rounds=2000, seed=12):
    """Is method A actually better than method B, or is the gap just noise?

    Reports three things:

      only_a / only_b   how many advertisements each method alone got right
      mcnemar_p         how surprising that split would be if they were equal
      difference + CI   how big the macro F1 gap is, and how much it could move

    The field to look at is "established". It is True only when the confidence
    interval stays entirely on one side of zero. If the interval includes zero,
    the honest statement is "A scored higher" and NOT "A is better".

    The interval is built the same way as in bootstrap_macro_f1: by drawing
    employers with replacement, because employers are the real unit here.
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
        # True only when the interval does not cross zero.
        "established": low > 0 or high < 0,
    }


def save_predictions(units, true_labels, predicted_labels, path, method, split):
    """Write one row per advertisement: what it was, what the method said.

    Every method must write this file. Two analyses depend on it and neither is
    possible from aggregate scores alone:

      - whether the three methods fail on the SAME advertisements, which is how
        we tell a shared limit in the data from genuine differences between
        methods
      - whether the errors sit in the off-type occupations inside a label, for
        example the 15.3% of label 2512 that is Projektledare, IT rather than a
        developer

    The ad_id makes each row joinable back to the raw advertisement, so the
    occupation title can be looked up later without re-running anything.
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

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump({"method": method, "split": split, "n": len(rows),
                   "predictions": rows}, file, ensure_ascii=False, indent=2)

    return path


def report(true_labels, predicted_labels, title="results"):
    """Print the full scorecard and return the numbers as a dictionary."""
    labels = sorted(set(true_labels))

    macro = f1_score(true_labels, predicted_labels, labels=labels, average="macro",
                     zero_division=0)
    accuracy = sum(t == p for t, p in zip(true_labels, predicted_labels)) / len(true_labels)
    families = family_errors(true_labels, predicted_labels)

    print(f"=== {title} ===")
    print(f"  n                  {len(true_labels):,}")
    print(f"  accuracy           {accuracy:.3f}")
    print(f"  macro F1           {macro:.3f}")
    print(f"  majority baseline  {majority_baseline(true_labels):.3f}")
    print()
    print(f"  within-family errors   {families['within_family_errors']:>5,}"
          f"  ({families['within_family_rate']:.1%})   fine confusions")
    print(f"  across-family errors   {families['across_family_errors']:>5,}"
          f"  ({families['across_family_rate']:.1%})   gross confusions")
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
