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

from collections import Counter

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
