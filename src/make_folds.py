"""Build employer-disjoint folds for cross-validation.

Why bother, when splits_*.json already exists. The single test split holds 253
advertisements per label, and because they come from only 317 employers the
honest confidence interval on macro F1 is about plus or minus 3.8 points. Two
methods would have to differ by roughly 5 points before the difference could be
called established, and they probably will not.

Cross-validation fixes that without collecting any more data. Every
advertisement takes a turn in the test fold, so the estimate rests on 1,603 per
label instead of 253 — about 14,400 test predictions instead of 2,277. That
narrows the interval by roughly a factor of 2.5.

The rules are the same as for the single split, with one addition:

  one text per cluster       near-duplicate reposts are not many examples
  single-label clusters only
  Swedish-dominant, 50+ words
  max CAP per employer per label
  employers never split across folds FOR THE SAME LABEL, so a model never
                             sees an employer under a label in training and
                             then meets that same pairing in testing
  equal label sizes in every fold
  NEW: pilot advertisements are dropped entirely

WHAT "EMPLOYER-DISJOINT" MEANS HERE, PRECISELY. The unit is the pairing of an
employer with a label, not the employer alone. Chosen deliberately on
2026-10-16, and the reason is Swedish public-sector hiring: 1,102 of 2,860
employers (39%) advertise under more than one of the nine labels, and the
largest are municipalities posting across all nine from one house template.

    2321000131   50 ads across 2221 2330 2341 2511 2512 2514 5312 5321 5323
    2120000142   46 ads across all nine
    2120001967   42 ads across all nine

81% of the 14,400 advertisements belong to such an employer. So a model trained
on four folds may have seen Göteborg's template under 2341 and then meet
Göteborg under 2512 in the test fold.

That is not the leakage that inflates scores, which is seeing the SAME
employer-label pairing on both sides, and that is excluded. The cross-label
case could help, by signalling "this is a municipality", or hurt, by having
learned to attach that template to the wrong label. The direction is unknown
and measurable afterwards: compare accuracy on advertisements from multi-label
employers against single-label ones.

The alternative — one employer in exactly one fold regardless of label — gives
a cleaner sentence but makes balanced folds impossible, because a 50-ad
municipality spanning nine labels has to land somewhere whole. Label imbalance
would confound the method comparison, which is a known problem traded for an
unknown one. Hence this choice.

That last one is forced by the design. In cross-validation every advertisement
ends up in a test fold once, and the label-selection pilot advertisements are
not allowed in any test set, so there is nowhere to put them. Dropping them
costs 84 units on the binding label, about 5%, which is cheap.

ON HYPERPARAMETERS. These folds are for measuring, not for tuning. The
settings come from the search already done on the frozen train/validation
split, and they stay fixed across all folds. Tuning inside each fold as well
would need nested cross-validation and five times the runs.

This does mean the hyperparameters were chosen using 253 advertisements per
label that later appear in test folds, so the estimate is very slightly
optimistic. That is disclosed rather than hidden, and the alternative — a
separate tuning set carved out of an already small corpus — would cost more
than it buys.
"""

import random
from collections import Counter, defaultdict

from files import load_json, save_json
from make_splits import (CAP, LANGUAGE, MIN_WORDS, cap_per_employer,
                         eligible_positions, one_unit_per_cluster,
                         pilot_clusters, pure_clusters)
from paths import (CLUSTERS_FILE, DATA_DIR, EMPLOYERS_FILE, FOLDS_FILE,
                   LANGUAGE_FILE, PERIOD)

N_FOLDS = 5
SEED = 20261016


def balance_labels(units, target, generator):
    """Cut every label down to target units, so all nine classes are equal.

    Same decision as in make_splits.py: the classes must be the same size, or
    any difference measured between methods is partly a difference in how much
    each method saw of each class.

    The target is the smallest label, so this throws away a lot of 2512 and
    2341. That is the price of a clean comparison, and the discarded data is
    still in the corpus if a later analysis wants it.
    """
    by_label = defaultdict(list)

    for unit in units:
        by_label[unit["label"]].append(unit)

    kept = []

    for label in sorted(by_label):
        available = by_label[label]
        kept += generator.sample(available, min(target, len(available)))

    return kept


def assign_folds(units, n_folds, generator):
    """Spread employers across folds, keeping each label evenly represented.

    The constraint is that an employer must land in exactly one fold. That
    makes perfect balance impossible, because employers come in different
    sizes, so this does the standard greedy thing: handle the biggest
    employers first and always give the next one to whichever fold currently
    has the fewest advertisements for that label.

    Biggest first matters. Placing a 20-advertisement employer last would
    leave one fold 20 ahead of the others with no way to even it out.

    Returns {fold number: [units]}.
    """
    folds = {number: [] for number in range(n_folds)}

    by_label = defaultdict(lambda: defaultdict(list))

    for unit in units:
        by_label[unit["label"]][unit["employer"]].append(unit)

    for label in sorted(by_label):
        employers = sorted(by_label[label])
        # Shuffle first so that employers of equal size are not always ordered
        # the same way, then sort by size so the big ones go first.
        generator.shuffle(employers)
        employers.sort(key=lambda name: len(by_label[label][name]), reverse=True)

        # How many of this label each fold holds so far.
        held = {number: 0 for number in range(n_folds)}

        for employer in employers:
            group = by_label[label][employer]
            # min() on the count picks the emptiest fold; ties go to the
            # lowest fold number, which keeps the result repeatable.
            emptiest = min(held, key=lambda number: held[number])
            folds[emptiest] += group
            held[emptiest] += len(group)

    return folds


def trim_folds(folds):
    """Cut every label down to the smallest count any fold reached.

    Employers move as whole blocks, so the greedy assignment leaves folds
    slightly uneven. Trimming makes every fold hold the same number per label,
    which means the per-fold scores are directly comparable and the macro
    average is not quietly weighted.
    """
    labels = sorted({unit["label"] for fold in folds.values() for unit in fold})

    smallest = {}

    for label in labels:
        counts = [sum(1 for unit in fold if unit["label"] == label)
                  for fold in folds.values()]
        smallest[label] = min(counts)

    trimmed = {}

    for number, fold in folds.items():
        kept = []
        for label in labels:
            same = [unit for unit in fold if unit["label"] == label]
            kept += same[:smallest[label]]
        trimmed[number] = kept

    return trimmed


def check_no_employer_spans_folds(folds):
    """Count how often the disjointness rule is broken, and how often it is not.

    Returns two numbers:

      pairs_spanning   employer-label pairings in more than one fold. MUST be
                       zero: this is the leakage that would inflate scores.
      employers_spanning
                       employers in more than one fold under DIFFERENT labels.
                       Expected to be large, roughly a third, because Swedish
                       municipalities advertise across all nine labels. Not
                       leakage, but worth printing so nobody mistakes the zero
                       above for a stronger guarantee than it is.
    """
    pair_folds = defaultdict(set)
    employer_folds = defaultdict(set)

    for number, fold in folds.items():
        for unit in fold:
            pair_folds[(unit["label"], unit["employer"])].add(number)
            employer_folds[unit["employer"]].add(number)

    return (
        sum(1 for numbers in pair_folds.values() if len(numbers) > 1),
        sum(1 for numbers in employer_folds.values() if len(numbers) > 1),
    )


def report(folds):
    """Print the size of every fold by label, and the employer check."""
    labels = sorted({unit["label"] for fold in folds.values() for unit in fold})
    numbers = sorted(folds)

    print(f"{'label':8}" + "".join(f"{'fold ' + str(n):>9}" for n in numbers)
          + f"{'total':>9}")

    for label in labels:
        row = [sum(1 for unit in folds[n] if unit["label"] == label) for n in numbers]
        print(f"{label:8}" + "".join(f"{v:>9,}" for v in row) + f"{sum(row):>9,}")

    print(f"{'TOTAL':8}" + "".join(f"{len(folds[n]):>9,}" for n in numbers)
          + f"{sum(len(folds[n]) for n in numbers):>9,}")
    print()

    for number in numbers:
        employers = len({(u["label"], u["employer"]) for u in folds[number]})
        print(f"  fold {number}: {len(folds[number]):>6,} advertisements "
              f"from {employers:>5,} label-employer pairs")

    pairs_spanning, employers_spanning = check_no_employer_spans_folds(folds)
    all_employers = len({u["employer"] for fold in folds.values() for u in fold})

    print()
    print(f"employer-label pairings in more than one fold: {pairs_spanning}"
          f"   <- must be 0, this is the leakage that matters")
    print(f"employers in more than one fold under different labels: "
          f"{employers_spanning:,} of {all_employers:,}"
          f"   <- expected, municipalities advertise across all nine labels")


def main():
    annotations = load_json(LANGUAGE_FILE)
    employers = load_json(EMPLOYERS_FILE)
    clusters = load_json(CLUSTERS_FILE)

    threshold = str(clusters["primary_threshold"])
    cluster_ids = clusters["cluster_ids"][threshold]

    print(f"folds: {N_FOLDS}   employer cap: {CAP}   threshold: {threshold}")
    print()

    generator = random.Random(SEED)

    swedish = eligible_positions(annotations, swedish=True)
    keep = pure_clusters(swedish, annotations, cluster_ids)
    units = one_unit_per_cluster(swedish, annotations, cluster_ids, employers, keep)
    units = cap_per_employer(units, CAP, generator)

    print(f"Swedish single-label clusters after the cap: {len(units):,}")

    # Every advertisement becomes a test item in one fold, and pilot
    # advertisements may not be test items, so they have to go.
    blocked = pilot_clusters(DATA_DIR, annotations, cluster_ids)
    before = len(units)
    units = [unit for unit in units if unit["cluster"] not in blocked]
    print(f"after dropping {len(blocked):,} label-selection pilot clusters: "
          f"{len(units):,}  (lost {before - len(units):,})")

    per_label = Counter(unit["label"] for unit in units)
    target = min(per_label.values())
    print(f"smallest label: {target:,}  ({min(per_label, key=per_label.get)})")

    units = balance_labels(units, target, generator)
    print(f"after balancing all labels to {target:,}: {len(units):,} units")
    print()

    folds = assign_folds(units, N_FOLDS, generator)
    folds = trim_folds(folds)
    report(folds)

    save_json(
        {
            "period": PERIOD,
            "n_folds": N_FOLDS,
            "seed": SEED,
            "employer_cap": CAP,
            "min_words": MIN_WORDS,
            "language": LANGUAGE,
            "clustering_threshold": float(threshold),
            "candidate_count": len(annotations),
            "pilot_clusters_dropped": len(blocked),
            "per_label_per_fold": min(
                sum(1 for u in folds[0] if u["label"] == label)
                for label in sorted(per_label)
            ),
            "folds": {
                str(number): [
                    {k: unit[k] for k in
                     ("position", "ad_id", "cluster", "label", "employer")}
                    for unit in fold
                ]
                for number, fold in folds.items()
            },
        },
        FOLDS_FILE,
    )

    print()
    print("Folds saved:", FOLDS_FILE)
    print("Hyperparameters come from the frozen split search. Do not tune here.")


if __name__ == "__main__":
    main()
