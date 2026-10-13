"""Build the frozen train, validation and test splits.

This is the last preprocessing step. It picks which advertisements go into
training, which into validation and which into testing, and then saves those
choices to disk so that every later run uses exactly the same data.

Seven rules are applied, and each one removes a way the results could be wrong:

1. One text per cluster. Employers repost the same advertisement many times.
   One template family here has 6,813 members. Without this rule that one
   template would become 6,813 training examples and the model would mostly
   learn it.

2. Drop clusters that carry more than one label. If two near-identical texts
   were filed under different occupation codes, no method can get both right,
   so the example only adds noise.

3. Swedish must be the dominant language. "Dominant" means Swedish has the most
   words, so an advertisement that is 70% Swedish and 30% English is kept. Only
   advertisements where Swedish is the minority are excluded. About 99% of the
   training texts contain some non-Swedish text, which is normal for Swedish
   job advertisements.

4. At least 50 words. Shorter texts turned out to be fragments of requirement
   lists with no job title, not real advertisements.

5. At most CAP clusters per employer, per label. One employer held 240 of
   label 2514's clusters, and the ten largest held 40% of it. Without a cap a
   few employers would dominate a whole class.

6. No employer may appear in two splits. Otherwise a model could score well by
   recognising an employer's house style instead of understanding the
   occupation. This matters for comparing methods, not just for the accuracy
   number, because recognising boilerplate is what a bag-of-words model is best
   at, so leaving it in would flatter TF-IDF.

7. Every label gets the same number of texts. Class imbalance would otherwise
   be mixed into any difference we measure between methods.

It also writes a second, smaller test set of advertisements where Swedish is
NOT the dominant language, for the three IT labels only. That set is for
inference only, never training. The other six labels are left out because
Swedish teaching and care work is almost never advertised in English: label
5323 has eight such advertisements in total.

Run this once. The output is frozen. Never tune anything against the test sets.
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from inspect_data import save_json

from paths import PROJECT_DIR

CAP = 20                  # most clusters any one employer may contribute to a label
MIN_WORDS = 50            # shorter texts are fragments, not advertisements
LANGUAGE = "SWEDISH"
PROPORTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
IT_LABELS = ["2511", "2512", "2514"]
SEED = 20260413           # fixed so the same splits come out every time


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def check_alignment(splits, candidates):
    """Make sure the saved positions still point at the right advertisements.

    The splits store both a position and an advertisement id for every text.
    The position is only valid for the candidate file the splits were built
    from. If that file is rebuilt, positions shift and would quietly select
    the wrong advertisements, with no error anywhere.

    So before using the splits, check a few of them: does the candidate at
    that position still have the advertisement id we recorded? Raises if not,
    which is the whole point. A loud failure beats a silent wrong answer.

    Call this from any script that reads splits_<period>.json.
    """

    if len(candidates) != splits["candidate_count"]:
        raise ValueError(
            f"Candidate file has {len(candidates):,} records but the splits "
            f"were built from {splits['candidate_count']:,}. The candidate file "
            f"has been rebuilt, so the saved positions are no longer valid."
        )

    groups = list(splits["splits"].values()) + [splits["test_nonsv"]]

    for units in groups:
        for unit in units:
            recorded = unit["ad_id"]
            actual = candidates[unit["position"]]["source_ad_ids"][0]
            if recorded != actual:
                raise ValueError(
                    f"Position {unit['position']} should be advertisement "
                    f"{recorded} but the candidate file has {actual}."
                )

    return True


def eligible_positions(annotations, swedish):
    """Find the texts we are allowed to use, as positions in the candidate list.

    A "position" is just the index of a text in candidates_<period>.json. The
    language annotations and the cluster ids are stored in the same order, so
    position 57 means the same advertisement in all three files.

    Set swedish=True to get the texts where Swedish is the dominant language,
    which become train, validation and test. Set swedish=False to get the rest,
    which become the second test set. The two groups never overlap.
    """

    wanted = []

    for position, annotation in enumerate(annotations):
        if annotation["n_words"] < MIN_WORDS:
            continue
        # True when Swedish has the most words in this advertisement.
        is_swedish = annotation["dominant_language"] == LANGUAGE
        if is_swedish == swedish:
            wanted.append(position)

    return wanted


def pure_clusters(positions, annotations, cluster_ids):
    """Cluster ids whose eligible texts all carry the same label."""

    found = defaultdict(set)

    for position in positions:
        # Collect every label seen inside each cluster.
        found[cluster_ids[position]].add(annotations[position]["label"])

    # Exactly one label means the cluster is safe to use.
    return {cluster for cluster, labels in found.items() if len(labels) == 1}


def one_unit_per_cluster(positions, annotations, cluster_ids, employers, keep):
    """Reduce each cluster of near-duplicates down to a single example.

    A cluster is a group of advertisements that MinHash found to be near
    copies of each other. We only want one of them, otherwise a template that
    an employer reposted 240 times would become 240 training examples.

    Returns a list of small dictionaries, one per cluster. Each is called a
    "unit" and holds what later steps need: the label, the employer, the
    cluster id, the position of the text we picked, and its advertisement id.

    Two ways of pointing at the same text are stored on purpose:

      position  fast, but only valid for THIS candidates file. If the candidate
                file is ever rebuilt with different labels or years, every
                position shifts and would silently point at the wrong text.
      ad_id     slow to look up, but stable forever. Lets a later script check
                that the position still refers to the advertisement we meant.

    Which member of a cluster gets picked is whichever comes first in the
    candidate file. They are near-identical by definition, so it does not
    matter much, and going in file order keeps the result repeatable.
    """

    units = []
    seen = set()

    for position in positions:
        cluster = cluster_ids[position]
        # Skip if we already took this cluster, or if it had mixed labels.
        if cluster in seen or cluster not in keep:
            continue
        seen.add(cluster)
        units.append({
            "label": annotations[position]["label"],
            "employer": employers[position],
            "cluster": cluster,
            "position": position,
            # First source id, so the text can be found again without positions.
            "ad_id": annotations[position]["source_ad_ids"][0],
        })

    return units


def cap_per_employer(units, cap, generator):
    """Limit how many advertisements any single employer contributes to a label.

    Without this, a few big employers dominate a class. Label 2514 had one
    employer holding 240 of its 2,698 clusters, and the ten largest held 40%.
    A model could then score well on 2514 by learning those employers rather
    than the occupation.

    Capping costs data: at cap 20 the usable total drops from 53,412 to
    roughly half. We accept that because it buys a test set that is not
    dominated by one employer's writing style.
    """

    groups = defaultdict(list)

    for unit in units:
        # Group by label AND employer, since an employer can post in several labels.
        groups[(unit["label"], unit["employer"])].append(unit)

    kept = []

    for key in sorted(groups):
        group = groups[key]
        if len(group) <= cap:
            kept += group
        else:
            # Pick at random so we do not always keep the same reposts.
            kept += generator.sample(group, cap)

    return kept


def split_employers(units, target_per_label, generator):
    """Divide the data into train, validation and test without splitting employers.

    The important idea: employers move as whole blocks. If an employer has 12
    advertisements, all 12 go to the same split. That way a model never sees
    an employer in training and then meets the same employer in testing, so it
    cannot score by recognising a company's house style.

    This is done separately for each label. An employer that advertises both
    nursing and IT jobs may land in train for one label and test for the
    other, which is fine, because the leakage we care about is within a label.

    target_per_label is how many advertisements each label should end up with
    in total across the three splits.
    """

    assignment = {name: [] for name in PROPORTIONS}

    by_label = defaultdict(lambda: defaultdict(list))

    for unit in units:
        by_label[unit["label"]][unit["employer"]].append(unit)

    assignment = {name: [] for name in PROPORTIONS}

    by_label = defaultdict(lambda: defaultdict(list))

    for unit in units:
        by_label[unit["label"]][unit["employer"]].append(unit)

    for label in sorted(by_label):
        employers = sorted(by_label[label])
        # Shuffle so which employers land in test is not decided by their id.
        generator.shuffle(employers)

        # How many advertisements each split should get for this label.
        wanted = {
            "test": int(target_per_label * PROPORTIONS["test"]),
            "val": int(target_per_label * PROPORTIONS["val"]),
        }
        wanted["train"] = target_per_label - wanted["test"] - wanted["val"]

        filled = {name: [] for name in wanted}

        # Fill test first, then val, then train. Test and val are small, so
        # filling them first makes their targets easier to hit exactly.
        for name in ("test", "val", "train"):
            for employer in list(employers):
                if len(filled[name]) >= wanted[name]:
                    break
                filled[name] += by_label[label][employer]
                # Remove it so no later split can take the same employer.
                employers.remove(employer)
            # The last employer added usually overshoots, so cut the extra off.
            filled[name] = filled[name][:wanted[name]]

        for name in filled:
            assignment[name] += filled[name]

    return assignment


def trim_to_equal(assignment, name_order):
    """Make every label exactly the same size in every split.

    Why this is needed: employers move as whole blocks, so a label can run
    short. Label 2514 has the fewest employers and ended up 11 advertisements
    behind the others in training. Eleven out of 1,181 would not change any
    result, but "all classes are the same size" is easier to state and to
    defend than "all classes are the same size except one".

    So this finds the smallest a label reached and cuts all the others down to
    match. It throws away a handful of advertisements on purpose.
    """

    labels = sorted({u["label"] for name in assignment for u in assignment[name]})

    for name in name_order:
        # The smallest any label reached in this split.
        smallest = min(
            sum(1 for u in assignment[name] if u["label"] == label)
            for label in labels
        )
        trimmed = []
        for label in labels:
            same = [u for u in assignment[name] if u["label"] == label]
            trimmed += same[:smallest]
        assignment[name] = trimmed

    return assignment


def report(assignment, name_order):
    """Print the split sizes, and check that no employer appears in two splits.

    The overlap count at the end is the important line. It must be zero. If it
    is not, the employer-disjoint rule is broken somewhere and the test scores
    would be optimistic.
    """

    labels = sorted({u["label"] for name in assignment for u in assignment[name]})

    print(f"{'label':8}" + "".join(f"{name:>9}" for name in name_order))

    for label in labels:
        row = [sum(1 for u in assignment[name] if u["label"] == label)
               for name in name_order]
        print(f"{label:8}" + "".join(f"{n:>9,}" for n in row))

    print(f"{'TOTAL':8}" + "".join(f"{len(assignment[n]):>9,}" for n in name_order))
    print()

    leaks = 0

    for label in labels:
        # The set of employers each split uses for this label.
        sets = {
            name: {u["employer"] for u in assignment[name] if u["label"] == label}
            for name in name_order
        }
        for first in name_order:
            for second in name_order:
                # first < second so each pair is only compared once.
                if first < second:
                    leaks += len(sets[first] & sets[second])

    print("employer overlaps between splits (must be 0):", leaks)


def main():
    period = input("Enter the period (for example 2023_2026): ").strip()

    project_dir = PROJECT_DIR

    candidate_path = project_dir / "data" / f"candidates_{period}.json"
    candidates = load_json(candidate_path)

    annotations = load_json(project_dir / "data" / f"language_{period}.json")
    employers = load_json(project_dir / "data" / f"employers_{period}.json")
    clusters = load_json(project_dir / "data" / f"clusters_{period}.json")

    # All four files must describe the same texts in the same order.
    if not len(candidates) == len(annotations) == len(employers):
        raise ValueError(
            f"Files disagree on how many texts there are: "
            f"candidates {len(candidates):,}, annotations {len(annotations):,}, "
            f"employers {len(employers):,}."
        )

    threshold = str(clusters["primary_threshold"])
    cluster_ids = clusters["cluster_ids"][threshold]

    print("texts:", len(annotations))
    print("clustering threshold:", threshold)
    print("employer cap:", CAP)
    print()

    generator = random.Random(SEED)

    # Step 1: the Swedish-dominant pool. This becomes train, val and test.
    swedish = eligible_positions(annotations, swedish=True)
    keep = pure_clusters(swedish, annotations, cluster_ids)
    units = one_unit_per_cluster(swedish, annotations, cluster_ids, employers, keep)

    print("Swedish clusters with one label:", f"{len(units):,}")

    units = cap_per_employer(units, CAP, generator)
    per_label = Counter(u["label"] for u in units)

    print("after the employer cap:", f"{len(units):,}")
    print("  " + "  ".join(f"{l}:{per_label[l]:,}" for l in sorted(per_label)))

    # The smallest label decides the size of every label, so the classes match.
    target = min(per_label.values())
    print("balanced size per label:", f"{target:,}")
    print()

    assignment = split_employers(units, target, generator)
    assignment = trim_to_equal(assignment, ["train", "val", "test"])
    report(assignment, ["train", "val", "test"])

    # Step 2: the second test set, where Swedish is NOT the dominant language.
    # IT labels only, and used for inference only. It answers one question:
    # does each method hold up when the advertisement is mostly English?
    other = eligible_positions(annotations, swedish=False)
    keep_other = pure_clusters(other, annotations, cluster_ids)
    other_units = one_unit_per_cluster(other, annotations, cluster_ids,
                                       employers, keep_other)
    other_units = [u for u in other_units if u["label"] in IT_LABELS]
    other_units = cap_per_employer(other_units, CAP, generator)

    # An international company often posts in both Swedish and English. Drop any
    # employer the model has already seen for this label, or the language slice
    # would measure employer recognition instead of language.
    used = {(u["label"], u["employer"])
            for name in ("train", "val") for u in assignment[name]}

    before = len(other_units)
    other_units = [u for u in other_units
                   if (u["label"], u["employer"]) not in used]
    print()
    print(f"non-Swedish units: {before:,} -> {len(other_units):,} "
          f"after removing employers seen in train or val")

    # Match the Swedish test size so the two are equally precise.
    test_size = int(target * PROPORTIONS["test"])
    by_label = defaultdict(list)

    for unit in other_units:
        by_label[unit["label"]].append(unit)

    test_nonsv = []

    for label in IT_LABELS:
        available = by_label[label]
        count = min(test_size, len(available))
        test_nonsv += generator.sample(available, count)

        if count < test_size:
            print(f"  warning: {label} has only {count} non-Swedish clusters, "
                  f"wanted {test_size}")

    print()
    print("non-Swedish IT test set:", f"{len(test_nonsv):,}")
    print("  " + "  ".join(
        f"{l}:{sum(1 for u in test_nonsv if u['label'] == l):,}" for l in IT_LABELS))

    overlap = sum(1 for u in test_nonsv if (u["label"], u["employer"]) in used)
    print("  employers shared with train or val (must be 0):", overlap)

    splits = {
        "period": period,
        # Fingerprint of the candidate file these positions refer to.
        # check_alignment() uses it to catch a rebuilt candidate file.
        "candidate_file": candidate_path.name,
        "candidate_count": len(candidates),
        "candidate_file_bytes": candidate_path.stat().st_size,
        "clustering_threshold": float(threshold),
        "employer_cap": CAP,
        "min_words": MIN_WORDS,
        "language": LANGUAGE,
        "proportions": PROPORTIONS,
        "seed": SEED,
        "size_per_label": target,
        "splits": {
            name: [
                {k: u[k] for k in ("position", "ad_id", "cluster", "label", "employer")}
                for u in assignment[name]
            ]
            for name in assignment
        },
        "test_nonsv": [
            {k: u[k] for k in ("position", "ad_id", "cluster", "label", "employer")}
            for u in test_nonsv
        ],
    }

    out_path = project_dir / "data" / f"splits_{period}.json"
    save_json(splits, out_path)

    print()
    print("Splits saved:", out_path)
    print("These are frozen. Never tune against test or test_nonsv.")


if __name__ == "__main__":
    main()
