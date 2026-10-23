"""Group near-duplicate advertisements into clusters.

Employers repost the same advertisement with small changes, so removing exact
copies is not enough. Two texts count as near-duplicates when their sets of
five-word sequences (shingles) have a Jaccard similarity at or above the
threshold. Comparing every pair would take billions of comparisons, so each
text gets a MinHash signature and locality-sensitive hashing (LSH) finds the
likely pairs. Pairs are then joined into clusters: texts linked by a chain of
near-duplicates share a cluster.

Several thresholds are run so they can be compared; make_splits.py uses 0.8.
"""

import time
from collections import defaultdict

from datasketch import MinHash, MinHashLSH

from files import load_json, save_json
from paths import CANDIDATES_FILE, CLUSTERS_FILE, PERIOD, REPORTS_DIR

SHINGLE_WORDS = 5
NUM_PERM = 256
THRESHOLDS = [0.6, 0.7, 0.8, 0.9]
PRIMARY_THRESHOLD = 0.8


def build_signature(text):
    """Return the MinHash signature of the text's five-word sequences."""
    words = text.split()
    signature = MinHash(num_perm=NUM_PERM)

    for i in range(len(words) - SHINGLE_WORDS + 1):
        signature.update(" ".join(words[i:i + SHINGLE_WORDS]).encode("utf-8"))

    return signature


def find_pairs(signatures, threshold):
    """Return every pair of positions whose estimated Jaccard passes the threshold."""
    index = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)

    for position, signature in enumerate(signatures):
        index.insert(position, signature)

    pairs = set()

    for position, signature in enumerate(signatures):
        for other in index.query(signature):
            if other != position:
                pairs.add((min(position, other), max(position, other)))

    return pairs


def cluster_positions(pairs, n_texts):
    """Give every text a cluster id; texts linked by a chain of pairs share one.

    Starting from each text without an id, walk to its neighbours, their
    neighbours and so on, giving them all the same id.
    """
    neighbours = defaultdict(set)

    for first, second in pairs:
        neighbours[first].add(second)
        neighbours[second].add(first)

    cluster_ids = [None] * n_texts
    next_id = 0

    for start in range(n_texts):
        if cluster_ids[start] is not None:
            continue

        to_visit = [start]

        while to_visit:
            position = to_visit.pop()

            if cluster_ids[position] is not None:
                continue

            cluster_ids[position] = next_id
            to_visit.extend(neighbours[position])

        next_id += 1

    return cluster_ids


def summarise(cluster_ids, labels):
    """Count the clusters, their sizes and how many mix two labels."""
    members = defaultdict(list)

    for position, cluster_id in enumerate(cluster_ids):
        members[cluster_id].append(position)

    sizes = [len(positions) for positions in members.values()]

    mixed = sum(1 for positions in members.values()
                if len({labels[p] for p in positions}) > 1)

    per_label = defaultdict(set)

    for position, cluster_id in enumerate(cluster_ids):
        per_label[labels[position]].add(cluster_id)

    per_label = {label: len(ids) for label, ids in sorted(per_label.items())}

    return {
        "n_texts": len(cluster_ids),
        "n_clusters": len(members),
        "singletons": sizes.count(1),
        "largest_cluster": max(sizes),
        "mixed_label_clusters": mixed,
        "clusters_per_label": per_label,
        "floor_label": min(per_label, key=per_label.get),
        "floor_clusters": min(per_label.values()),
    }


def main():
    candidates = load_json(CANDIDATES_FILE)
    print("Candidate records loaded:", len(candidates), flush=True)

    started = time.perf_counter()
    signatures = [build_signature(record["text"]) for record in candidates]
    print(f"Signatures built in {time.perf_counter() - started:.0f} s", flush=True)

    # Keep only the labels; the texts are not needed any more and take memory.
    labels = [record["label"] for record in candidates]
    candidates = None

    cluster_ids_by_threshold = {}
    summaries = {}

    for threshold in THRESHOLDS:
        started = time.perf_counter()

        pairs = find_pairs(signatures, threshold)
        cluster_ids = cluster_positions(pairs, len(signatures))

        summary = summarise(cluster_ids, labels)
        summary["n_pairs"] = len(pairs)
        summary["seconds"] = round(time.perf_counter() - started, 1)

        cluster_ids_by_threshold[str(threshold)] = cluster_ids
        summaries[str(threshold)] = summary

        print(f"threshold {threshold}: {summary['n_clusters']} clusters, "
              f"largest {summary['largest_cluster']}, "
              f"mixed-label {summary['mixed_label_clusters']} "
              f"[{summary['seconds']} s]", flush=True)

    settings = {"period": PERIOD, "primary_threshold": PRIMARY_THRESHOLD,
                "num_perm": NUM_PERM, "shingle_words": SHINGLE_WORDS}

    save_json({**settings, "cluster_ids": cluster_ids_by_threshold}, CLUSTERS_FILE)

    report_path = REPORTS_DIR / f"duplicates_{PERIOD}.json"
    save_json({**settings, "thresholds": summaries}, report_path)

    print("Clusters saved:", CLUSTERS_FILE)
    print("Report saved:", report_path)


if __name__ == "__main__":
    main()
