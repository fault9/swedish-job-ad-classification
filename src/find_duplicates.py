import time
from collections import defaultdict

from datasketch import MinHash, MinHashLSH

from files import load_json, save_json

from paths import CANDIDATES_FILE, CLUSTERS_FILE, PERIOD, REPORTS_DIR

THRESHOLDS = [0.6, 0.7, 0.8, 0.9]
PRIMARY_THRESHOLD = 0.8
SIZE_BANDS = [(1, 1), (2, 2), (3, 5), (6, 20), (21, 100), (101, 10 ** 9)]


def text_shingles(text, k=5):
    """Return the set of k-word sequences in one text."""

    words = text.split()

    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def build_minhash(text, num_perm=256, k=5):
    """Return a MinHash signature for one text."""

    signature = MinHash(num_perm=num_perm)

    for shingle in text_shingles(text, k):
        signature.update(shingle.encode("utf-8"))

    return signature


def build_all_minhashes(candidates, num_perm=256, k=5, report_every=25000):
    """Return one MinHash signature per candidate text, in the same order."""

    signatures = []

    for record in candidates:
        signatures.append(build_minhash(record["text"], num_perm, k))

        if len(signatures) % report_every == 0:
            print("signatures", len(signatures), "of", len(candidates), flush=True)

    return signatures


def find_duplicate_pairs(signatures, threshold=0.8, num_perm=256):
    """Return the set of position pairs whose estimated Jaccard passes threshold."""

    index = MinHashLSH(threshold=threshold, num_perm=num_perm)

    for position, signature in enumerate(signatures):
        index.insert(position, signature)

    pairs = set()

    for position, signature in enumerate(signatures):
        for other in index.query(signature):
            if other != position:
                pairs.add((min(position, other), max(position, other)))

    return pairs


def cluster_positions(pairs, n_texts):
    """Give every position a cluster id, joining positions linked by pairs."""

    neighbours = defaultdict(set)

    for first, second in pairs:
        neighbours[first].add(second)
        neighbours[second].add(first)

    cluster_ids = [None] * n_texts
    next_id = 0

    for start in range(n_texts):
        if cluster_ids[start] is not None:
            continue

        stack = [start]

        while stack:
            position = stack.pop()

            if cluster_ids[position] is not None:
                continue

            cluster_ids[position] = next_id

            for other in neighbours.get(position, ()):
                if cluster_ids[other] is None:
                    stack.append(other)

        next_id += 1

    return cluster_ids


def summarise_clusters(cluster_ids, labels):
    """Count clusters, size bands and label purity for one threshold."""

    members = defaultdict(list)

    for position, cluster_id in enumerate(cluster_ids):
        members[cluster_id].append(position)

    sizes = [len(positions) for positions in members.values()]

    size_bands = {}

    for low, high in SIZE_BANDS:
        name = str(low) if low == high else f"{low}-{high}" if high < 10 ** 9 else f"{low}+"
        size_bands[name] = sum(1 for size in sizes if low <= size <= high)

    mixed_label_clusters = sum(
        1 for positions in members.values()
        if len(positions) > 1 and len({labels[p] for p in positions}) > 1
    )

    label_clusters = defaultdict(set)

    for position, cluster_id in enumerate(cluster_ids):
        label_clusters[labels[position]].add(cluster_id)

    clusters_per_label = {
        label: len(ids) for label, ids in sorted(label_clusters.items())
    }

    return {
        "n_texts": len(cluster_ids),
        "n_clusters": len(members),
        "singletons": size_bands.get("1", 0),
        "largest_cluster": max(sizes),
        "size_bands": size_bands,
        "mixed_label_clusters": mixed_label_clusters,
        "clusters_per_label": clusters_per_label,
        "floor_label": min(clusters_per_label, key=clusters_per_label.get),
        "floor_clusters": min(clusters_per_label.values()),
    }


def main():
    report_path = REPORTS_DIR / f"duplicates_{PERIOD}.json"

    if not CANDIDATES_FILE.exists():
        raise FileNotFoundError(f"No candidate file at {CANDIDATES_FILE}")

    candidates = load_json(CANDIDATES_FILE)

    print("Candidate records loaded:", len(candidates), flush=True)

    started = time.perf_counter()
    signatures = build_all_minhashes(candidates)
    print(f"Signatures built in {time.perf_counter() - started:.0f} s", flush=True)

    # The texts are no longer needed. Keep only the labels, and release the
    # candidate list before the sweep, which is the memory-hungry part.
    labels = [record["label"] for record in candidates]
    candidates = None

    cluster_ids_by_threshold = {}
    summaries = {}

    for threshold in THRESHOLDS:
        started = time.perf_counter()

        pairs = find_duplicate_pairs(signatures, threshold=threshold)
        cluster_ids = cluster_positions(pairs, len(signatures))

        summary = summarise_clusters(cluster_ids, labels)
        summary["n_pairs"] = len(pairs)
        summary["seconds"] = round(time.perf_counter() - started, 1)

        cluster_ids_by_threshold[str(threshold)] = cluster_ids
        summaries[str(threshold)] = summary

        print(
            f"threshold {threshold}: {summary['n_clusters']} clusters, "
            f"floor {summary['floor_clusters']} ({summary['floor_label']}), "
            f"largest {summary['largest_cluster']}, "
            f"mixed-label {summary['mixed_label_clusters']} "
            f"[{summary['seconds']} s]",
            flush=True,
        )

        pairs = None

    save_json(
        {
            "period": PERIOD,
            "primary_threshold": PRIMARY_THRESHOLD,
            "num_perm": 256,
            "shingle_words": 5,
            "cluster_ids": cluster_ids_by_threshold,
        },
        CLUSTERS_FILE,
    )

    save_json(
        {
            "period": PERIOD,
            "primary_threshold": PRIMARY_THRESHOLD,
            "num_perm": 256,
            "shingle_words": 5,
            "thresholds": summaries,
        },
        report_path,
    )

    print("Clusters saved:", CLUSTERS_FILE)
    print("Report saved:", report_path)


if __name__ == "__main__":
    main()
