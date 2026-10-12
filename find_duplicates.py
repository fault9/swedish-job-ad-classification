import json
from pathlib import Path
from collections import Counter

from datasketch import MinHash, MinHashLSH


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


def build_all_minhashes(candidates, num_perm=256, k=5):
    """Return one MinHash signature per candidate text, in the same order."""

    return [build_minhash(record["text"], num_perm, k) for record in candidates]


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

    neighbours = {}

    for first, second in pairs:
        neighbours.setdefault(first, set()).add(second)
        neighbours.setdefault(second, set()).add(first)

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
