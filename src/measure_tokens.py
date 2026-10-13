"""Find the word budget that fits both BERT models.

Each model splits text into tokens and accepts at most 512 of them. The two
models split Swedish differently, so the same advertisement costs a different
number of tokens in each. This script measures that and prints the largest
number of words that still fits in both.
"""

import json
import random
from pathlib import Path

from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

from paths import PROJECT_DIR

MODELS = {
    "KB-BERT": "KB/bert-base-swedish-cased",
    "mBERT": "google-bert/bert-base-multilingual-cased",
}

MAX_TOKENS = 512          # the limit of both models
WORD_BUDGETS = [150, 200, 250, 300, 350, 400, 450, 500]
TARGET_COVERAGE = 0.95    # a budget passes if 95% of texts fit


def load_records(candidate_path, sample_size=2000, seed=42):
    """Return candidate records. Use sample_size=0 for all of them."""
    with candidate_path.open("r", encoding="utf-8") as file:
        candidates = json.load(file)

    if sample_size == 0 or sample_size >= len(candidates):
        return candidates

    return random.Random(seed).sample(candidates, sample_size)


def count_tokens(tokenizer, text):
    """Return how many tokens this model would make from the text."""
    # add_special_tokens adds [CLS] and [SEP], which also count towards the 512.
    # truncation=False means count the real length instead of stopping at 512.
    encoded = tokenizer(text, add_special_tokens=True, truncation=False)

    return len(encoded["input_ids"])


def first_words(text, n_words):
    """Return the first n_words words of the text."""
    return " ".join(text.split()[:n_words])


def head_tail_words(text, n_words):
    """Return half the budget from the start and half from the end."""
    words = text.split()

    if len(words) <= n_words:
        return " ".join(words)

    half = n_words // 2

    # n_words - half, not half, so an odd budget loses no word.
    return " ".join(words[:half] + words[-(n_words - half):])


def percentile(values, fraction):
    """Return the value that the given fraction of the data falls below.

    percentile(values, 0.50) is the median.
    """
    ordered = sorted(values)
    position = min(int(fraction * len(ordered)), len(ordered) - 1)

    return ordered[position]


def describe_full_texts(texts, tokenizers):
    """Print the token cost of the full, uncut advertisements."""
    print("=== Untruncated advertisements ===")
    print(f"texts measured: {len(texts)}")

    word_counts = [len(text.split()) for text in texts]
    print(f"words: median {percentile(word_counts, 0.50)}"
          f"  p95 {percentile(word_counts, 0.95)}"
          f"  max {max(word_counts)}")
    print()

    print(f"{'model':10}{'tok/word':>10}{'median':>9}{'p95':>8}{'max':>8}{'over 512':>10}")

    for name, tokenizer in tokenizers.items():
        token_counts = [count_tokens(tokenizer, text) for text in texts]
        ratio = sum(token_counts) / sum(word_counts)
        over = sum(1 for n in token_counts if n > MAX_TOKENS) / len(token_counts)

        print(f"{name:10}{ratio:>10.2f}{percentile(token_counts, 0.50):>9}"
              f"{percentile(token_counts, 0.95):>8}{max(token_counts):>8}"
              f"{over:>9.1%}")

    print()


def describe_word_budgets(texts, tokenizers):
    """Print how often each budget fits, and return the largest one that does."""
    print("=== Does the first N words fit in 512 tokens? ===")

    header = f"{'N words':>9}" + "".join(f"{name:>14}" for name in tokenizers)
    print(header)
    print("-" * len(header))

    safe_budgets = []

    for n_words in WORD_BUDGETS:
        truncated = [first_words(text, n_words) for text in texts]
        shares = {}

        for name, tokenizer in tokenizers.items():
            counts = [count_tokens(tokenizer, text) for text in truncated]
            shares[name] = sum(1 for n in counts if n <= MAX_TOKENS) / len(counts)

        print(f"{n_words:>9}" + "".join(f"{shares[name]:>13.1%}" for name in tokenizers))

        if all(share >= TARGET_COVERAGE for share in shares.values()):
            safe_budgets.append(n_words)

    print()

    if safe_budgets:
        print(f"Largest budget where at least {TARGET_COVERAGE:.0%} of texts fit "
              f"under BOTH models: {max(safe_budgets)} words")
    else:
        print(f"No tested budget reaches {TARGET_COVERAGE:.0%} under both models. "
              f"Try smaller values than {min(WORD_BUDGETS)}.")

    print()
    print("Use that budget to truncate the TEXT before tokenising, and give the")
    print("same truncated text to all three methods. Record the number you chose.")
    print()

    return max(safe_budgets) if safe_budgets else None


def describe_loss_per_label(records, budget):
    """Print how many texts each label loses to the budget."""
    print(f"=== Texts longer than the {budget}-word budget, per label ===")

    per_label = {}
    for record in records:
        words = len(record["text"].split())
        counts = per_label.setdefault(record["label"], [0, 0])
        counts[0] += 1
        if words > budget:
            counts[1] += 1

    print(f"{'label':8}{'texts':>8}{'truncated':>12}{'share':>9}")

    shares = {}
    for label in sorted(per_label):
        total, truncated = per_label[label]
        shares[label] = truncated / total
        print(f"{label:8}{total:>8}{truncated:>12}{shares[label]:>8.0%}")

    spread = max(shares.values()) - min(shares.values())
    print()
    print(f"spread between labels: {spread:.0%} "
          f"({min(shares, key=shares.get)} lowest, {max(shares, key=shares.get)} highest)")

    if spread > 0.10:
        print("Report this. Truncation loss is uneven across labels, so per-class")
        print("scores are not directly comparable to each other.")


def main():
    hf_logging.set_verbosity_error()

    period = input("Enter the period (for example 2023_2026): ").strip()

    project_dir = PROJECT_DIR
    candidate_path = project_dir / "data" / f"candidates_{period}.json"

    if not candidate_path.exists():
        raise FileNotFoundError(f"No candidate file at {candidate_path}")

    with candidate_path.open("r", encoding="utf-8") as file:
        candidates = json.load(file)

    # If the splits exist, measure the TRAIN split only. That is the text the
    # models will actually be fitted on, and it is a filtered, employer-capped
    # subset of the candidates, so its lengths can differ from the whole corpus.
    splits_path = project_dir / "data" / f"splits_{period}.json"

    if splits_path.exists():
        from make_splits import check_alignment

        with splits_path.open("r", encoding="utf-8") as file:
            splits = json.load(file)

        # Stop if the candidate file has been rebuilt since the splits were made.
        check_alignment(splits, candidates)

        train = splits["splits"]["train"]
        records = [candidates[unit["position"]] for unit in train]
        # Labels live in the splits, not the candidate records we just pulled.
        for record, unit in zip(records, train):
            record["label"] = unit["label"]
        print(f"measuring the TRAIN split: {len(records):,} texts")
    else:
        records = candidates
        print(f"no splits file, measuring all candidates: {len(records):,} texts")

    print()

    # Tokenising is slow, so the token measurements use a sample of those.
    sample = random.Random(42).sample(records, min(2000, len(records)))
    texts = [record["text"] for record in sample]

    tokenizers = {}
    for name, model_id in MODELS.items():
        print(f"loading tokenizer: {name} ({model_id})")
        tokenizers[name] = AutoTokenizer.from_pretrained(model_id)
    print()

    describe_full_texts(texts, tokenizers)
    budget = describe_word_budgets(texts, tokenizers)

    if budget is not None:
        # Counting words is fast, so use every record for the per-label table.
        describe_loss_per_label(records, budget)


if __name__ == "__main__":
    main()
