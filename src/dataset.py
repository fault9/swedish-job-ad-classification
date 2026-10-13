"""Load the frozen splits and turn them into model input.

All three methods use this, so they all see exactly the same text. That is the
point: if TF-IDF read whole advertisements while the BERT models read only the
first part, any difference between them would partly be a difference in how
much text they saw, not in how well they represent Swedish.

The truncation happens here, on words, before any tokenizer runs. Measured on
the train split, mBERT needs 1.97 tokens per Swedish word against KB-BERT's
1.31, so cutting at 512 TOKENS would give the two models different amounts of
text. Cutting at 200 WORDS gives them the same text, and 200 is the largest
budget where both models fit at least 95% of advertisements inside their
512-token limit.

Nothing here touches the test splits unless you ask for them by name.
"""

import json
from pathlib import Path

WORD_BUDGET = 200     # words kept per advertisement, measured in measure_tokens.py


def first_words(text, n_words):
    """Return the first n_words words of the text."""
    return " ".join(text.split()[:n_words])


def head_tail_words(text, n_words):
    """Return half the budget from the start of the text and half from the end.

    Sun et al. (2019) compared keeping the start, keeping the end, and keeping
    both, on documents too long for BERT. Both was best on their two datasets.
    It suits Swedish job advertisements because the employer description tends
    to come first and the actual requirements tend to come last, so keeping
    only the start would throw away the part that names the skills.
    """
    words = text.split()

    if len(words) <= n_words:
        return " ".join(words)

    half = n_words // 2
    # n_words - half, not half, so an odd budget loses no word.
    return " ".join(words[:half] + words[-(n_words - half):])


def load_split(project_dir, period, name, budget=WORD_BUDGET, truncate=True):
    """Load one split and return its texts and labels as two matching lists.

    name is "train", "val", "test" or "test_nonsv".

    Set truncate=False to get the untruncated text. That is only for the one
    extra TF-IDF run that measures what the 200-word budget costs; the three
    methods being compared must all use the default.

    Before returning anything this checks that the saved positions still point
    at the advertisements they did when the splits were frozen. If the
    candidate file has been rebuilt the check raises, because training on
    silently misaligned data is far worse than stopping.
    """
    from make_splits import check_alignment

    data_dir = Path(project_dir) / "data"

    with (data_dir / f"candidates_{period}.json").open(encoding="utf-8") as file:
        candidates = json.load(file)

    with (data_dir / f"splits_{period}.json").open(encoding="utf-8") as file:
        splits = json.load(file)

    check_alignment(splits, candidates)

    # test_nonsv sits at the top level; the other three are nested under "splits".
    units = splits["test_nonsv"] if name == "test_nonsv" else splits["splits"][name]

    texts, labels = [], []

    for unit in units:
        text = candidates[unit["position"]]["text"]
        if truncate:
            text = head_tail_words(text, budget)
        texts.append(text)
        # Take the label from the splits, not the candidate, so the two can
        # never disagree about what this example is.
        labels.append(unit["label"])

    return texts, labels


def describe(texts, labels, name):
    """Print a one-line summary of a loaded split."""
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    words = [len(t.split()) for t in texts]
    sizes = sorted(set(counts.values()))

    print(f"  {name:11} {len(texts):>6,} texts   {len(counts)} labels   "
          f"{sizes[0]}-{sizes[-1]} per label   "
          f"words min {min(words)} median {sorted(words)[len(words)//2]} max {max(words)}")
