"""Measure how many tokens each BERT model needs per Swedish word.

Both models read at most 512 tokens. The result decides WORD_BUDGET in
truncate.py: the largest number of words that fits for both models.
"""

import random

from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

from dataset import load_split
from truncate import first_words

MODELS = {
    "KB-BERT": "KB/bert-base-swedish-cased",
    "mBERT": "google-bert/bert-base-multilingual-cased",
}

MAX_TOKENS = 512
WORD_BUDGETS = [150, 200, 250, 300]
SAMPLE_SIZE = 2000        # tokenising is slow, so a sample of the train split


def count_tokens(tokenizer, text):
    """Return how many tokens the model makes from the text, [CLS] and [SEP] included."""
    return len(tokenizer(text, truncation=False)["input_ids"])


def main():
    hf_logging.set_verbosity_error()

    texts, _ = load_split("train", truncate=False)
    texts = random.Random(42).sample(texts, SAMPLE_SIZE)
    words = sum(len(text.split()) for text in texts)

    tokenizers = {name: AutoTokenizer.from_pretrained(model_id)
                  for name, model_id in MODELS.items()}

    print(f"{SAMPLE_SIZE:,} training advertisements")
    print()

    for name, tokenizer in tokenizers.items():
        tokens = sum(count_tokens(tokenizer, text) for text in texts)
        print(f"  {name:8} {tokens / words:.2f} tokens per word")
    print()

    print(f"  share of advertisements whose first N words fit in {MAX_TOKENS} tokens")
    print(f"  {'N words':>8}" + "".join(f"{name:>10}" for name in tokenizers))
    for budget in WORD_BUDGETS:
        cut = [first_words(text, budget) for text in texts]
        row = f"  {budget:>8}"
        for tokenizer in tokenizers.values():
            fits = sum(1 for text in cut if count_tokens(tokenizer, text) <= MAX_TOKENS)
            row += f"{fits / len(cut):>10.1%}"
        print(row)


if __name__ == "__main__":
    main()
