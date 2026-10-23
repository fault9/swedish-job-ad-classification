"""Methods 2 and 3: fine-tune a BERT model across the five folds.

The same script runs KB-BERT and mBERT with the same code, so any difference
in the results comes from the model and not from the code around it.

It follows train_tfidf_cv.py on purpose: train on four folds, predict the
fifth, save the predictions in the same format, then pool all five.

The training settings are the standard ones from Devlin et al. (2019), fixed
before any run and shared by both models. Nothing is tuned.

Each model runs with several seeds. The confidence interval from resampling
employers does not cover training randomness, and the gap between the two
BERTs is small enough that one seed could flip it. A run whose results file
already exists is skipped, so the script can be restarted after a crash.

The fine-tuned models are not saved. Everything the analysis needs is in the
prediction files, and a model can be retrained in a few minutes with the same
seed.
"""

import time

import torch
import transformers
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding,
                          get_linear_schedule_with_warmup)

from dataset import load_fold
from evaluate import bootstrap_macro_f1, macro_f1, report, save_predictions
from files import load_json, save_json
from paths import PERIOD, RESULTS_DIR
from truncate import INPUT_STRATEGY

N_FOLDS = 5

MODELS = {
    "kbbert": "KB/bert-base-swedish-cased",
    "mbert": "google-bert/bert-base-multilingual-cased",
}

# Every method runs once per seed.
METHODS = ["kbbert", "mbert"]
SEEDS = [1, 2, 3]

# BERT reads at most 512 tokens. The texts are already cut to 200 words.
MAX_TOKENS = 512

LEARNING_RATE = 2e-5
BATCH_SIZE = 16
EPOCHS = 3
WARMUP_SHARE = 0.1


def to_rows(tokenizer, texts, label_ids=None):
    """Tokenize texts into the rows the DataLoader expects."""
    encoded = tokenizer(texts, truncation=True, max_length=MAX_TOKENS)

    rows = []
    for position, ids in enumerate(encoded["input_ids"]):
        row = {"input_ids": ids,
               "attention_mask": encoded["attention_mask"][position]}
        if label_ids is not None:
            row["labels"] = label_ids[position]
        rows.append(row)

    return rows


def run_fold(fold_number, model_name, seed, tokenizer, device):
    """Fine-tune on four folds and predict the fifth.

    Returns:
        The fold data, the predicted labels, one row per epoch, and the [CLS]
        representation of every test ad, which plot_embeddings.py draws.
    """
    torch.manual_seed(seed)

    data = load_fold(fold_number)

    # BERT needs each label as a number from 0 to 8, not as "2511".
    labels = sorted(set(data["train_labels"]))
    label_to_id = {label: number for number, label in enumerate(labels)}

    train_rows = to_rows(tokenizer, data["train_texts"],
                         [label_to_id[label] for label in data["train_labels"]])
    test_rows = to_rows(tokenizer, data["test_texts"])

    # Texts in a batch have different lengths. The collator pads the short
    # ones so every batch is a neat table.
    collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(train_rows, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collator)
    test_loader = DataLoader(test_rows, batch_size=BATCH_SIZE,
                             collate_fn=collator)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Start with small steps, grow to the full learning rate, then shrink
    # back to zero by the end.
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * WARMUP_SHARE), total_steps)

    model.train()
    epoch_rows = []

    for epoch in range(EPOCHS):
        started = time.perf_counter()
        total_loss = 0

        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                output = model(**batch)

            output.loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += output.loss.item()

        loss = total_loss / len(train_loader)
        seconds = time.perf_counter() - started
        epoch_rows.append({"epoch": epoch + 1, "loss": round(loss, 4),
                           "seconds": round(seconds, 1)})
        print(f"    epoch {epoch + 1}   loss {loss:.4f}   {seconds:.0f} s")

    # Now the test fold: ads the model has never seen. No labels go in.
    model.eval()
    predicted_ids = []
    embeddings = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {key: value.to(device) for key, value in batch.items()}

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                output = model(**batch, output_hidden_states=True)

            # The label with the highest score is the model's answer.
            predicted_ids += output.logits.argmax(dim=1).tolist()

            # The last layer's [CLS] vector is what the classifier reads.
            embeddings.append(output.hidden_states[-1][:, 0].float().cpu())

    # Turn 0 to 8 back into "2221" to "5323".
    predictions = [labels[number] for number in predicted_ids]

    # Free the graphics card before the next fold loads a new model.
    del model, optimizer
    torch.cuda.empty_cache()

    return data, predictions, epoch_rows, torch.cat(embeddings).numpy()


def run_method(method, seed, device):
    """Run one method with one seed across all five folds and save it."""
    model_name = MODELS[method]

    print(f"{method} across {N_FOLDS} folds, seed {seed}")
    print(f"  model {model_name}   running on {device}")
    print(f"  settings: learning rate {LEARNING_RATE}, batch {BATCH_SIZE}, "
          f"epochs {EPOCHS}, warmup {WARMUP_SHARE}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    all_units = []
    all_true = []
    all_predicted = []
    fold_rows = []

    for fold_number in range(N_FOLDS):
        print(f"  fold {fold_number}")
        started = time.perf_counter()
        data, predictions, epoch_rows, _ = run_fold(fold_number, model_name,
                                                    seed, tokenizer, device)
        seconds = time.perf_counter() - started

        score = macro_f1(data["test_labels"], predictions)

        all_units += data["test_units"]
        all_true += data["test_labels"]
        all_predicted += predictions

        fold_rows.append({
            "fold": fold_number,
            "n_train": len(data["train_texts"]),
            "n_test": len(data["test_texts"]),
            "epochs": epoch_rows,
            "macro_f1": score,
            "seconds": round(seconds, 1),
        })

        print(f"    macro F1 {score:.4f}   {seconds:.0f} s in total")

        save_predictions(
            data["test_units"], data["test_labels"], predictions,
            RESULTS_DIR / f"predictions_{method}_seed{seed}_fold{fold_number}"
                          f"_{PERIOD}.json",
            method=method, split=f"fold{fold_number}",
        )

    scores = [row["macro_f1"] for row in fold_rows]
    mean = sum(scores) / len(scores)

    print()
    print(f"  per-fold mean {mean:.4f}   lowest {min(scores):.4f}   "
          f"highest {max(scores):.4f}   spread {max(scores) - min(scores):.4f}")
    print()

    # The pooled result: every advertisement predicted exactly once.
    print("=" * 62)
    results = report(all_true, all_predicted,
                     f"POOLED over {N_FOLDS} folds, every advertisement once")

    print()
    print("  bootstrapping the interval by resampling employers ...")
    interval = bootstrap_macro_f1(all_units, all_true, all_predicted)

    print(f"  macro F1   {interval['macro_f1']:.4f}")
    print(f"  95% CI     [{interval['ci_low']:.4f}, {interval['ci_high']:.4f}]"
          f"   width {interval['ci_high'] - interval['ci_low']:.4f}")
    print(f"  from       {interval['n_advertisements']:,} advertisements, "
          f"{interval['n_employers']:,} employers")

    save_json(
        {
            "method": method,
            "input": INPUT_STRATEGY,
            "model": model_name,
            "period": PERIOD,
            "n_folds": N_FOLDS,
            "settings": {"learning_rate": LEARNING_RATE,
                         "batch_size": BATCH_SIZE, "epochs": EPOCHS,
                         "warmup_share": WARMUP_SHARE, "seed": seed,
                         "max_tokens": MAX_TOKENS,
                         "chosen_on": "Devlin et al. (2019), not tuned"},
            "versions": {"torch": torch.__version__,
                         "transformers": transformers.__version__,
                         "gpu": torch.cuda.get_device_name(0)
                                if device == "cuda" else "none"},
            "folds": fold_rows,
            "per_fold_mean": mean,
            "per_fold_spread": max(scores) - min(scores),
            "pooled": {k: results[k] for k in
                       ("n", "accuracy", "macro_f1", "majority_baseline",
                        "families", "confusion_matrix", "labels")},
            "pooled_interval": interval,
        },
        results_path(method, seed),
    )

    print()
    print("Saved:", results_path(method, seed))
    print(f"Saved: {N_FOLDS} per-fold prediction files")
    print()


def results_path(method, seed):
    return RESULTS_DIR / f"{method}_seed{seed}_cv_{PERIOD}.json"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for method in METHODS:
        for seed in SEEDS:
            if results_path(method, seed).exists():
                print(f"{method} seed {seed}: already done, skipped")
                continue

            run_method(method, seed, device)

    # One line per run, read back from the saved files.
    print("=" * 62)
    print("ALL RUNS")
    for method in METHODS:
        for seed in SEEDS:
            path = results_path(method, seed)
            if path.exists():
                saved = load_json(path)
                interval = saved["pooled_interval"]
                print(f"  {method:7} seed {seed}   macro F1 "
                      f"{interval['macro_f1']:.4f}   "
                      f"CI [{interval['ci_low']:.4f}, {interval['ci_high']:.4f}]")


if __name__ == "__main__":
    main()
