"""Plot which labels each method confuses, as one heatmap per method.

Rows are the true labels, columns the predicted ones, and each row shows the
share of that label's 1,600 advertisements, pooled over the five folds. The
diagonal (correct answers) is grey so the colour scale is spent on the
errors. Lines separate the three families. The BERT models use seed 1.

Only the saved prediction files are read. Nothing is trained.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from files import load_json
from paths import PERIOD, RESULTS_DIR

N_FOLDS = 5

LABELS = ["2511", "2512", "2514", "2330", "2341", "5312",
          "2221", "5321", "5323"]

FILES = {
    "TF-IDF": "predictions_tfidf_fold{}_" + PERIOD + ".json",
    "mBERT": "predictions_mbert_seed1_fold{}_" + PERIOD + ".json",
    "KB-BERT": "predictions_kbbert_seed1_fold{}_" + PERIOD + ".json",
}

# One blue, from almost white for "no errors" to dark for "many".
SURFACE = "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list(
    "blues", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIAGONAL = "#e9e8e4"


def confusion_shares(file_pattern):
    """Return a 9x9 array: the share of each true label given each prediction."""
    counts = np.zeros((len(LABELS), len(LABELS)))

    for fold in range(N_FOLDS):
        for row in load_json(RESULTS_DIR / file_pattern.format(fold))["predictions"]:
            counts[LABELS.index(row["true"]), LABELS.index(row["predicted"])] += 1

    return counts / counts.sum(axis=1, keepdims=True)


def main():
    """Plot the confusion matrices of the three methods side by side."""
    shares = {name: confusion_shares(pattern) for name, pattern in FILES.items()}

    # The same colour scale in every panel, set by the largest error anywhere.
    off_diagonal = [s[~np.eye(len(LABELS), dtype=bool)].max() for s in shares.values()]
    top = max(off_diagonal)

    sns.set_theme(style="white", rc={"figure.facecolor": SURFACE,
                                     "axes.facecolor": DIAGONAL,
                                     "text.color": "#0b0b0b",
                                     "axes.labelcolor": "#52514e",
                                     "xtick.color": "#52514e",
                                     "ytick.color": "#52514e"})

    # A thin fourth column holds the colour bar, so the three panels match in size.
    figure, (*axes, bar) = plt.subplots(
        1, 4, figsize=(16, 5.6), gridspec_kw={"width_ratios": [1, 1, 1, 0.05]})

    for axis, (name, share) in zip(axes, shares.items()):
        # Only the errors get a colour; the diagonal stays grey.
        sns.heatmap(share * 100, ax=axis, mask=np.eye(len(LABELS), dtype=bool),
                    cmap=BLUES, vmin=0, vmax=top * 100,
                    cbar=axis is axes[-1], cbar_ax=bar,
                    cbar_kws={"label": "% of the true label"},
                    linewidths=1, linecolor=SURFACE, square=True,
                    xticklabels=LABELS, yticklabels=LABELS)

        # Write the percentage in each cell from 1% up, and every diagonal cell.
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                value = share[i, j] * 100
                if i != j and value < 1:
                    continue
                dark = i != j and value > top * 100 * 0.55
                axis.text(j + 0.5, i + 0.5, f"{value:.0f}", ha="center", va="center",
                          fontsize=8, color="#fcfcfb" if dark else "#52514e")

        # Lines between the three families.
        for edge in (3, 6):
            axis.axhline(edge, color="#52514e", linewidth=1)
            axis.axvline(edge, color="#52514e", linewidth=1)

        axis.set_title(name, fontsize=13)
        axis.set_xlabel("predicted label")
        axis.set_ylabel("true label" if axis is axes[0] else "")
        axis.tick_params(axis="x", rotation=90)
        axis.tick_params(axis="y", rotation=0)

    figure.suptitle("Which labels each method confuses, all 14,400 test advertisements "
                    "(grey diagonal = correct)", fontsize=14)
    figure.tight_layout()

    for suffix in ("png", "pdf"):
        path = RESULTS_DIR / f"confusion.{suffix}"
        figure.savefig(path, dpi=200, bbox_inches="tight")
        print("Saved:", path)


if __name__ == "__main__":
    main()
