"""Plot how each model places the test advertisements of one fold, in 2D.

Each dot is one advertisement. Colour is the family, marker is the label.
TF-IDF is drawn from its word weights, the BERT models from the [CLS] vector
their classifier reads. Both are reduced to two dimensions: TruncatedSVD for
the sparse TF-IDF vectors, PCA for the dense BERT vectors.

The BERT models are retrained on the same fold and seed as the saved run,
because the fine-tuned models were not kept. The vectors are cached, so
changing the plot does not retrain anything.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA, TruncatedSVD
from transformers import AutoTokenizer

from dataset import load_fold
from evaluate import FAMILIES
from files import load_json
from paths import PERIOD, RESULTS_DIR
from train_bert import MODELS, run_fold
from train_tfidf import build_model
from train_tfidf_cv import C_VALUE, LOWERCASE, MIN_DF, NGRAM

FOLD = 0
SEED = 1

# Families as in the report. The three colours are the first three slots of a
# colour-blind-safe palette, the only ones that stay apart in a scatter.
FAMILY_NAMES = {"IT": "IT", "Education": "Education", "Healthcare": "Nursing"}
COLOURS = {"IT": "#2a78d6", "Education": "#eb6834", "Nursing": "#1baf7a"}
MARKERS = {
    "2511": "o", "2512": "^", "2514": "s",
    "2330": "o", "2341": "^", "5312": "s",
    "2221": "o", "5321": "^", "5323": "s",
}
LABEL_ORDER = ["2511", "2512", "2514", "2330", "2341", "5312",
               "2221", "5321", "5323"]

# Where each family's centre is placed in every panel.
TARGETS = {"IT": (-1.0, 0.6), "Education": (1.0, 0.6), "Nursing": (0.0, -1.0)}


def bert_vectors(method, device):
    """Return the [CLS] vectors of the test fold, retraining only if not cached."""
    path = RESULTS_DIR / f"embeddings_{method}_fold{FOLD}_seed{SEED}.npy"

    if path.exists():
        return np.load(path)

    print(f"retraining {method} on fold {FOLD}, seed {SEED}")
    model_name = MODELS[method]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    _, predictions, _, vectors = run_fold(FOLD, model_name, SEED, tokenizer, device)

    # Check that the retrained model behaves like the saved run.
    saved = load_json(RESULTS_DIR / f"predictions_{method}_seed{SEED}_fold{FOLD}_{PERIOD}.json")
    same = sum(1 for row, p in zip(saved["predictions"], predictions) if row["predicted"] == p)
    print(f"  same prediction as the saved run on {same / len(predictions):.1%} of ads")

    np.save(path, vectors)
    return vectors


def tfidf_points(data):
    """Return 2D points and the explained variance for the TF-IDF vectors."""
    vectorizer, _ = build_model(NGRAM, MIN_DF, C_VALUE, LOWERCASE)
    vectorizer.fit(data["train_texts"])
    vectors = vectorizer.transform(data["test_texts"])

    svd = TruncatedSVD(n_components=2, random_state=0)
    return svd.fit_transform(vectors), svd.explained_variance_ratio_


def pca_points(vectors):
    """Return 2D points and the explained variance for dense vectors."""
    pca = PCA(n_components=2, random_state=0)
    return pca.fit_transform(vectors), pca.explained_variance_ratio_


def align(points, families):
    """Rotate and mirror the points so each family lands in the same place.

    PCA picks the directions but not which way they point, so without this the
    same clusters can appear in different corners in each panel. Rotating and
    mirroring keep every distance, so nothing about the picture changes except
    its orientation.
    """
    families = np.array(families)
    centred = points - points.mean(axis=0)
    centroids = np.array([centred[families == name].mean(axis=0) for name in TARGETS])
    target = np.array(list(TARGETS.values())) * np.abs(centroids).max()

    rotation, _ = orthogonal_procrustes(centroids, target)
    return centred @ rotation


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_fold(FOLD)
    families = [FAMILY_NAMES[FAMILIES[label]] for label in data["test_labels"]]

    panels = [("TF-IDF", *tfidf_points(data))]
    for method, title in [("mbert", "mBERT"), ("kbbert", "KB-BERT")]:
        panels.append((title, *pca_points(bert_vectors(method, device))))
    panels = [(title, align(points, families), variance)
              for title, points, variance in panels]

    sns.set_theme(style="white", rc={"axes.facecolor": "#fcfcfb",
                                     "figure.facecolor": "#fcfcfb",
                                     "text.color": "#0b0b0b",
                                     "axes.labelcolor": "#52514e",
                                     "xtick.color": "#52514e",
                                     "ytick.color": "#52514e"})

    figure, axes = plt.subplots(1, 3, figsize=(15, 5.2))

    for axis, (title, points, variance) in zip(axes, panels):
        frame = pd.DataFrame({
            "x": points[:, 0],
            "y": points[:, 1],
            "family": families,
            "label": data["test_labels"],
        })
        sns.scatterplot(data=frame, x="x", y="y", hue="family", style="label",
                        palette=COLOURS, hue_order=list(COLOURS),
                        markers=MARKERS, style_order=LABEL_ORDER,
                        s=16, alpha=0.6, edgecolor="#fcfcfb", linewidth=0.3,
                        legend=axis is axes[-1], ax=axis)
        axis.set_title(f"{title}\n{variance.sum():.1%} of the variation shown", fontsize=13)
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_xticks([])
        axis.set_yticks([])
        sns.despine(ax=axis)

    # One legend for all three panels, to the right of the last one.
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    figure.suptitle(f"Test advertisements of fold {FOLD}, as each model represents them",
                    fontsize=14)
    figure.tight_layout()

    for suffix in ("png", "pdf"):
        path = RESULTS_DIR / f"embeddings_fold{FOLD}.{suffix}"
        figure.savefig(path, dpi=200, bbox_inches="tight")
        print("Saved:", path)


if __name__ == "__main__":
    main()
