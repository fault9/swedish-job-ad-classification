# Occupational classification of Swedish job advertisements

TF-IDF with logistic regression, KB-BERT and multilingual BERT, compared on
predicting nine SSYK occupational groups from the text of Swedish job
advertisements from the JobTech historical API (2023 to 2026).

## Setup

```
python -m venv .venv
pip install -r requirements.txt
```

For the BERT models, install PyTorch for your CUDA version first, as described
at the top of `requirements-train.txt`.

## How to run

From this folder, in this order:

```
python src/read_data.py          # download the advertisements (about 75 min)
python src/inspect_data.py       # deduplicate, build candidate texts
python src/check_language.py     # detect the language of every text
python src/add_employers.py      # find the employer of every text
python src/find_duplicates.py    # cluster near-duplicates (MinHash, LSH)
python src/make_splits.py        # validation split for choosing settings
python src/train_tfidf.py        # choose the TF-IDF settings
python src/make_folds.py         # the five folds, 14,400 advertisements
python src/train_tfidf_cv.py     # TF-IDF on the folds
python src/train_bert.py         # KB-BERT and mBERT, three seeds (GPU)
python src/compare_runs.py       # paired comparisons (H1, H2)
python src/error_overlap.py      # shared errors
python src/plot_confusion.py     # confusion matrices
```

Results are saved in `reports/v2/`. The scores, comparisons and figures are in
the repository. The prediction files per advertisement are not, because some
employers are listed by a person's name.

## Data and models

The advertisement texts are not in the repository, because they come from real
advertisements. `read_data.py` downloads them again, but the API can change, so a new download
may differ slightly. The fine-tuned models were
not saved; `train_bert.py` retrains them.
