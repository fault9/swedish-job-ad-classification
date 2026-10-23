# SSYK occupational classification of Swedish job advertisements

Comparing three methods for predicting an occupational category from the body
text of a Swedish job advertisement:

1. **TF-IDF with logistic regression**
2. **KB Swedish BERT** — `KB/bert-base-swedish-cased`
3. **Multilingual BERT** — `google-bert/bert-base-multilingual-cased`

One nine-label classifier per method, trained on identical data and identical
splits. The labels are SSYK occupational groups in three families of three, so
that errors can be separated into fine confusions within a family and gross
confusions across families.

```
IT          2511  Systemanalytiker och IT-arkitekter m.fl.
            2512  Mjukvaru- och systemutvecklare m.fl.
            2514  Systemtestare och testledare

Education   2341  Grundskollärare
            2330  Gymnasielärare
            5312  Elevassistenter m.fl.

Healthcare  2221  Grundutbildade sjuksköterskor
            5321  Undersköterskor, hemtjänst, hemsjukvård, äldreboende
            5323  Undersköterskor, vård- och specialavdelning och mottagning
```

Family membership is used only for error analysis, never as a label.

## Layout

```
src/        all the Python
scripts/    shell wrappers for long runs
data/       downloads and derived datasets      (not in git)
reports/    summaries and results               (not in git)
docs/       decisions, sources, project status  (not in git)
```

`data/`, `reports/` and `docs/` are excluded from version control because they
contain advertisement text. Everything in them is either reproducible from the
scripts or backed up to GU OneDrive.

## Pipeline, in order

Each stage reads what the stage before it wrote. Run them from the project root.

| # | Script | Does | Takes |
|---|---|---|---|
| 1 | `src/read_data.py` | Download advertisements from the JobTech historical API | ~75 min |
| 2 | `src/inspect_data.py` | Filter by date, deduplicate, build candidate texts | ~4 min |
| 3 | `src/check_language.py` | Annotate the language of every text | ~110 min |
| 4 | `src/add_employers.py` | Map each text to the employer that posted it | ~4 min |
| 5 | `src/find_duplicates.py` | Cluster near-duplicates with MinHash and LSH | ~30 min |
| 6 | `src/make_splits.py` | Build the frozen train, validation and test splits | seconds |

Stages 3, 5 and a token measurement can be chained:

```bash
./scripts/run_chain.sh
```

## Modelling

| Script | Does |
|---|---|
| `src/dataset.py` | Loads a split and truncates it to the word budget. Shared by all three methods, so they all see identical text. |
| `src/evaluate.py` | Scores predictions the same way for every method. |
| `src/train_tfidf.py` | Method 1. |

## Analysis

| Script | Does |
|---|---|
| `src/measure_tokens.py` | How many tokens each BERT needs per Swedish word, and the word budget that fits both. |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`src/paths.py` holds the project folder locations, so no script has to work
them out for itself.

## Corpus as it stands

```
454,852  advertisements downloaded, 2023-01-01 to 2026-10-12
305,869  candidate texts after deduplication
 15,843  texts in the frozen splits
```

Per label: 1,170 train, 253 validation, 253 test, plus a 759-text
Swedish-minority test set covering the three IT labels.

The splits are employer-disjoint: no employer appears in two splits, and no
employer contributes more than 20 advertisements to a label. The reasoning is
in `docs/status.md`.

## Reading the documentation

`docs/` is not in git but is the record of why the project looks like this.

- `status.md` — every decision with its reason, the measurements behind it, and
  the next actions. **Start here.**
- `label-selection.md` — the protocol used to choose the nine labels, written
  before the measurements were taken.
- `sources.md` — tools and data sources, with verbatim quotes and versions.
