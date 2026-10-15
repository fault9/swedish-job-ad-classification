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

WHICH 200 words matters enormously. The default is "head", the first 200 words,
and that was not the first choice. The first choice was "head_tail", half from
each end, taken from Sun et al. (2019) who found it best on film reviews and
news. It does not transfer: Swedish job advertisements are front-loaded, and
84.3% of final-100-word chunks are application instructions, URLs, collective
agreement text and recruiter disclaimers, against 12.2% of opening chunks.

Measured with the same TF-IDF model, validation macro F1 at a 200-word budget:

    head           0.7549      first 200 words
    head_tail_90   0.7192      180 and 20
    head_tail_75   0.6814      150 and 50
    head_tail      0.5796      100 and 100, the original choice
    gated          0.7586      boilerplate sentences dropped, then first 200
    (untruncated)  0.8274      the ceiling, if length were free

Every word spent on the tail costs accuracy. "head" recovers 91% of the
untruncated ceiling; "head_tail" recovered 70%. The "gated" variant wins by
0.004, which is inside the noise, so it is not worth a hand-written list of
Swedish phrases to defend. Plain "head" it is.

Nothing here touches the test splits unless you ask for them by name.
"""

import json
import re
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


# Sentences matching this are administrative rather than about the job: how to
# apply, contact details, the company's certifications and values statements.
# Measured on the train split, these markers appear in 84.3% of final-100-word
# chunks against 12.2% of opening chunks, which is why keeping the tail of an
# advertisement is a poor use of the budget.
BOILERPLATE = re.compile(
    r"www\.|http|@|"
    r"ansök|ansöker|ansökan|urval sker|intervjuer sker|sista ansökningsdag|"
    r"kollektivavtal|ISO-certifi|auktoriserat företag|"
    r"jämställdhetsplan|mångfald och jämlikhet|bakgrundskontroll|"
    r"vi undanber|rekryteringsföretag|annonsförsälj|"
    r"varmt välkommen med din|skicka din ansökan|registerkontroll",
    re.IGNORECASE,
)

# A wider net. Looking at what survived BOILERPLATE at the end of
# advertisements showed four whole categories it missed: employment conditions,
# benefits lists, diversity statements and closing greetings. None of them say
# anything about which occupation the advertisement is for.
#
# This is hand-written and therefore arbitrary, which is why it is measured
# rather than assumed. It will sometimes delete a real sentence.
BOILERPLATE_WIDE = re.compile(
    r"www\.|http|@|"
    # applying
    r"ansök|ansöker|ansökan|urval sker|intervjuer sker|sista ansökningsdag|"
    r"varmt välkommen med din|skicka din ansökan|"
    # conditions of employment
    r"kollektivavtal|ISO-certifi|auktoriserat företag|registerkontroll|"
    r"belastningsregistret|arbetstillstånd|identitetskort|medborgarskap|"
    r"upphandlade avtal|bakgrundskontroll|e-legitimation|"
    r"tillträde enligt|enligt överenskommelse|provanställning|"
    # benefits
    r"friskvård|förmånsportal|pensionsrådgivning|sjukvårdsförsäkring|"
    r"förmåner|tjänstepension|personalrabatt|"
    # values and diversity statements
    r"jämställdhet|mångfald|öppen för alla|diskriminer|"
    r"vi fokuserar på din kompetens|lika villkor|"
    # recruiters and sign-offs
    r"vi undanber|rekryteringsföretag|annonsförsälj|bemanningsföretag|"
    r"välkommen till oss|hoppas vi hörs|låter det intressant",
    re.IGNORECASE,
)

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def drop_boilerplate(text, pattern=BOILERPLATE):
    """Remove the sentences that are not about the job itself.

    Splits on sentence punctuation and newlines, drops any piece matching the
    pattern, and rejoins. If that would remove everything, the original is
    returned instead, so a text can never become empty.

    Pass BOILERPLATE for the narrow rule, which catches how-to-apply text, or
    BOILERPLATE_WIDE for the broad one, which also catches employment
    conditions, benefits, diversity statements and sign-offs.

    Both are blunt rules and will sometimes remove a real sentence that happens
    to sit next to an application link. They are here so the effect can be
    measured rather than assumed.
    """
    pieces = [p for p in SENTENCE_SPLIT.split(text) if p.strip()]
    kept = [p for p in pieces if not pattern.search(p)]

    if not kept:
        return text

    return " ".join(kept)


def split_words(text, n_words, head_share):
    """Keep n_words, taking head_share of them from the start and the rest from the end.

    head_share of 1.0 is head-only, 0.5 is the even head+tail split.
    """
    words = text.split()

    if len(words) <= n_words:
        return " ".join(words)

    head = int(round(n_words * head_share))
    tail = n_words - head

    if tail <= 0:
        return " ".join(words[:n_words])

    return " ".join(words[:head] + words[-tail:])


def truncate_text(text, budget, strategy):
    """Cut a text down to the budget using the named strategy.

    "none"        keep everything
    "head"        first budget words
    "head_tail"   half from the start, half from the end
    "head_tail_N" N percent from the start, the rest from the end
    "gated"       drop boilerplate sentences first, then take the first words
    "gated_tail"  drop boilerplate first, then 75 percent head and 25 percent tail
    """
    if strategy == "none":
        return text
    if strategy == "head":
        return first_words(text, budget)
    if strategy == "head_tail":
        return head_tail_words(text, budget)
    if strategy.startswith("head_tail_"):
        return split_words(text, budget, int(strategy.split("_")[-1]) / 100)
    if strategy == "gated":
        return first_words(drop_boilerplate(text), budget)
    if strategy == "gated_tail":
        return split_words(drop_boilerplate(text), budget, 0.75)
    if strategy == "gated_wide":
        return first_words(drop_boilerplate(text, BOILERPLATE_WIDE), budget)
    if strategy == "gated_wide_tail":
        return split_words(drop_boilerplate(text, BOILERPLATE_WIDE), budget, 0.75)

    raise ValueError(f"Unknown truncation strategy: {strategy}")


def load_split(project_dir, period, name, budget=WORD_BUDGET, truncate=True,
               strategy="head"):
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
        text = truncate_text(text, budget, strategy if truncate else "none")
        texts.append(text)
        # Take the label from the splits, not the candidate, so the two can
        # never disagree about what this example is.
        labels.append(unit["label"])

    return texts, labels


def load_units(project_dir, period, name):
    """Return the split's unit records: ad_id, cluster, employer, label, position.

    load_split gives the texts and labels needed for training. This gives the
    identifiers needed afterwards, in the same order, so a prediction can be
    traced back to the advertisement it came from.

    Needed for two analyses the project depends on: whether the three methods
    fail on the same advertisements, and whether their errors sit in the
    off-type occupations inside a label.
    """
    data_dir = Path(project_dir) / "data"

    with (data_dir / f"splits_{period}.json").open(encoding="utf-8") as file:
        splits = json.load(file)

    if name == "test_nonsv":
        return splits["test_nonsv"]

    return splits["splits"][name]


def load_fold(project_dir, period, fold_number, budget=WORD_BUDGET,
              strategy="head"):
    """Load one cross-validation fold: its test part and its training part.

    Returns four lists: test texts, test labels, train texts, train labels.

    "Test" is the one fold asked for. "Train" is every OTHER fold joined
    together. With five folds that means 2,880 test and 11,520 train, and
    because the folds are employer-disjoint no employer appears on both sides.

    The units are returned as well, so predictions can be traced back to
    advertisements afterwards.
    """
    data_dir = Path(project_dir) / "data"

    with (data_dir / f"candidates_{period}.json").open(encoding="utf-8") as file:
        candidates = json.load(file)

    with (data_dir / f"folds_{period}.json").open(encoding="utf-8") as file:
        saved = json.load(file)

    if len(candidates) != saved["candidate_count"]:
        raise ValueError(
            f"Candidate file has {len(candidates):,} records but the folds were "
            f"built from {saved['candidate_count']:,}. Rebuild the folds."
        )

    def to_texts(units):
        texts = []
        for unit in units:
            # Spot-check that the position still means what it did.
            actual = candidates[unit["position"]]["source_ad_ids"][0]
            if actual != unit["ad_id"]:
                raise ValueError(
                    f"Position {unit['position']} should be advertisement "
                    f"{unit['ad_id']} but the candidate file has {actual}."
                )
            texts.append(truncate_text(candidates[unit["position"]]["text"],
                                       budget, strategy))
        return texts

    test_units = saved["folds"][str(fold_number)]

    train_units = []
    for number, units in saved["folds"].items():
        if number != str(fold_number):
            train_units += units

    return {
        "test_texts": to_texts(test_units),
        "test_labels": [u["label"] for u in test_units],
        "test_units": test_units,
        "train_texts": to_texts(train_units),
        "train_labels": [u["label"] for u in train_units],
        "n_folds": saved["n_folds"],
    }


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
