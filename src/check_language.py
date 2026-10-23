from collections import Counter, defaultdict

from lingua import LanguageDetectorBuilder

from files import load_json, save_json
from paths import CANDIDATES_FILE, LANGUAGE_FILE, PERIOD, REPORTS_DIR

def summarize_spans(spans): 
    """Turn one texts detected spans into summary dictionary of language shares."""

    if not spans:
        return {
            "n_words": 0,
            "span_count": 0,
            "swedish_share": 0.0,
            "dominant_language": None,
            "dominant_share": 0.0,
            "n_words_per_lang": {},
        }
    n_words_per_lang = Counter()

    for span in spans:
        n_words_per_lang[span.language.name] += span.word_count

    n_words = sum(n_words_per_lang.values())
    swedish_share = n_words_per_lang["SWEDISH"]/n_words
    dominant_language, dominant_count = n_words_per_lang.most_common(1)[0] 
    dominant_share = dominant_count/n_words

    return {
        "n_words": n_words,
        "span_count": len(spans),
        "swedish_share": swedish_share,
        "dominant_language": dominant_language,
        "dominant_share": dominant_share,
        "n_words_per_lang": dict(n_words_per_lang),
    }


def annotate_candidates(candidates, detector, batch_size=10000):
    """Detect languages for all candidate texts and return one annotation per text."""

    annotations = []

    # In batches so that only one batch of span results sits in memory at a time,
    # and so a long run prints progress instead of going silent.
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start:start + batch_size]

        texts = [record["text"] for record in batch]

        results = detector.detect_multiple_languages_in_parallel_of(texts)

        for record, spans in zip(batch, results):
            summary = summarize_spans(spans)
            annotation = {
                "label": record["label"],
                "source_ad_ids": record["source_ad_ids"],
                **summary,
            }
            annotations.append(annotation)

        print("annotated", len(annotations), "of", len(candidates), flush=True)

    return annotations


def share_bucket(share):
    """Return the name of the swedish_share band that this value falls in."""
    if share == 0.0:
        return "0.00"
    elif share < 0.50:
        return "0.00-0.50"
    elif share < 0.80:
        return "0.50-0.80"
    elif share < 0.95:
        return "0.80-0.95"
    elif share < 1.0:
        return "0.95-1.00"
    else:
        return "1.00"


def word_bucket(n_words):
    """Return the name of the word-count band that this value falls in."""
    if n_words < 10:
        return "<10"
    elif n_words < 50:
        return "10-49"
    elif n_words < 100:
        return "50-99"
    elif n_words < 200:
        return "100-199"
    elif n_words < 400:
        return "200-399"
    else:
        return "400+"


def nest_counts(flat_counts):
    """Turn {(outer, inner): count} into {outer: {inner: count}}."""

    result = defaultdict(dict)

    for (outer, inner), count in flat_counts.items():
        result[outer][inner] = count

    # Hand back a plain dict, so nothing downstream can accidentally create
    # entries just by looking for a label that is not there.
    return dict(result)

def build_report(annotations):
    """Count the swedish_share bands and the dominant languages"""
    bucket_counts = Counter()
    dominant_counts = Counter()
    label_bucket_counts = Counter()
    label_dominant_counts = Counter()
    word_counts = Counter()
    label_word_counts = Counter()

    for annotation in annotations:
        label = annotation["label"]
        language = annotation["dominant_language"]
        bucket = share_bucket(annotation["swedish_share"])
        bucket_counts[bucket] += 1
        dominant_counts[language] += 1
        label_bucket_counts[(label, bucket)] += 1
        label_dominant_counts[(label, language)] += 1
        words = word_bucket(annotation["n_words"])
        word_counts[words] += 1
        label_word_counts[(label, words)] += 1

    return {
        "n_texts": len(annotations),
        "bucket_counts": dict(bucket_counts),
        "dominant_counts": dict(dominant_counts),
        "label_bucket_counts": nest_counts(label_bucket_counts),
        "label_dominant_counts": nest_counts(label_dominant_counts),
        "word_counts": dict(word_counts),
        "label_word_counts": nest_counts(label_word_counts),
    }


def main():
    report_path = REPORTS_DIR / f"language_{PERIOD}.json"

    if not CANDIDATES_FILE.exists():
        raise FileNotFoundError(f"No candidate file at {CANDIDATES_FILE}")

    if LANGUAGE_FILE.exists():
        annotations = load_json(LANGUAGE_FILE)

        print("Existing annotations loaded:", len(annotations))
    else:
        candidates = load_json(CANDIDATES_FILE)

        print("Candidate records loaded:", len(candidates))

        detector = LanguageDetectorBuilder.from_all_languages().build()

        annotations = annotate_candidates(candidates, detector)

        save_json(annotations, LANGUAGE_FILE)

        print("Annotations saved:", LANGUAGE_FILE)

    report = build_report(annotations)
    save_json(report, report_path)

    print("Report saved:", report_path)


if __name__ == "__main__":
    main()

