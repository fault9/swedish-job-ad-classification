import json
import random
from pathlib import Path

from check_language import SHARE_BANDS, WORD_BANDS, share_bucket, word_bucket

from paths import PROJECT_DIR

def group_positions_by_band(annotations, bucket_of, field):
    """Return {band_name: [positions]}, using bucket_of on the given field."""

    positions_by_band = {}

    for position, annotation in enumerate(annotations):
        band = bucket_of(annotation[field])
        if band not in positions_by_band:
            positions_by_band[band] = []
        positions_by_band[band].append(position)

    return positions_by_band

def select_review_sample(annotations, bucket_of, field, bands, per_band=8, seed=42):
    """Pick a reproducible sample of positions from each band, in band order."""

    positions_by_band = group_positions_by_band(annotations, bucket_of, field)
    generator = random.Random(seed)

    selected = {}

    for band in bands:
        positions = positions_by_band.get(band, [])
        count = min(per_band, len(positions))
        selected[band] = generator.sample(positions, count)

    return selected

def show_case(position, record, annotation):
    """Print one review case for manual reading."""
    print("Position:", position)
    print("Label:", annotation["label"])
    print("Dominant language:", annotation["dominant_language"])
    print("Swedish share:", round(annotation["swedish_share"], 3))
    print("Number of words:", annotation["n_words"])
    print("Span count:", annotation["span_count"])
    print("Languages:", annotation["n_words_per_lang"])
    print("Ad id:", annotation["source_ad_ids"][0])
    print(" ".join(record["text"].split())[:1200])
    print("-" * 70)


def main():
    period = input("Enter the period (for example 2023_2025): ").strip()

    project_dir = PROJECT_DIR

    with (project_dir / "data" / f"candidates_{period}.json").open(encoding="utf-8") as file:
        candidates = json.load(file)

    with (project_dir / "data" / f"language_{period}.json").open(encoding="utf-8") as file:
        annotations = json.load(file)

    sections = [
        ("SWEDISH SHARE", share_bucket, "swedish_share", SHARE_BANDS),
        ("WORD COUNT", word_bucket, "n_words", WORD_BANDS),
    ]

    for title, bucket_of, field, bands in sections:
        selected = select_review_sample(annotations, bucket_of, field, bands)

        for band in bands:
            print("=" * 70)
            print(title, "BAND", band, "-", len(selected[band]), "cases")
            print("=" * 70)

            for position in selected[band]:
                show_case(position, candidates[position], annotations[position])


if __name__ == "__main__":
    main()