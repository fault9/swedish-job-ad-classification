import json
from pathlib import Path
from datetime import datetime
from collections import Counter

def load_ads(file_path):
    with file_path.open("r", encoding="utf-8") as file:
        ads = json.load(file)

    return ads

def clean_text(text):
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    return text.strip()

def save_json(data, file_path):
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

def prepare_candidates(data_dir, start_date, end_date):
    seen_ids = set()
    duplicate_id_count = 0
    empty_text_count = 0

    files = sorted(data_dir.rglob("*.json"))

    if not files:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")

    total_ads = 0

    outside_year_count = 0

    text_labels = {}
    text_ad_ids = {}

    duplicate_text_count = 0

    for file_path in files:
        ads = load_ads(file_path)
        total_ads += len(ads)

        for ad in ads:
            publication_date = datetime.fromisoformat(
                ad["publication_date"]
            )

            if not start_date <= publication_date < end_date:
                outside_year_count += 1
                continue
            ad_id = ad["id"]

            if ad_id in seen_ids:
                duplicate_id_count += 1
                continue

            seen_ids.add(ad_id)

            text = ad["description"]["text"]
            cleaned_text = clean_text(text)

            if not cleaned_text:
                empty_text_count += 1
                continue

            label = ad["occupation_group"]["legacy_ams_taxonomy_id"]

            if cleaned_text in text_labels:
                duplicate_text_count += 1
                text_labels[cleaned_text].add(label)
                text_ad_ids[cleaned_text].append(ad_id)
            else:
                text_labels[cleaned_text] = {label}
                text_ad_ids[cleaned_text] = [ad_id]
    conflicting_text_count = 0
    label_counts = Counter()
    candidate_ads = []

    for cleaned_text, labels in text_labels.items():
        if len(labels) > 1:
            conflicting_text_count += 1
            continue

        for label in labels:
            label_counts[label] += 1

            candidate_ads.append({
                "text": cleaned_text,
                "label": label,
                "source_ad_ids": text_ad_ids[cleaned_text],
            })



    print("Files loaded:", len(files))
    print("Total records:", total_ads)
    print("Records outside the selected period:", outside_year_count)
    print("Unique advertisement IDs in the selected period:", len(seen_ids))
    print("Repeated ID occurrences:", duplicate_id_count)
    print("Advertisements with empty text:", empty_text_count)
    print("Repeated text occurrences:", duplicate_text_count)
    print("Distinct nonempty cleaned texts:", len(text_labels))
    print("Texts with conflicting labels:", conflicting_text_count)
    print("Candidate texts per occupation:")

    for label, count in sorted(label_counts.items()):
        print(label, count)

    summary = {
        "period_start_inclusive": start_date.isoformat(),
        "period_end_exclusive": end_date.isoformat(),
        "files_loaded": len(files),
        "raw_records": total_ads,
        "records_outside_year": outside_year_count,
        "repeated_id_occurrences": duplicate_id_count,
        "unique_ids_in_year": len(seen_ids),
        "empty_text_ads": empty_text_count,
        "repeated_text_occurrences": duplicate_text_count,
        "distinct_nonempty_texts": len(text_labels),
        "conflicting_text_groups": conflicting_text_count,
        "candidate_texts_per_label": dict(label_counts),
        "candidate_texts_total": sum(label_counts.values()),
    }
    return candidate_ads, summary

def main():
    year = int(input("Enter the collection year: "))

    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data" / f"collection_{year}"

    start_date = datetime(year, 1, 1)
    end_date = datetime(year + 1, 1, 1)

    candidate_ads, summary = prepare_candidates(
        data_dir,
        start_date,
        end_date,
    )

    report_path = project_dir / "reports" / f"inspection_{year}.json"
    candidate_path = project_dir / "data" / f"candidates_{year}.json"

    save_json(summary, report_path)
    save_json(candidate_ads, candidate_path)

    print("Report saved:", report_path)
    print("Candidate records saved:", len(candidate_ads))
    print("Dataset saved:", candidate_path)


if __name__ == "__main__":
    main()