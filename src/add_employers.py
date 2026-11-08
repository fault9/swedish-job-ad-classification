"""Find out which employer posted each candidate advertisement.

Why this is needed. The candidate records hold only three things: the text,
the label, and the ids of the advertisements the text came from. They do not
hold the employer. But make_splits.py has to keep every employer inside a
single split, so that a model cannot score well by recognising a company's
house style instead of understanding the occupation.

So this script reads the raw downloads again, builds a lookup from
advertisement id to employer, and writes one employer per candidate in the
same order as the candidate file. make_splits.py then reads that list.

It takes a few minutes, because it opens all 15,089 downloaded files. It only
has to be run once per period.
"""

from files import load_json, save_json
from paths import CANDIDATES_FILE, COLLECTION_DIRS, EMPLOYERS_FILE


def employers_by_ad_id(data_dirs):
    """Build a lookup from advertisement id to employer id.

    Each raw advertisement has an "employer" section. We prefer
    organization_number, the Swedish company registration number, because it
    is stable: the same company always has the same number, even if it writes
    its name differently in different advertisements.

    If there is no number we fall back to the name, and if there is no name
    either we invent a unique id from the advertisement id. That last case
    makes each unknown employer count as its own employer, which is the safe
    choice: it never merges two different companies into one.
    """

    employers = {}

    for data_dir in data_dirs:
        for file_path in sorted(data_dir.rglob("*.json")):
            ads = load_json(file_path)

            for ad in ads:
                # "or {}" because the field can be present but null.
                employer = ad.get("employer") or {}
                employers[ad["id"]] = (
                    employer.get("organization_number")
                    or employer.get("name")
                    or f"UNKNOWN_{ad['id']}"
                )

        print("read", data_dir.name, "-", len(employers), "ids so far", flush=True)

    return employers


def candidate_employers(candidates, employers):
    """Give each candidate text exactly one employer.

    A candidate text can come from several advertisement ids, because
    inspect_data.py grouped identical texts together. Those are nearly always
    the same employer reposting the same advertisement, so picking any of them
    gives the same answer.

    We sort and take the first so that the result is the same on every run.
    Picking an arbitrary one would make the splits change between runs, which
    would quietly break reproducibility.
    """

    result = []

    for record in candidates:
        found = sorted(
            employers[ad_id]
            for ad_id in record["source_ad_ids"]
            if ad_id in employers
        )

        result.append(found[0] if found else None)

    return result


def main():
    """Find the employer of every candidate and save the list."""
    candidates = load_json(CANDIDATES_FILE)

    print("candidates:", len(candidates), flush=True)

    employers = employers_by_ad_id(COLLECTION_DIRS)
    per_candidate = candidate_employers(candidates, employers)

    missing = sum(1 for e in per_candidate if e is None)
    print("candidates with no employer found:", missing)
    print("distinct employers across all candidates:", len(set(per_candidate)))

    save_json(per_candidate, EMPLOYERS_FILE)

    print("Employers saved:", EMPLOYERS_FILE)


if __name__ == "__main__":
    main()
