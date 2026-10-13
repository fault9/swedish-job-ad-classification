import json
import truststore
from urllib.request import urlopen
from pathlib import Path
from datetime import datetime, timedelta

from paths import PROJECT_DIR

truststore.inject_into_ssl()      # use the macOS keychain for HTTPS

MAX_OFFSET = 2000 # the API won't go higher than this
WINDOW_DAYS = 1

OCCUPATIONS = {
    # IT
    "2511": "UXKZ_3zZ_ipB",
    "2512": "DJh5_yyF_hEM",
    "2514": "D9SL_mtn_vGM",

    # Education
    "2341": "oQUQ_D11_HPx",
    "2330": "4KhP_FxL_uZ5",
    "5312": "XW9L_tA9_WaX",

    # Healthcare
    "2221": "Z8ci_bBE_tmx",
    "5321": "jY19_knH_MJp",
    "5323": "LP4Z_ABV_gRm",
}

def fetch_ads(url):
    with urlopen(url, timeout=30) as response:
        data = json.load(response)
    
    return data


def build_url(concept_id, start_date, end_date, offset=0):
    url = (
        "https://historical.api.jobtechdev.se/search"
        f"?occupation-group={concept_id}"
        "&limit=100" # change limit for more or less
        "&sort=id"
        f"&offset={offset}" # to read ads in chunks
        f"&published-after={start_date.isoformat()}"
        f"&published-before={end_date.isoformat()}"
    )
    return url

def date_windows(year, window_days=WINDOW_DAYS):
    """Return the windows covering one year, as (start, end) pairs."""
    windows = []

    start_date = datetime(year, 1, 1)
    year_end = datetime(year + 1, 1, 1)

    while start_date < year_end:
        end_date = min(start_date + timedelta(days=window_days), year_end)
        windows.append((start_date, end_date))
        start_date = end_date

    return windows

def window_dir(project_dir, start_date, end_date):
    """Return the folder for one window, inside its year's collection."""
    name = f"{start_date.date()}_{end_date.date()}"

    return project_dir / "data" / f"collection_{start_date.year}"/name

def save_ads(ads, output_path):
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(ads, file, ensure_ascii=False, indent=2)

def download_code(code, concept_id, start_date, end_date, output_dir):
    """Download every page for one occupation group in one window."""
    first = fetch_ads(build_url(concept_id, start_date, end_date, 0))
    count = first["total"]["value"]

    if count > MAX_OFFSET:
        raise ValueError(
            f"{code} {start_date.date()}: {count} records exceeds the "
            f"offset cap of {MAX_OFFSET}. Use a smaller window."
        )

    save_ads(first["hits"], output_dir / f"{code}_offset_0.json")

    for offset in range(100, count, 100):
        page = fetch_ads(build_url(concept_id, start_date, end_date, offset))
        ads = page["hits"]

        expected = min(100, count - offset)

        if len(ads) != expected:
            raise ValueError(
                f"{code} {start_date.date()} offset {offset}: "
                f"expected {expected} ads, got {len(ads)}"
            )

        save_ads(ads, output_dir / f"{code}_offset_{offset}.json")

    return count

def download_window(project_dir, start_date, end_date):
    """Download all occupation groups for one window."""
    output_dir = window_dir(project_dir, start_date, end_date)
    output_dir.mkdir(parents=True, exist_ok=True)

    for code, concept_id in OCCUPATIONS.items():
        done_path = output_dir / f"{code}.done"

        if done_path.exists():
            continue

        count = download_code(code, concept_id, start_date, end_date, output_dir)

        done_path.write_text("")

        print(start_date.date(), code, count)


def main():
    first_year = int(input("Enter the first year: "))
    last_year = int(input("Enter the last year: "))

    project_dir = PROJECT_DIR

    for year in range(first_year, last_year + 1):
        for start_date, end_date in date_windows(year):
            download_window(project_dir, start_date, end_date)


if __name__ == "__main__":
    main()

