import json
from urllib.request import urlopen
from pathlib import Path
from datetime import datetime, timedelta


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
        f"&published-after={start_date}"
        f"&published-before={end_date}"
    )
    return url

def download_window(start_date, end_date):
    window_name = f"{start_date[:10]}_{end_date[:10]}"
    output_dir = (
        Path(__file__).resolve().parent
        / "data"
        / "collection_2025"
        / window_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    offset = 0
    for code, concept_id in occupations.items():
        url = build_url(concept_id, start_date, end_date, offset)
        data = fetch_ads(url)
        count = data["total"]["value"]

        ads = data["hits"]
        
        if len(ads) != min(100, count):
            raise ValueError("Unexpected first batch size. Check the download.")

        if ads:
            name = ads[0]["occupation_group"]["label"]
        else:
            name = "No matching ads"

        print(code, name, count)

        output_path = output_dir / f"{code}_offset_{offset}.json"

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(ads, file, ensure_ascii=False, indent=2)
        
        print("Saved:", len(ads), "ads to", output_path)

        if count > 2000:
            raise ValueError("Too many matches. Use a smaller date window.")

        for page_offset in range(100, count, 100):
            url = build_url(concept_id, start_date, end_date, page_offset)
            page = fetch_ads(url)
            page_ads = page["hits"]

            expected = min(100, count - page_offset)

            if page["total"]["value"] != count or len(page_ads) != expected:
                raise ValueError("Unexpected response. Check the download.")

            output_path = output_dir / f"{code}_offset_{page_offset}.json"

            with output_path.open("w", encoding="utf-8") as file:
                json.dump(page_ads, file, ensure_ascii=False, indent=2)

            print("Saved:", len(page_ads), "ads to", output_path)


occupations = {
    "2511": "UXKZ_3zZ_ipB",
    "2512": "DJh5_yyF_hEM",
    "2514": "D9SL_mtn_vGM",
    "2341": "oQUQ_D11_HPx",
    "2342": "CFW8_eBa_NaV",
    "2343": "5ek3_Cgq_WeZ",
    "2224": "oj4c_P4b_cja",
    "2225": "bEDv_SL7_VrL",
    "2228": "dyWo_sDb_MLQ",
}

#output_dir = Path(__file__).resolve().parent / "data" / "raw"
current_start = datetime(2025, 1, 1)
year_end = datetime(2026, 1, 1)

while current_start < year_end:
    current_end = min(
        current_start + timedelta(days=7),
        year_end
    )

    download_window(
        current_start.isoformat(),
        current_end.isoformat()
    )

    current_start = current_end
