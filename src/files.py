"""Read and write JSON files, always as UTF-8 so å, ä and ö survive on Windows."""

import json


def load_json(path):
    """Read one JSON file and return its content."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data, path):
    """Write data to a JSON file, creating its folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
