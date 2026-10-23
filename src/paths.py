"""Where the project's folders and data files are.

PROJECT_DIR is two levels up from this file: src/paths.py -> src/ -> project.
"""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"
REPORTS_DIR = PROJECT_DIR / "reports"

# Each version keeps its results in its own folder. v1 read the first 200
# words, v2 skips the employer introduction.
VERSION = "v2"
RESULTS_DIR = REPORTS_DIR / VERSION

FIRST_YEAR = 2023
LAST_YEAR = 2026
PERIOD = f"{FIRST_YEAR}_{LAST_YEAR}"

COLLECTION_DIRS = [DATA_DIR / f"collection_{year}"
                   for year in range(FIRST_YEAR, LAST_YEAR + 1)]
CANDIDATES_FILE = DATA_DIR / f"candidates_{PERIOD}.json"
LANGUAGE_FILE = DATA_DIR / f"language_{PERIOD}.json"
EMPLOYERS_FILE = DATA_DIR / f"employers_{PERIOD}.json"
CLUSTERS_FILE = DATA_DIR / f"clusters_{PERIOD}.json"
SPLITS_FILE = DATA_DIR / f"splits_{PERIOD}.json"
FOLDS_FILE = DATA_DIR / f"folds_{PERIOD}.json"
