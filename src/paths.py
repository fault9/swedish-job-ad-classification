"""Where the project's folders are.

Every script needs to find data/ and reports/, and they all used to work it out
themselves with Path(__file__).resolve().parent. That broke the moment the
scripts moved into src/, because then "the folder this file is in" is src/, not
the project root.

So the answer lives here once. PROJECT_DIR goes up two levels: from this file,
to src/, to the project root.

    src/paths.py  ->  src/  ->  ssyk-classification/
"""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"
REPORTS_DIR = PROJECT_DIR / "reports"
DOCS_DIR = PROJECT_DIR / "docs"

# Each version is one complete set of experiment results in its own folder,
# so a new experiment never overwrites an old one. Change this to start a
# new version. v1 is the first 200 words of each advertisement.
VERSION = "v1"
RESULTS_DIR = REPORTS_DIR / VERSION
