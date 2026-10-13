#!/bin/bash
# Run the three slow preprocessing stages in order.
# Each only starts if the previous one exits cleanly.
set -e

# This script lives in scripts/, so the project root is one level up.
cd "$(dirname "$0")/.."

P=2023_2026

echo "=== check_language.py $(date '+%H:%M:%S') ==="
printf "$P\n" | .venv/bin/python3 -u src/check_language.py

echo "=== find_duplicates.py $(date '+%H:%M:%S') ==="
printf "$P\n" | .venv/bin/python3 -u src/find_duplicates.py

echo "=== measure_tokens.py $(date '+%H:%M:%S') ==="
printf "$P\n" | .venv/bin/python3 -u src/measure_tokens.py

echo "=== CHAIN COMPLETE $(date '+%H:%M:%S') ==="
