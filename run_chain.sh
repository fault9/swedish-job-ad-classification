#!/bin/bash
# Run the three remaining preprocessing stages in order.
# Each only starts if the previous one exits cleanly.
set -e
cd "$(dirname "$0")"
P=2023_2026
echo "=== check_language.py $(date '+%H:%M:%S') ==="
printf "$P\n" | .venv/bin/python3 -u check_language.py
echo "=== find_duplicates.py $(date '+%H:%M:%S') ==="
printf "$P\n" | .venv/bin/python3 -u find_duplicates.py
echo "=== measure_tokens.py $(date '+%H:%M:%S') ==="
printf "$P\n" | .venv/bin/python3 -u measure_tokens.py
echo "=== CHAIN COMPLETE $(date '+%H:%M:%S') ==="
