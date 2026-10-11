#!/usr/bin/env bash
# Rebuilds demo/report.html and demo/drift.html from the synthetic data in demo/data/.
set -eu
cd "$(dirname "$0")/.."
D=demo/data
NOTE="Synthetic demo data from a fixed pattern; not a measurement of any judge."
python3 -m judgecal report --items $D/items.jsonl --rubric $D/rubric.json \
  --replay frontier=$D/frontier.jsonl --replay mid=$D/mid.jsonl --replay small=$D/small.jsonl \
  --repeats 3 --strata $D/strata.jsonl --fixture --fixture-note "$NOTE" \
  --source "demo/data/, written by demo/data/make_synthetic.py" \
  --title "Judge consistency report (synthetic demo)" --date 2026-10-10 --html demo/report.html >/dev/null
python3 -m judgecal compare $D/verdicts-2026-09-28.jsonl $D/verdicts-2026-10-05.jsonl --rubric $D/rubric.json \
  --title "Judge drift report (synthetic demo)" --source "$NOTE demo/data/, written by demo/data/make_synthetic.py" \
  --date 2026-10-10 --html demo/drift.html >/dev/null
echo "wrote demo/report.html and demo/drift.html"
