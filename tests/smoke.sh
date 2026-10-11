#!/usr/bin/env bash
# Headless smoke test: builds the demo report and a report full of hostile text, loads both in
# headless Chrome, and fails unless the page renders, nothing from the input executes, and the
# summary carries the fixture figures. Exits non-zero on any failure. No network beyond fonts.
set -u
cd "$(dirname "$0")/.."
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
PY="${PY:-python3}"
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
fail=0; check(){ if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }

$PY -W error -m unittest discover -s tests >"$T/unit.txt" 2>&1
check "unit tests ($(grep -o 'Ran [0-9]* tests' "$T/unit.txt"))" "grep -q '^OK' '$T/unit.txt'"

D=demo/data
$PY -m judgecal report --items $D/items.jsonl --rubric $D/rubric.json --repeats 3 --strata $D/strata.jsonl \
  --replay frontier=$D/frontier.jsonl --replay mid=$D/mid.jsonl --replay small=$D/small.jsonl \
  --fixture --fixture-note "Synthetic demo data from a fixed pattern; not a measurement of any judge." \
  --html "$T/demo.html" --date 2026-10-10 >/dev/null 2>&1
check "demo report written" "test -s '$T/demo.html'"

X='<script>document.title="PWNED"</script><img src=x onerror="document.title=1">'
$PY - "$T" "$X" <<'PYEOF'
import json, sys
d, x = sys.argv[1], sys.argv[2]
open(f"{d}/i.jsonl", "w").write("".join(json.dumps({"id": f"{x}{n}", "text": x}) + "\n" for n in range(3)))
open(f"{d}/j.jsonl", "w").write("".join(json.dumps({"item": f"{x}{n}", "repeat": r, "class": x if (n + r) % 2 else "Q",
                                                    "confidence": x}) + "\n" for n in range(3) for r in range(2)))
PYEOF
$PY -m judgecal report --items "$T/i.jsonl" --replay "evil=$T/j.jsonl" --title "$X" --notes "$X" \
  --source "$X" --html "$T/hostile.html" >/dev/null 2>&1
check "hostile report written" "test -s '$T/hostile.html'"

dom(){ "$CHROME" --headless=new --disable-gpu --virtual-time-budget=3000 --dump-dom "file://$1" 2>/dev/null; }
dom "$T/demo.html" >"$T/demo.dom"
check "demo renders with its title" "grep -q '<title>Judge Calibrator Report</title>' '$T/demo.dom'"
check "summary shows band and synthetic stamp" "grep -q '±0.04' '$T/demo.dom' && grep -q 'Synthetic demo data' '$T/demo.dom'"
check "no study figures in the demo" "! grep -q '/197' '$T/demo.dom'"
check "table wrappers are focusable regions" "grep -q 'class=\"tablewrap wide\" tabindex=\"0\" role=\"region\"' '$T/demo.dom'"
check "print stylesheet present" "grep -q '@media print' '$T/demo.dom'"
dom "$T/hostile.html" >"$T/hostile.dom"
check "hostile page title untouched" "grep -q '<title>Judge Calibrator Report</title>' '$T/hostile.dom'"
check "no script element in hostile DOM" "! grep -q '<script' '$T/hostile.dom'"
check "no img element in hostile DOM" "! grep -q '<img' '$T/hostile.dom'"
check "payload shown as text" "grep -q '&lt;script&gt;' '$T/hostile.dom'"
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer --print-to-pdf="$T/r.pdf" "file://$T/demo.html" 2>/dev/null
check "prints to PDF" "test -s '$T/r.pdf'"
exit $fail
