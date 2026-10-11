judgecal runs model judges over a set of items repeatedly and reports test-retest agreement, pairwise kappa and a judge-noise band to quote beside any judged benchmark number.

# The Judge Calibrator

**Status:** version 0.2.x, first public release 11 October 2026.

Lab page: https://dinand.com/experiments/judge-calibrator/

Source: https://github.com/dtinholt/judge-calibrator

License: Apache-2.0.

`judgecal` runs one or more model judges over a set of items, twice or more, and tells you
how often each judge agrees with its own earlier run and with the other judges. It prints
the judge-noise band to quote beside any judged benchmark number, and writes a one-page
HTML report in the lab's ledger style.

The method and the maths come from a registered judge-reliability study run in October
2026, not yet public: raw agreement with Wilson 95% intervals, Cohen's κ, confusion tables,
and a noise band set at half the reference judge's observed test-retest disagreement.

Python 3.9 or later, standard library only. The tests pass on 3.9.6 and 3.14.5.

judgecal reports how consistent judges are with themselves and with each other. Scoring a
judge's accuracy takes a ground truth to compare against, and the tool has no input for one.

## Run the demo

```
bash demo/build.sh
```

`demo/report.html` and `demo/drift.html` are built from synthetic data in `demo/data/`,
written by `demo/data/make_synthetic.py` from a fixed pattern and a seeded shuffle. No model
judged anything there, and both pages carry a stamped label saying so. The three "judges"
are named frontier, mid and small only to show how tiers read on the page.

Some regression tests replay verdicts from a judge-reliability study that is not yet public.
Its fixture is not part of this repository, so those tests skip themselves when it is absent.
The rest of the suite runs on the synthetic data and the hand-written fixtures.

## Point it at a judge endpoint

`ChatCompletionsJudge` works with any endpoint that speaks the chat-completions protocol
(`POST {base_url}/chat/completions`), hosted or local.

1. Write a rubric file:

   ```json
   {"name": "pass-fail",
    "classes": ["PASS", "FAIL"],
    "instruction": "Decide whether the evidence supports the claim.",
    "descriptions": {"PASS": "evidence is sufficient",
                     "FAIL": "evidence is missing or wrong"}}
   ```

   The judge gets the instruction, the class list and an answer format asking for
   `CLASS: <...>` and `CONFIDENCE: <0-100>` lines. Override `answer_format` in the file if
   your judge needs something else; JSON replies with `class` and `confidence` keys parse
   too.

2. Write the items, one JSON object per line, with an `id` and a `text` (or `input`, or
   `prompt`). Any other keys stay on your side of the wire.

3. Put the key in an environment variable and run:

   ```
   export JUDGE_API_KEY=...
   python3 -m judgecal report --items items.jsonl --rubric rubric.json \
     --endpoint big   https://api.example/v1   big-model-name  JUDGE_API_KEY \
     --endpoint cheap http://localhost:8000/v1 small-model     - \
     --repeats 2 --html report.html --verdicts-out verdicts.jsonl
   ```

   `JUDGE_API_KEY` is also the default variable name in Python. `-` in the key slot means
   no key. Each repeat is a fresh request. `--temperature` sets the sampling temperature
   for every endpoint judge; leave it unset to measure the judge as you would run it.
   `--verdicts-out` saves every verdict with its token counts and the run date, so you can
   replay the run later with `ReplayJudge(path, name, judge_field=name)` at no cost, or
   compare it with a later run (see Drift below).

   Each endpoint judge sends up to four requests at once (`--workers`, default 4) and puts
   the verdicts back in item order. `--item-timeout` (default 300 s) caps the time one item
   may take, retries and backoff included; an item that runs out of time is recorded as
   unreadable. Add `--price big=3,15` to price a judge in US dollars per million input and
   output tokens; the report then prints what the run cost from the token counts the endpoint
   returned, or "unknown" when it returned none.

Connection failures, timeouts, HTTP 408, 429 and 5xx are retried three times with
exponential backoff (2, 4, 8 seconds). A 200 reply with a malformed body gets one extra
attempt. A call that still fails is recorded as an unreadable verdict and the run carries
on; pass `on_error="raise"` to `Calibration.run` to stop at the first failure instead.

From Python:

```python
from judgecal import Calibration, ChatCompletionsJudge, Items, ReplayJudge, Rubric
from judgecal.report import to_html

res = Calibration.run(
    Items.load("items.jsonl"),
    [ChatCompletionsJudge("http://localhost:8000/v1", "my-judge", api_key_env=None),
     ReplayJudge("old-verdicts.jsonl", "last-month")],
    repeats=2,
    rubric=Rubric.load("rubric.json"),
)
print(res.band_line())
open("report.html", "w").write(to_html(res))
```

The `Judge` protocol is one method, `judge(item, rubric, repeat) -> Verdict | None`, so any
other kind of judge (a human panel's spreadsheet, a different API) drops in.

## What counts in the denominator

Every agreement figure counts only the items where both sides have a readable verdict.
Unreadable replies (no rubric class in the reply, or a call that failed after its retries)
and missing items (no verdict at all) are dropped from the denominator and counted as
unpaired. The text and HTML reports print "paired n / unpaired m" beside every agreement
figure, and the noise-band line names its paired n. A run with many unpaired items gives a
band computed on fewer items than the item file holds; the unpaired count shows how many.

## Read the noise band

The band is half the share of paired items on which the reference judge changed its class
between two runs. The reference is the first judge with two readable runs unless you name
one with `--reference`. Rounded to two places, it goes on its own line beside a judged
result:

```
completion rate 0.62 [0.53, 0.70] (Wilson, n = 120); judge noise ±0.05
```

(illustrative numbers)

The bracket is sampling error from the number of items. The judge's own wobble goes in the
band, a convention that the study derived from a single pair of runs, so it sits apart on the
line, after the semicolon.

The report also shows:

- **Test-retest** per judge: agreement, Wilson interval, κ and the judge's own band.
- **Between judges**: pairwise agreement and κ on one chosen run (`--pair-repeat`,
  default 1). A cheaper judge whose agreement with the reference sits inside the
  reference's own retest interval is hard to tell apart from it on that rubric.
- **Confidence**: the share of verdicts marked low (the label `low`, or a number under 50
  of 100; decimals between 0 and 1 count as probabilities), unreadable replies, missing
  items and the class mix per run.
- **Confusion tables**, so you can see which classes a judge wavers between.
- **Where the judge wavers**: specific agreement and flips per class, the class that took
  part in the most flips, the commonest move between two classes, and a paragraph written
  from those figures.
- **κ bootstrap intervals** beside every Wilson interval: percentile, 10,000 draws, seed 1
  unless you pass `--seed` or `--boot-draws` (0 turns it off). Resampling items with
  replacement is done as one exact multinomial draw over the confusion cells, so the same seed
  gives the same interval on Python 3.9 and 3.14. Draws where κ does not exist (every
  resampled verdict in one class) are counted and printed.
- **Repeat sensitivity**, with `--repeats 3` or more: every pair of runs gets a row, the band
  pooled over the first m runs is shown for each m, and a note says whether the band, rounded
  to two places, moves as runs are added.
- **Strata**, with `--strata strata.jsonl`: rows of `{"id": ..., "stratum": ...}` that you
  write from your own design, such as a difficulty split fixed before the run. The
  report gives agreement, Wilson and κ within each stratum. judgecal never derives strata
  from verdicts or answers; items the file leaves out sit under "(no stratum)".
- **Cost**: one line per run, from returned token counts and your prices.

The HTML report opens with a one-page summary (band, the reference judge's retest with both
intervals, the closest other judge, the wavers paragraph, cost and the print line) and prints
cleanly: a print stylesheet puts the summary on page one and the detail after it.

## Drift

```
python3 -m judgecal compare verdicts-2026-10-01.jsonl verdicts-2026-10-08.jsonl \
  --rubric rubric.json --html drift.html
```

`compare` reads two files written by `--verdicts-out` on different dates and reports, for
each judge in both, agreement and κ between the sessions. This is test-retest across
sessions, the design a registered judge-reliability study (October 2026) used for drift.
Anything that changed between the two days lands in that figure, sampling noise included.
Where a file holds
two runs from one session, the within-session retest is printed beside it, with a sentence
saying whether the across-session rate sits inside that interval. Dates come from the files
(or `--date-a`, `--date-b`); calendar dates give whole days, so daylight-saving changes and
year boundaries count like any other night, and timestamps with UTC offsets give elapsed
hours. `demo/drift.html` runs it on the synthetic data, one week apart.

## What it does not do

- Two judges can agree on every item and both be wrong; agreement figures cannot show it.
- A flip has many possible causes. Sampling at a non-zero temperature, the order of the
  prompt and a model that changed between runs all land in the same retest number.
- Items are pooled. If they fall into kinds, run each kind as its own item file so the band
  for each kind stays visible.
- The tests make no call outside the machine. The endpoint judge has run against fake
  transports, a patched opener and two loopback servers, never a real model endpoint.
- The endpoint judge refuses redirects. A 3xx is an error, because following it would send
  the key to whatever origin the redirect names; point `--endpoint` at the final address.
- Requests are uncached. A 200-item, two-judge, two-repeat run is 800 requests, four at a
  time per judge by default.
- A thread stuck inside a socket read cannot be killed from Python. The item timeout is
  enforced before each attempt and backoff, and the default transport's socket timeout is cut
  to the time left, so an item overruns its limit by at most one partial read.
- Cost counts the replies the tool read. Tokens spent on a malformed reply that was retried,
  and on calls that failed, are left out, so a run with many retries costs more than the line
  says.
- Strata κ has no bootstrap interval, and the pooled band in the repeat-sensitivity table
  reuses the same items across pairs, so it describes the run rather than estimating anything.
- Judge names cannot hold `=` when used with `--price`, and a `--replay` name that holds
  `=` works only when the path after the last `=` exists.

## Saved verdicts, progress and small samples

`--verdicts-out` appends each verdict and flushes it to disk the moment the call returns,
before any report is written, so a failed `--html` or a Ctrl-C keeps every call already paid
for. Rows land in completion order and replay in any order. Once a run has taken more than
2 s, stderr gets one line each time a judge finishes a repeat. Any agreement figure on fewer
than 20 paired items (retest, pairwise, stratum or drift) prints a warning on stderr, in the
text report and in the HTML summary.

## Item ids

Every id passes through one rule: items, replayed verdicts, strata and saved verdict files
all drop a trailing `.md`, so `a1.md` and `a1` name the same item. Two items that differ
only by that suffix are a duplicate.

## Errors and hostile input

Bad input stops the run with one line on stderr and exit code 2, naming the file and line:
malformed JSON, a line that is not an object, an id that is empty or not a string or number,
a duplicate id (both lines named), a repeat that is not a whole number, an empty items file, a
rubric with fewer than two classes or a broken `answer_format`. Files are read as UTF-8 with
or without a byte-order mark; CRLF, lone CR and raw U+2028 inside strings all read cleanly.
A confidence outside 0 to 100 (negative, NaN, infinite) counts as not low. Every piece of user
text in the HTML is escaped; `tests/smoke.sh` loads a report built from `<script>` and
`<img onerror>` payloads in headless Chrome and checks that nothing runs. In the text report,
control characters are shown as `\x1b` rather than sent to the terminal. `--json` writes
strict JSON (`null` where a figure is undefined). See CHANGELOG.md for what the stress round
found.

## Tests

```
python3 -W error -m unittest discover -s tests -v   # 135 tests, about 8 s, no outside network
bash tests/smoke.sh                                  # 13 headless Chrome checks; exit 1 on failure
```

`tests/test_judgecal.py` holds the original cases, `tests/test_elevate.py` covers the 0.2.0
features, `tests/test_stress.py` the hostile inputs and `tests/test_review2.py` the fixes from
the second review (redirects on two loopback servers, incremental saving, ids, narrow screens,
progress, small samples, shipped surfaces, copy). All pass under `-W error` on Python 3.9.6 and
3.14.5.

## Files

- `judgecal/stats.py` Wilson interval, Cohen's κ, confusion table, noise band, exact
  binomial and multinomial samplers, κ bootstrap, specific agreement
- `judgecal/rubric.py` `Rubric`: classes, instruction, prompt and reply parsing
- `judgecal/items.py` `Items` loader, the shared JSONL reader, `load_strata`
- `judgecal/judges.py` `Judge` protocol, `ChatCompletionsJudge`, `ReplayJudge`
- `judgecal/calibration.py` `Calibration.run` (thread pool, strata, sensitivity, cost) and
  the result objects
- `judgecal/compare.py` `compare`: drift between two verdict files
- `judgecal/report.py` text and HTML report, inline SVG chart, fixture label
- `judgecal/cli.py` `python3 -m judgecal report ...` and `compare ...` (or `judgecal ...`
  after `pip install .`)
- `tests/` 135 unittest cases in four files, `smoke.sh`; fixtures in `tests/fixtures/`
  (`elevate/` is hand-written test data, not a measurement)
- `demo/report.html`, `demo/drift.html`, `demo/build.sh`, and `demo/data/` (synthetic)
- `CHANGELOG.md` what the stress and review round found and changed

## Sources

- The registered judge-reliability study, October 2026 (not yet public): its design, its
  measures and its analysis script, whose `wilson` and `kappa` functions
  `judgecal/stats.py` reuses unchanged. The study will be linked here once it publishes.
