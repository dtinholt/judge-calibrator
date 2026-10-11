# Changelog

## 0.2.1, 10 October 2026

Fixes from the second independent check.

- **Redirects refused.** The endpoint judge used urllib's default opener, which follows a
  301, 302 or 303 by re-sending the request as a GET, Bearer header included, to whatever
  origin the Location names. The transport now uses an opener with no redirect handling, and
  any 3xx is an error that is never retried. A test on two loopback servers shows the second
  origin receives nothing; a control test shows plain `urlopen` leaking the key to it.
- **Verdicts saved as they land.** `--verdicts-out` used to be written after the HTML and JSON,
  so a failed `--html` or a Ctrl-C lost every paid call. Each verdict is now appended and
  flushed the moment it returns, from a callback that never runs two at once, before any
  report is written. Tests cover a failed `--html` and a Ctrl-C mid-run.
- **`OpenAICompatJudge` removed**, with its export, README line and test.
- **Study figures out of shipped surfaces.** The demos are built from synthetic data in
  `demo/data/` (`demo/build.sh`, `demo/data/make_synthetic.py`) and say so in a stamp. The
  README and the `noise_band` docstring no longer quote the study's results. The study fixture
  stays for regression tests, `MANIFEST.in` prunes it from the sdist, `pyproject.toml` ships
  only `judgecal/` with package data off, and the publish-day TODO says so. Not verified with a
  real build: neither interpreter here has setuptools 61 or later, and the build ran offline.
- **One id rule.** Replay keys dropped a trailing `.md`; item ids, strata and saved verdict
  files did not, so `a1.md` in the items never met `a1` in a replay. Every id now goes through
  the same function. The saved-verdict reader also learned the `packet` key.
- **Narrow screens.** Table wrappers are focusable named regions (`tabindex="0"`,
  `role="region"`, `aria-label` from the caption) with a "scrolls sideways" cue under 40 rem.
  The chart keeps a 560 px minimum width and scrolls, so no chart text drops below 11 px.
  Gridlines and the reference line run only through each row's bar band, clear of the labels.
- **Progress.** Once a run passes 2 s, stderr gets one line each time a judge finishes a repeat.
- **Small samples.** Any retest, pairwise, stratum or drift figure on fewer than 20 paired
  items prints a warning on stderr, in the text report and in the HTML summary.
- **Copy.** Rewrote a parallel pair and two three-item lists in the README, the legend, the
  drift paragraph and the limits list. The accuracy caveat now appears once, in the
  README.

## 0.2.0, 10 October 2026

Stress, review and elevate round. Tests went from 44 to 113 (`python3 -W error -m unittest
discover -s tests`, green on Python 3.9.6 and 3.14.5), plus `tests/smoke.sh` with 10 headless
checks.

### What broke under stress, and the fix

| Input | Before | Now |
|---|---|---|
| A malformed JSON line | raw `JSONDecodeError` traceback, no file or line | `judgecal: error: items.jsonl:2: not valid JSON (...)`, exit 2 |
| A line that is a number or a list | `TypeError` traceback | named as "each line must be a JSON object", exit 2 |
| A UTF-8 byte-order mark | `JSONDecodeError` on line 1 | read as UTF-8 with or without the mark |
| A raw U+2028 or U+0085 inside a JSON string | the line was cut in two (`str.splitlines`) and failed | lines end at `\n` only; the text keeps its separator |
| 100,000 levels of nesting | `RecursionError` traceback on 3.9.6 | "JSON nested too deeply to read", exit 2 (3.14 decodes it) |
| An empty items file | exit 0 and a report on 0 items | "no items", exit 2 |
| `"repeat": "x"`, `1.5`, `-1`, `NaN` | traceback, or 1.5 silently read as 1 | rejected with file and line |
| `--repeats 0`, `--pair-repeat 0`, a one-class rubric, a missing file | tracebacks | one-line error, exit 2 |
| `--temperature nan` | accepted; the request body would carry `NaN`, which is not JSON | rejected; bodies are serialised with `allow_nan=False` |
| Confidence `-5`, `150`, `NaN`, `inf` | `-5` counted as low confidence | a number outside 0 to 100 counts as not low |
| `--json` on a run with an undefined κ | wrote `NaN`, which strict JSON readers refuse | writes `null` |
| A judge name containing `=` | `--replay` split the name and failed to find the file | falls back to the last `=` when that names real files |
| ANSI escapes in a judge or class name | sent to the terminal as-is | shown as `\x1b` in the text report |
| `<script>` and `<img onerror>` in every text field | already escaped | confirmed in headless Chrome: title unchanged, no script or img element in the DOM |
| Duplicate ids | error named the id only | names both line numbers |
| A 5 MB item | loaded | still loads; covered by a test |
| 100x the sample (19,700 items, three judges) | 0.36 s without the bootstrap | 0.71 s on 3.14.5 and 1.11 s on 3.9.6 with 10,000-draw bootstraps on four tables |

Render time for the 100x report in headless Chrome, `performance.now()` around the body,
three loads: 265, 260 and 187 ms to the end of the body, 442, 422 and 428 ms to the load
event. The demo report: 137, 120 and 147 ms; 300, 397 and 375 ms. The report holds aggregates
only, so its size does not grow with the item count (24 KB at 100x). The page itself never
needed chunking. The run before it can take minutes against a live endpoint, which 0.2.1
answers with a progress line per repeat once a run passes 2 s.

### What the hostile review found

- Zero cells in the confusion tables used the rule colour, which sits too close to the paper
  to read. They use graphite now.
- Meaning carried by colour alone: the confusion diagonal (now also underlined) and the two
  kinds of chart row (judge against itself is a round marker, judge against judge a square).
- The confusion tables had an empty `<th>` in the corner and column headers without `scope`.
  The SVG chart had a title and nothing else; it now carries a `<desc>` with every value.
  The page gained a skip link.
- `urllib` HTTP error responses were read but never closed.
- `_OpenRubric` grew its class list without a lock, which matters now that calls can run
  concurrently.
- No per-item time limit: four attempts at a 120 s socket timeout plus backoff could hold one
  item for over eight minutes.
- A rubric `answer_format` with stray braces raised `KeyError` at the first call, mid-run.
  It is checked when the file loads. `descriptions` that is not an object, and `classes`
  given as a string, are rejected the same way.
- Files were read in the locale's encoding. They are read as UTF-8.
- Ctrl-C printed a traceback. It exits 130 with one line.
- One silent catch stays on purpose: `Rubric.parse` tries JSON first and falls back to reading
  `CLASS:` lines when the reply is not JSON. Two tests cover the fallback.
- "1 judges" in the lede.
- Print: the chart's end tick labels were clipped and a judge heading was left alone at the
  foot of a page. Phone: row headers broke mid-word. A κ cell with undefined bootstrap draws
  ran off the table.
- Not applicable to this build: localStorage, keyboard traps and async races in page code.
  The report has no script at all, and a test asserts it.

### Added

- Percentile bootstrap intervals for κ, 10,000 draws by default, seeded (`--seed`,
  `--boot-draws`), printed beside the Wilson interval on agreement. The resampling is done
  as one multinomial draw over the confusion cells, which is exact, so a seeded interval is
  identical on 3.9 and 3.14. Draws where κ does not exist are counted and reported.
- Per-class agreement (specific agreement and flips per class), the class most often
  flipped, the commonest move between two classes, and a "where the judge wavers"
  paragraph written from those figures.
- `--strata`: a JSONL of item id and stratum that you supply; agreement, Wilson and κ per
  stratum for every retest and every pair of judges.
- Repeat sensitivity: with three or more runs, every pair of runs gets its own row, and the
  band pooled over the first m runs is shown for each m, with a note on whether it changes.
- Concurrency for endpoint judges: a bounded thread pool (`--workers`, default 4), results
  put back in item order, and a per-item timeout (`--item-timeout`, default 300 s) that covers
  retries and backoff and cuts each attempt's socket timeout to the time left.
- A cost line from the token counts the endpoint returns, at prices you give per judge
  (`--price NAME=IN,OUT`, US dollars per million tokens), or "unknown" when the endpoint
  returns none. `--verdicts-out` now saves token counts and the run date.
- `judgecal compare A B`: drift between two saved verdict files from different dates,
  labelled as test-retest across sessions, with the within-session retest beside it where
  a file holds two runs, and the gap between the dates.
- HTML report: a one-page summary at the top and a print stylesheet that breaks after it.
