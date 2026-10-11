"""Command line: python3 -m judgecal report ... | python3 -m judgecal compare ..."""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
import time
from pathlib import Path

from .calibration import Calibration
from .compare import compare
from .items import Items, load_strata
from .judges import ChatCompletionsJudge, JudgeError, ReplayJudge
from .report import drift_html, drift_text, to_html, to_text
from .rubric import Rubric


def _positive_int(s: str) -> int:
    try:
        v = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {s!r}") from None
    if v < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more, got {v}")
    return v


def _nonneg_int(s: str) -> int:
    try:
        v = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {s!r}") from None
    if v < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or more, got {v}")
    return v


def _seconds(s: str) -> float:
    try:
        v = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected seconds, got {s!r}") from None
    if not (math.isfinite(v) and v > 0):
        raise argparse.ArgumentTypeError(f"must be a positive number of seconds, got {s!r}")
    return v


def _temperature(s: str) -> float:
    try:
        v = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {s!r}") from None
    if not (math.isfinite(v) and v >= 0):
        raise argparse.ArgumentTypeError(f"must be a finite number of 0 or more, got {s!r}")
    return v


def parse_price(spec: str) -> tuple[str, tuple[float, float]]:
    """NAME=IN,OUT in US dollars per million input and output tokens."""
    name, sep, rest = spec.partition("=")
    parts = rest.split(",")
    if not sep or not name or len(parts) != 2:
        raise ValueError(f"--price needs NAME=IN,OUT (USD per million tokens), got {spec!r}")
    try:
        pi, po = float(parts[0]), float(parts[1])
    except ValueError:
        raise ValueError(f"--price {spec!r}: prices must be numbers") from None
    if not all(math.isfinite(x) and x >= 0 for x in (pi, po)):
        raise ValueError(f"--price {spec!r}: prices must be finite and 0 or more")
    return name, (pi, po)


def _common_out(r: argparse.ArgumentParser) -> None:
    r.add_argument("--html", metavar="PATH", help="write the HTML report here")
    r.add_argument("--json", metavar="PATH", help="write the result as JSON here")
    r.add_argument("--title", help="report title")
    r.add_argument("--source", help="source line printed in small mono under the headline")
    r.add_argument("--date", help="date stamped on the report and on saved verdicts (default today)")
    r.add_argument("--seed", default="1", help="bootstrap seed (default 1)")
    r.add_argument("--boot-draws", type=_nonneg_int, default=10_000,
                   help="bootstrap draws for the kappa interval; 0 turns it off (default 10000)")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="judgecal", description="Measure judge consistency.")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="run judges over items and report consistency")
    r.add_argument("--items", required=True, help="items JSONL (one object per line with an id)")
    r.add_argument("--rubric", help="rubric JSON: classes, instruction, descriptions")
    r.add_argument("--classes", help="comma-separated classes when no rubric file is given")
    r.add_argument("--replay", action="append", default=[], metavar="NAME=PATH[,PATH...]",
                   help="a recorded judge; several paths mean one file per repeat")
    r.add_argument("--fixture", action="store_true",
                   help="the --replay files are recorded test fixtures; the report carries the fixture label")
    r.add_argument("--fixture-note", help="text of the stamped label at the top of the report")
    r.add_argument("--endpoint", action="append", nargs=4, default=[],
                   metavar=("NAME", "BASE_URL", "MODEL", "API_KEY_ENV"),
                   help="a chat-completions judge; API_KEY_ENV '-' for none")
    r.add_argument("--temperature", type=_temperature, help="temperature for endpoint judges")
    r.add_argument("--workers", type=_positive_int, default=4,
                   help="concurrent requests per endpoint judge (default 4)")
    r.add_argument("--item-timeout", type=_seconds, default=300.0,
                   help="seconds one item may take on an endpoint, retries included (default 300)")
    r.add_argument("--price", action="append", default=[], metavar="NAME=IN,OUT",
                   help="USD per million input and output tokens for one judge")
    r.add_argument("--strata", metavar="PATH",
                   help="JSONL of {id, stratum} that you supply; agreement is reported per stratum")
    r.add_argument("--repeats", type=_positive_int, default=2)
    r.add_argument("--reference", help="judge whose test-retest sets the band (default: first)")
    r.add_argument("--pair-repeat", type=_positive_int, default=1,
                   help="which run (1-based) to compare across judges")
    r.add_argument("--verdicts-out", metavar="PATH", help="save every verdict as JSONL (replayable)")
    r.add_argument("--notes", help="one paragraph printed under the lede")
    _common_out(r)

    c = sub.add_parser("compare", help="drift: compare two saved verdict files from different dates")
    c.add_argument("a", help="earlier verdict file (from report --verdicts-out)")
    c.add_argument("b", help="later verdict file")
    c.add_argument("--judge", action="append", default=[], help="judge to compare (default: all in both)")
    c.add_argument("--repeat", type=_positive_int, default=1, help="which run (1-based) of each file")
    c.add_argument("--rubric", help="rubric JSON, to read classes in rubric order")
    c.add_argument("--classes", help="comma-separated classes when no rubric file is given")
    c.add_argument("--date-a", help="date of session A, when the file carries none")
    c.add_argument("--date-b", help="date of session B, when the file carries none")
    _common_out(c)
    return p


def _rubric(a) -> Rubric | None:
    if a.rubric:
        return Rubric.load(a.rubric)
    if a.classes:
        return Rubric([c for c in a.classes.split(",") if c.strip()])
    return None


def _report(a) -> int:
    items = Items.load(a.items)
    rubric = _rubric(a)
    judges = []
    for spec in a.replay:
        name, _, paths = spec.partition("=")
        if paths and not all(Path(x).exists() for x in paths.split(",") if x):
            # a judge name may itself hold "=": fall back to the last "=" when that names real files
            n2, _, p2 = spec.rpartition("=")
            if n2 and p2 and all(Path(x).exists() for x in p2.split(",") if x):
                name, paths = n2, p2
        if not name or not paths:
            raise ValueError(f"--replay needs NAME=PATH, got {spec!r}")
        ps = [x for x in paths.split(",") if x]
        judges.append(ReplayJudge(ps, name, repeat_per_file=len(ps) > 1, fixture=a.fixture))
    for name, url, model, env in a.endpoint:
        if rubric is None:
            raise ValueError("endpoint judges need --rubric or --classes")
        judges.append(ChatCompletionsJudge(url, model, None if env == "-" else env, name=name,
                                           temperature=a.temperature, item_timeout=a.item_timeout))
    if not judges:
        raise ValueError("give at least one --replay or --endpoint judge")
    prices = dict(parse_price(s) for s in a.price)
    strata = load_strata(a.strata) if a.strata else None
    date = a.date or _dt.date.today().isoformat()
    vout = open(a.verdicts_out, "w", encoding="utf-8") if a.verdicts_out else None
    try:
        # Each verdict is appended and flushed the moment it arrives, before any report is
        # written, so a failed --html or a Ctrl-C mid-run keeps every call already paid for.
        res = Calibration.run(items, judges, repeats=a.repeats, rubric=rubric, reference=a.reference,
                              pair_repeat=a.pair_repeat - 1, workers=a.workers, strata=strata,
                              prices=prices, boot_draws=a.boot_draws, seed=a.seed,
                              on_verdict=(lambda v: write_verdict(vout, v, date)) if vout else None,
                              progress=make_progress(sys.stderr))
    finally:
        if vout is not None:
            vout.close()
    for w in res.small_n_warnings():
        print(f"judgecal: warning: {w}", file=sys.stderr)
    print(to_text(res, fixture_note=a.fixture_note))
    if a.html:
        Path(a.html).write_text(to_html(res, title=a.title or "Judge consistency report", source=a.source,
                                        date=date, notes=a.notes, fixture_note=a.fixture_note),
                                encoding="utf-8")
        print(f"\nwrote {a.html}", file=sys.stderr)
    if a.json:
        Path(a.json).write_text(json.dumps(res.to_dict(), indent=1, default=str, allow_nan=False),
                                encoding="utf-8")
    return 0


def write_verdict(f, v, date: str) -> None:
    """Append one verdict as a replayable JSONL row and flush it to disk."""
    row = {"item": v.item, "judge": v.judge, "repeat": v.repeat, "class": v.cls,
           "confidence": v.confidence, "date": date}
    if v.input_tokens is not None:
        row["input_tokens"] = v.input_tokens
    if v.output_tokens is not None:
        row["output_tokens"] = v.output_tokens
    if v.failed:
        row["failed"] = True
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    f.flush()


def make_progress(stream, clock=time.monotonic, after: float = 2.0):
    """A progress callback that stays quiet for short runs. Once the run has taken longer
    than `after` seconds, it prints one line each time a judge finishes a repeat."""
    t0 = clock()

    def note(judge: str, repeat: int, repeats: int, items: int) -> None:
        el = clock() - t0
        if el > after:
            print(f"judgecal: {judge} run {repeat + 1} of {repeats} done "
                  f"({items} items, {el:.1f} s elapsed)", file=stream, flush=True)
    return note


def _compare(a) -> int:
    d = compare(a.a, a.b, judges=a.judge or None, repeat=a.repeat - 1, rubric=_rubric(a),
                date_a=a.date_a, date_b=a.date_b, boot_draws=a.boot_draws, seed=a.seed)
    for w in d.small_n_warnings():
        print(f"judgecal: warning: {w}", file=sys.stderr)
    print(drift_text(d))
    if a.html:
        Path(a.html).write_text(drift_html(d, title=a.title or "Judge drift report", date=a.date,
                                           source=a.source), encoding="utf-8")
        print(f"\nwrote {a.html}", file=sys.stderr)
    if a.json:
        Path(a.json).write_text(json.dumps(d.to_dict(), indent=1, default=str, allow_nan=False),
                                encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    a = _parser().parse_args(argv)
    try:
        return _report(a) if a.cmd == "report" else _compare(a)
    except (ValueError, OSError, JudgeError) as e:
        # Bad input, a missing file or a judge that failed under on_error="raise": one line, exit 2.
        print(f"judgecal: error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("judgecal: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
