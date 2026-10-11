"""Drift: the same judge on the same items, read from two saved verdict files from different dates.

This is test-retest across sessions, the design a registered judge-reliability study
(October 2026) used to look for drift: the second session is a fresh run on another day,
so anything that changed in between, sampling noise included, lands in the across-session
figure. Where a file holds two runs of the same
judge from one session, the within-session retest is shown beside it for scale.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .calibration import SMALL_N, Agreement, _agreement, _clean, small_n_list
from .items import _id, read_jsonl
from .judges import _repeat
from .rubric import Rubric


@dataclass
class Session:
    path: str
    date: Optional[str]  # the single date the file's rows carry, or None
    dates: list[str]  # every distinct date seen in the file
    # verdicts[(judge, repeat)] = {item id: class or None when unreadable}
    verdicts: dict[tuple[str, int], dict[str, Optional[str]]]
    order: list[str]  # item ids in first-seen order

    def judges(self) -> list[str]:
        seen: list[str] = []
        for j, _ in self.verdicts:
            if j not in seen:
                seen.append(j)
        return seen


def load_session(path: str | Path, rubric: Optional[Rubric] = None) -> Session:
    """Read a file written by `judgecal report --verdicts-out` (rows: item, judge, repeat, class,
    optional confidence, optional date). Rows with no `judge` are filed under "judge"."""
    verdicts: dict[tuple[str, int], dict[str, Optional[str]]] = {}
    order: list[str] = []
    seen_ids: set[str] = set()
    dates: list[str] = []
    for n, d in read_jsonl(path):
        key = next((d[k] for k in ("item", "packet", "id") if k in d), None)
        iid = _id(key, f"{path}:{n}")
        judge = str(d.get("judge", "judge"))
        rep = _repeat(d.get("repeat", 0), f"{path}:{n}")
        raw = d.get("class")
        if rubric is not None:
            cls_ = rubric.normalise(raw)
        else:
            s = str(raw or "").strip().upper()
            cls_ = s or None
        slot = verdicts.setdefault((judge, rep), {})
        if iid in slot:
            raise ValueError(f"{path}:{n}: duplicate verdict for {iid!r}, judge {judge!r}, repeat {rep}")
        slot[iid] = cls_
        if iid not in seen_ids:
            seen_ids.add(iid)
            order.append(iid)
        if d.get("date") is not None:
            ds = str(d["date"]).strip()
            if ds and ds not in dates:
                dates.append(ds)
    if not verdicts:
        raise ValueError(f"{path}: no verdict rows")
    return Session(str(path), dates[0] if len(dates) == 1 else None, dates, verdicts, order)


def _parse_when(s: str):
    t = s.strip()
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    if len(t) == 10:
        return _dt.date.fromisoformat(t)
    return _dt.datetime.fromisoformat(t)


def gap_between(a: Optional[str], b: Optional[str]) -> tuple[Optional[str], list[str]]:
    """Elapsed time from date a to date b as text, and any notes.

    Calendar dates give whole days, so daylight-saving changes cannot shift the count and a
    year boundary counts like any other night. Timestamps with UTC offsets give elapsed time
    in absolute terms. A naive timestamp beside an offset one has no defined gap.
    """
    notes: list[str] = []
    if not a or not b:
        return None, notes
    try:
        x, y = _parse_when(a), _parse_when(b)
    except ValueError:
        notes.append("the session dates are not ISO 8601, so no gap is computed")
        return None, notes
    if isinstance(x, _dt.datetime) != isinstance(y, _dt.datetime):
        x = x.date() if isinstance(x, _dt.datetime) else x
        y = y.date() if isinstance(y, _dt.datetime) else y
    if isinstance(x, _dt.datetime):
        if (x.tzinfo is None) != (y.tzinfo is None):
            notes.append("one timestamp has a UTC offset and the other does not, so no gap is computed")
            return None, notes
        secs = (y - x).total_seconds()
        hours = abs(secs) / 3600
        text = f"{hours / 24:.1f} days" if hours >= 48 else f"{hours:.1f} hours"
    else:
        days = (y - x).days
        secs = days * 86400.0
        text = f"{abs(days)} day" + ("" if abs(days) == 1 else "s")
    if secs == 0:
        notes.append("both sessions carry the same date, so this reads as a within-session retest, not drift")
    elif secs < 0:
        notes.append("session B is dated before session A")
        text += " (B before A)"
    return text, notes


@dataclass
class DriftJudge:
    judge: str
    across: Agreement
    within_a: Optional[Agreement] = None
    within_b: Optional[Agreement] = None

    @property
    def band(self) -> Optional[float]:
        return (1 - self.across.rate) / 2 if self.across.n else None


@dataclass
class DriftResult:
    label_a: str
    label_b: str
    date_a: Optional[str]
    date_b: Optional[str]
    gap: Optional[str]
    repeat: int  # 0-based run compared from each file
    classes: list[str]
    judges: list[DriftJudge]
    notes: list[str] = field(default_factory=list)
    boot_draws: int = 0

    def small_n_warnings(self, threshold: int = SMALL_N) -> list[str]:
        ags = []
        for j in self.judges:
            ags += [j.across] + [w for w in (j.within_a, j.within_b) if w is not None]
        return small_n_list(ags, threshold)

    def to_dict(self) -> dict:
        from dataclasses import asdict
        d = asdict(self)
        for j, dj in zip(d["judges"], self.judges):
            j["band"] = dj.band
        return _clean(d)


def compare(
    path_a: str | Path,
    path_b: str | Path,
    judges: Optional[list[str]] = None,
    repeat: int = 0,
    rubric: Optional[Rubric] = None,
    date_a: Optional[str] = None,
    date_b: Optional[str] = None,
    boot_draws: int = 10_000,
    seed: object = 1,
) -> DriftResult:
    """Compare run `repeat` of each judge in file A with the same run in file B."""
    if repeat < 0:
        raise ValueError("repeat must be 0 or more")
    a, b = load_session(path_a, rubric), load_session(path_b, rubric)
    notes: list[str] = []
    for s, lab in ((a, "A"), (b, "B")):
        if len(s.dates) > 1:
            notes.append(f"file {lab} holds verdicts from {len(s.dates)} dates ({', '.join(s.dates)})")
    da = date_a or a.date
    db = date_b or b.date
    if not da or not db:
        notes.append("a session date is missing; pass --date-a and --date-b to label the sessions")
    gap, gnotes = gap_between(da, db)
    notes += gnotes
    common = [j for j in a.judges() if j in b.judges()]
    if judges:
        missing = [j for j in judges if j not in common]
        if missing:
            raise ValueError(f"judge(s) not in both files: {', '.join(missing)}")
        common = list(judges)
    if not common:
        raise ValueError("the two files share no judge name")
    if rubric is not None:
        classes = list(rubric.classes)
    else:
        found: list[str] = []
        for s in (a, b):
            for slot in s.verdicts.values():
                for c in slot.values():
                    if c is not None and c not in found:
                        found.append(c)
        classes = found
    out = []
    for j in common:
        va, vb = a.verdicts.get((j, repeat)), b.verdicts.get((j, repeat))
        if va is None or vb is None:
            notes.append(f"{j}: run {repeat + 1} is missing from one file, skipped")
            continue
        ids = list(dict.fromkeys([i for i in a.order if i in va] + [i for i in b.order if i in vb]))
        ra = {k: v for k, v in va.items() if v is not None}
        rb = {k: v for k, v in vb.items() if v is not None}
        across = _agreement(f"{j} A", f"{j} B", ra, rb, ids, classes, boot_draws, seed)
        within = []
        for s, tag in ((a, "A"), (b, "B")):
            other = 1 if repeat == 0 else 0
            x, y = s.verdicts.get((j, repeat)), s.verdicts.get((j, other))
            if x is None or y is None:
                within.append(None)
                continue
            r1, r2 = sorted((repeat, other))
            x, y = s.verdicts[(j, r1)], s.verdicts[(j, r2)]
            wid = list(dict.fromkeys(list(x) + list(y)))
            within.append(_agreement(f"{j} {tag}#{r1 + 1}", f"{j} {tag}#{r2 + 1}",
                                     {k: v for k, v in x.items() if v is not None},
                                     {k: v for k, v in y.items() if v is not None},
                                     wid, classes, boot_draws, seed, (r1 + 1, r2 + 1)))
        out.append(DriftJudge(j, across, within[0], within[1]))
    if not out:
        raise ValueError(f"no judge has run {repeat + 1} in both files")
    return DriftResult(Path(path_a).name, Path(path_b).name, da, db, gap, repeat, classes, out, notes, boot_draws)


def drift_sentence(dj: DriftJudge) -> str:
    """One plain sentence placing the across-session figure against the within-session one."""
    x = dj.across
    if not x.n:
        return f"{dj.judge} has no item with a readable verdict in both sessions."
    s = (f"Across sessions {dj.judge} gave the same class to {x.k} of {x.n} paired items "
         f"({x.rate:.3f}, Wilson 95% [{x.lo:.3f}, {x.hi:.3f}]).")
    w = dj.within_a or dj.within_b
    if w is not None and w.n:
        tag = "A" if dj.within_a is not None else "B"
        where = ("inside" if w.lo <= x.rate <= w.hi else ("below" if x.rate < w.lo else "above"))
        s += (f" Its two runs within session {tag} agreed on {w.k} of {w.n} ({w.rate:.3f}, "
              f"[{w.lo:.3f}, {w.hi:.3f}]); the across-session rate sits {where} that interval.")
    return s
