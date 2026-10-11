"""Run judges over items and measure how consistent they are."""
from __future__ import annotations

import math
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Callable, Optional, Sequence

from .items import Items
from .judges import Judge, JudgeError, Verdict
from .rubric import Rubric
from .stats import bootstrap_kappa, confusion, kappa, noise_band, specific_agreement, wilson

NO_STRATUM = "(no stratum)"


class _OpenRubric(Rubric):
    """Used when no rubric is given: accepts any label and learns the classes as it goes."""

    def __init__(self) -> None:
        self.classes = []
        self.instruction = ""
        self.descriptions = {}
        self.name = "inferred"
        self.answer_format = ""
        self._lock = threading.Lock()

    def normalise(self, label: object) -> Optional[str]:
        s = str(label or "").strip().upper()
        if not s:
            return None
        with self._lock:
            if s not in self.classes:
                self.classes.append(s)
        return s


def is_low(confidence: Optional[str], threshold: float = 50.0) -> bool:
    """True for the label "low", or for a numeric confidence under `threshold` (0-100 scale).

    A decimal between 0 and 1 is read as a probability and scaled to 0-100. Labels such as
    "medium-high" count as not low, as in the judge-reliability study. A number outside
    0-100 (negative, over 100, NaN, infinite) is not a confidence and counts as not low.
    """
    if confidence is None:
        return False
    c = confidence.strip().lower()
    if c == "low":
        return True
    try:
        x = float(c.rstrip("%"))
    except ValueError:
        return False
    if not (0.0 <= x <= 100.0):
        return False
    if x <= 1.0 and "." in c:
        x *= 100.0
    return x < threshold


SMALL_N = 20


def small_n_list(ags, threshold: int = SMALL_N) -> list[str]:
    """One warning per agreement figure with fewer than `threshold` paired items (n > 0)."""
    return [f"{a.a} vs {a.b} rests on {a.n} paired items (under {threshold}); its interval is wide "
            f"and its κ unstable" for a in ags if 0 < a.n < threshold]


@dataclass
class ClassAgreement:
    """How one class fared between two sides of a comparison."""
    cls: str
    named_a: int  # verdicts on side A naming the class (paired items only)
    named_b: int
    both: int  # items both sides put in the class
    specific: Optional[float]  # 2 * both / (named_a + named_b); None if neither side named it
    flips: int  # paired items where exactly one side named the class


@dataclass
class Agreement:
    a: str
    b: str
    k: int
    n: int
    rate: float
    lo: float
    hi: float
    kappa: float
    confusion: list[list[int]]
    unpaired: int  # items where either side had no readable verdict
    classes: list[str] = field(default_factory=list)
    kappa_lo: float = float("nan")  # percentile bootstrap interval for kappa
    kappa_hi: float = float("nan")
    kappa_undefined: int = 0  # bootstrap draws where kappa did not exist
    boot_draws: int = 0
    run_a: Optional[int] = None  # 1-based run numbers, for test-retest rows
    run_b: Optional[int] = None
    per_class: list[ClassAgreement] = field(default_factory=list)

    @property
    def disagreements(self) -> int:
        return self.n - self.k

    def most_flipped(self) -> tuple[list[str], int]:
        """The class or classes that took part in the most flips, and that count."""
        top = max((c.flips for c in self.per_class), default=0)
        if top == 0:
            return [], 0
        return [c.cls for c in self.per_class if c.flips == top], top

    def top_flip_pair(self) -> Optional[tuple[str, str, int, int]]:
        """(class x, class y, x->y count, y->x count) for the unordered pair moved between most often.

        Ties go to the pair that comes first in rubric order.
        """
        best = None
        m, cl = self.confusion, self.classes
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                t = m[i][j] + m[j][i]
                if t and (best is None or t > best[2] + best[3]):
                    best = (cl[i], cl[j], m[i][j], m[j][i])
        return best


@dataclass
class Share:
    k: int
    n: int

    @property
    def rate(self) -> float:
        return self.k / self.n if self.n else float("nan")


@dataclass
class RunCountBand:
    """The band pooled over every pair of runs among the first `runs` runs."""
    runs: int
    pairs: int
    disagreements: int
    paired: int

    @property
    def band(self) -> float:
        return self.disagreements / self.paired / 2.0 if self.paired else float("nan")


@dataclass
class JudgeSummary:
    name: str
    verdicts: list[int]  # readable verdicts per repeat
    unreadable: list[int]  # verdicts that came back without a rubric class, per repeat
    missing: list[int]  # items with no verdict at all, per repeat
    low_confidence: list[Share]  # per repeat
    class_mix: list[dict[str, int]]  # per repeat
    retest: list[Agreement] = field(default_factory=list)
    calls: int = 0  # verdicts returned (failed calls excluded)
    failed: int = 0  # calls that failed after their retries
    calls_with_usage: int = 0  # calls that reported both token counts
    input_tokens: int = 0
    output_tokens: int = 0
    replayed: bool = False
    sensitivity: list[RunCountBand] = field(default_factory=list)

    @property
    def headline_retest(self) -> Optional[Agreement]:
        return self.retest[0] if self.retest else None

    @property
    def band(self) -> Optional[float]:
        r = self.headline_retest
        return noise_band(r.rate) if r and r.n else None

    @property
    def low_confidence_pooled(self) -> Share:
        return Share(sum(s.k for s in self.low_confidence), sum(s.n for s in self.low_confidence))


@dataclass
class StratumResult:
    label: str
    n_items: int
    agreements: list[Agreement]  # each judge's run 1 vs run 2, then every pair of judges


@dataclass
class JudgeCost:
    name: str
    price_in: Optional[float]  # USD per million input tokens
    price_out: Optional[float]
    calls: int
    calls_with_usage: int
    input_tokens: int
    output_tokens: int
    failed: int
    replayed: bool

    @property
    def cost(self) -> Optional[float]:
        """USD over the calls that reported token counts; None when unpriced or no counts."""
        if self.price_in is None or self.calls_with_usage == 0:
            return None
        return (self.input_tokens * self.price_in + self.output_tokens * self.price_out) / 1e6

    def describe(self) -> str:
        if self.price_in is None:
            return f"{self.name} price not given"
        if self.calls == 0:
            return f"{self.name} $0 (no calls returned)"
        if self.calls_with_usage == 0:
            where = "in the replay file" if self.replayed else "returned"
            return f"{self.name} unknown (no token counts {where})"
        c = _usd(self.cost)
        parts = [f"{self.input_tokens:,} in / {self.output_tokens:,} out tokens at "
                 f"${self.price_in:g} / ${self.price_out:g} per million"]
        if self.calls_with_usage < self.calls:
            parts.append(f"token counts missing on {self.calls - self.calls_with_usage} of {self.calls} calls")
            c = f"at least {c}"
        if self.failed:
            parts.append(f"{self.failed} failed calls not counted")
        if self.replayed:
            parts.append("recorded counts from the original run, replayed here at no cost")
        return f"{self.name} {c} ({'; '.join(parts)})"


def _usd(x: float) -> str:
    return f"${x:,.4f}" if x < 1 else f"${x:,.2f}"


@dataclass
class CalibrationResult:
    rubric_name: str
    classes: list[str]
    n_items: int
    repeats: int
    pair_repeat: int
    judges: list[JudgeSummary]
    pairwise: list[Agreement]
    reference: Optional[str]
    verdicts: list[Verdict] = field(repr=False, default_factory=list)
    fixture_judges: list[str] = field(default_factory=list)  # judges replayed from flagged fixtures
    strata: list[StratumResult] = field(default_factory=list)
    strata_ignored: int = 0  # strata rows whose id is not among the items
    prices: dict[str, tuple[float, float]] = field(default_factory=dict)
    boot_draws: int = 0
    seed: object = None

    def judge(self, name: str) -> JudgeSummary:
        for j in self.judges:
            if j.name == name:
                return j
        raise KeyError(name)

    def pair(self, a: str, b: str) -> Agreement:
        for p in self.pairwise:
            if {p.a, p.b} == {a, b}:
                return p
        raise KeyError((a, b))

    @property
    def band(self) -> Optional[float]:
        """The noise band from the reference judge's test-retest, or None if not measured."""
        if self.reference is None:
            return None
        return self.judge(self.reference).band

    def band_line(self) -> str:
        """The line to print beside a judged benchmark number."""
        if self.reference is None or self.band is None:
            return "judge noise not measured (no judge has two readable repeats)"
        r = self.judge(self.reference).headline_retest
        return (
            f"judge noise ±{self.band:.2f} (half the {self.reference} test-retest "
            f"disagreement, {r.disagreements}/{r.n} = {r.disagreements / r.n:.3f}; "
            f"paired n = {r.n}, unpaired {r.unpaired})"
        )

    def small_n_warnings(self, threshold: int = SMALL_N) -> list[str]:
        """Warnings for every retest, pairwise and stratum figure under `threshold` paired items."""
        ags = [a for j in self.judges for a in j.retest] + list(self.pairwise)
        out = small_n_list(ags, threshold)
        for s in self.strata:
            out += [f"stratum {s.label}: {w}" for w in small_n_list(s.agreements, threshold)]
        return out

    def costs(self) -> list[JudgeCost]:
        out = []
        for j in self.judges:
            p = self.prices.get(j.name)
            out.append(JudgeCost(j.name, p[0] if p else None, p[1] if p else None, j.calls,
                                 j.calls_with_usage, j.input_tokens, j.output_tokens, j.failed, j.replayed))
        return out

    def cost_line(self) -> str:
        """One line: the run's cost per judge and in total, from returned token counts."""
        if not self.prices:
            return "cost: no prices given (pass --price NAME=IN,OUT in USD per million tokens)"
        cs = self.costs()
        known = [c.cost for c in cs if c.cost is not None and not c.replayed]
        open_ = [c.name for c in cs if c.price_in is not None and c.cost is None and c.calls]
        partial = any(c.calls_with_usage < c.calls for c in cs if c.cost is not None and not c.replayed)
        total = _usd(sum(known)) if known else ("unknown" if open_ else "$0")
        if known and (open_ or partial):
            total = f"at least {total}"
        tail = f"; unknown for {', '.join(open_)}" if open_ and known else ""
        return f"cost: {'; '.join(c.describe() for c in cs)}. Total {total}{tail}."

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("verdicts")
        d["band"] = self.band
        d["band_line"] = self.band_line()
        d["cost_line"] = self.cost_line()
        d["costs"] = [dict(asdict(c), cost=c.cost) for c in self.costs()]
        d["seed"] = None if self.seed is None else str(self.seed)
        for j, js in zip(d["judges"], self.judges):
            j["band"] = js.band
            for s, ss in zip(j["sensitivity"], js.sensitivity):
                s["band"] = ss.band
        return _clean(d)


def _clean(x):
    """JSON has no NaN: turn NaN and infinities into None, recursively."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def _agreement(a: str, b: str, va: dict, vb: dict, ids: Sequence[str], classes: list[str],
               draws: int = 0, seed: object = 1, runs: tuple[Optional[int], Optional[int]] = (None, None)
               ) -> Agreement:
    pairs = []
    unpaired = 0
    for i in ids:
        x, y = va.get(i), vb.get(i)
        if x is None or y is None:
            unpaired += 1
        else:
            pairs.append((x, y))
    k = sum(x == y for x, y in pairs)
    n = len(pairs)
    lo, hi = wilson(k, n)
    cm = confusion(pairs, classes)
    per = []
    for i, c in enumerate(classes):
        na, nb = sum(cm[i]), sum(r[i] for r in cm)
        per.append(ClassAgreement(c, na, nb, cm[i][i], specific_agreement(cm, i), na + nb - 2 * cm[i][i]))
    klo = khi = float("nan")
    und = 0
    if draws and n:
        klo, khi, und = bootstrap_kappa(cm, draws, f"{seed}|{a}|{b}")
    return Agreement(a, b, k, n, k / n if n else float("nan"), lo, hi, kappa(pairs), cm, unpaired,
                     list(classes), klo, khi, und, draws if n else 0, runs[0], runs[1], per)


def _collect(j: Judge, items: Items, rub: Rubric, repeats: int, workers: int, on_error: str,
             on_verdict: Optional[Callable[[Verdict], None]] = None,
             progress: Optional[Callable[[str, int, int, int], None]] = None,
             ) -> list[list[Optional[Verdict]]]:
    """Every verdict for one judge, as out[repeat][item index], in item order.

    Judges that set `concurrent = True` are called from a bounded thread pool of `workers`
    threads; results are put back in item order whatever order they finish in. With
    on_error="raise" the first failure in item order is raised and queued calls are cancelled.
    `on_verdict` is called once per verdict the moment it arrives (in completion order, never
    two at once), so a caller can save paid calls before the run ends. `progress(judge,
    repeat, repeats, items)` is called when every item of a repeat has come back.
    """
    tasks = [(r, it) for r in range(repeats) for it in items]
    lock = threading.Lock()
    done = [0] * repeats
    n_items = len(items)

    def call(task):
        r, it = task
        try:
            v = j.judge(it, rub, r)
        except JudgeError as e:
            if on_error == "raise":
                raise
            v = Verdict(it.id, j.name, r, None, None, f"error: {e}", failed=True)
        with lock:
            if v is not None and on_verdict is not None:
                on_verdict(v)
            done[r] += 1
            if done[r] == n_items and progress is not None:
                progress(j.name, r, repeats, n_items)
        return v

    if getattr(j, "concurrent", False) and workers > 1 and len(tasks) > 1:
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"judgecal-{j.name}"[:40])
        futures = [pool.submit(call, t) for t in tasks]
        try:
            flat = [f.result() for f in futures]
        except BaseException:
            for f in futures:
                f.cancel()
            raise
        finally:
            pool.shutdown(wait=True)
    else:
        flat = [call(t) for t in tasks]
    n = len(items)
    return [flat[r * n:(r + 1) * n] for r in range(repeats)]


class Calibration:
    """Run every judge `repeats` times over every item and measure consistency."""

    @staticmethod
    def run(
        items: Items,
        judges: Sequence[Judge],
        repeats: int = 2,
        rubric: Optional[Rubric] = None,
        reference: Optional[str] = None,
        pair_repeat: int = 0,
        low_threshold: float = 50.0,
        on_error: str = "record",
        workers: int = 4,
        strata: Optional[dict[str, str]] = None,
        prices: Optional[dict[str, tuple[float, float]]] = None,
        boot_draws: int = 10_000,
        seed: object = 1,
        on_verdict: Optional[Callable[[Verdict], None]] = None,
        progress: Optional[Callable[[str, int, int, int], None]] = None,
    ) -> CalibrationResult:
        """
        reference:   the judge whose test-retest sets the noise band. Defaults to the first
                     judge that has a measurable retest.
        pair_repeat: which repeat is used when judges are compared with each other.
        on_error:    "record" keeps a failed call as an unreadable verdict; "raise" stops.
        workers:     thread-pool size for judges that allow concurrent calls (endpoint judges).
        strata:      item id -> stratum label, supplied by the user; agreement is also
                     reported within each stratum.
        prices:      judge name -> (USD per million input tokens, USD per million output tokens).
        boot_draws:  bootstrap draws for the kappa interval (0 turns the bootstrap off).
        seed:        bootstrap seed; the same seed gives the same intervals.
        on_verdict:  called with each verdict as it arrives (the CLI appends it to --verdicts-out).
        progress:    called as progress(judge, repeat, repeats, items) when a repeat completes.
        """
        if repeats < 1:
            raise ValueError("repeats must be at least 1")
        if not 0 <= pair_repeat < repeats:
            raise ValueError(f"pair_repeat must name one of the {repeats} repeats")
        if workers < 1:
            raise ValueError("workers must be at least 1")
        if boot_draws < 0:
            raise ValueError("boot_draws must be 0 or more")
        if on_error not in ("record", "raise"):
            raise ValueError('on_error must be "record" or "raise"')
        if len(items) == 0:
            raise ValueError("no items to judge")
        if not judges:
            raise ValueError("give at least one judge")
        names = [j.name for j in judges]
        if len(set(names)) != len(names):
            raise ValueError("judge names must be unique")
        prices = dict(prices or {})
        for name, (pi, po) in prices.items():
            if name not in names:
                raise ValueError(f"price given for {name!r}, which is not one of the judges")
            if not all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in (pi, po)):
                raise ValueError(f"prices for {name!r} must be finite numbers of 0 or more")
        if reference is not None and reference not in names:
            raise ValueError(f"reference {reference!r} is not one of the judges")
        rub = rubric if rubric is not None else _OpenRubric()
        ids = items.ids()
        all_verdicts: list[Verdict] = []
        # by_judge[name][repeat] = {item id: class}
        by_judge: dict[str, list[dict[str, str]]] = {}
        summaries = []
        for j in judges:
            got_all = _collect(j, items, rub, repeats, workers, on_error, on_verdict, progress)
            reps, unread, miss, lows, mixes = [], [], [], [], []
            s = JudgeSummary(j.name, [], unread, miss, lows, mixes, replayed=bool(getattr(j, "replayed", False)))
            for r in range(repeats):
                got: dict[str, str] = {}
                mix: Counter = Counter()
                n_unread = n_miss = n_low = n_conf = 0
                for it, v in zip(items, got_all[r]):
                    if v is None:
                        n_miss += 1
                        continue
                    all_verdicts.append(v)
                    if v.failed:
                        s.failed += 1
                    else:
                        s.calls += 1
                        if v.input_tokens is not None and v.output_tokens is not None:
                            s.calls_with_usage += 1
                            s.input_tokens += v.input_tokens
                            s.output_tokens += v.output_tokens
                    if v.cls is None:
                        n_unread += 1
                        continue
                    got[it.id] = v.cls
                    mix[v.cls] += 1
                    n_conf += 1
                    n_low += is_low(v.confidence, low_threshold)
                reps.append(got)
                unread.append(n_unread)
                miss.append(n_miss)
                lows.append(Share(n_low, n_conf))
                mixes.append(dict(mix))
            s.verdicts = [len(g) for g in reps]
            by_judge[j.name] = reps
            summaries.append(s)

        classes = list(rub.classes)
        for s in summaries:
            reps = by_judge[s.name]
            for r1, r2 in combinations(range(repeats), 2):
                if reps[r1] and reps[r2]:
                    s.retest.append(_agreement(f"{s.name}#{r1 + 1}", f"{s.name}#{r2 + 1}",
                                               reps[r1], reps[r2], ids, classes, boot_draws, seed,
                                               (r1 + 1, r2 + 1)))
            if any(a.run_b >= 3 for a in s.retest):
                for m in range(2, repeats + 1):
                    use = [a for a in s.retest if a.run_b <= m]
                    if use:
                        s.sensitivity.append(RunCountBand(m, len(use), sum(a.disagreements for a in use),
                                                          sum(a.n for a in use)))
                if len(s.sensitivity) < 2:
                    s.sensitivity = []
        pairwise = []
        for a, b in combinations(names, 2):
            va, vb = by_judge[a][pair_repeat], by_judge[b][pair_repeat]
            if va and vb:
                pairwise.append(_agreement(a, b, va, vb, ids, classes, boot_draws, seed))

        strata_out: list[StratumResult] = []
        ignored = 0
        if strata:
            idset = set(ids)
            ignored = sum(1 for k in strata if k not in idset)
            groups: dict[str, list[str]] = {}
            for i in ids:
                groups.setdefault(strata.get(i, NO_STRATUM), []).append(i)
            for label, sids in groups.items():
                ags = []
                for s in summaries:
                    if s.headline_retest:
                        h = s.headline_retest
                        reps = by_judge[s.name]
                        ags.append(_agreement(h.a, h.b, reps[h.run_a - 1], reps[h.run_b - 1], sids, classes,
                                              runs=(h.run_a, h.run_b)))
                for p in pairwise:
                    ags.append(_agreement(p.a, p.b, by_judge[p.a][pair_repeat], by_judge[p.b][pair_repeat],
                                          sids, classes))
                strata_out.append(StratumResult(label, len(sids), ags))

        if reference is None:
            reference = next((s.name for s in summaries if s.band is not None), None)
        fixtures = [j.name for j in judges if getattr(j, "fixture", False)]
        return CalibrationResult(rub.name, classes, len(ids), repeats, pair_repeat, summaries,
                                 pairwise, reference, all_verdicts, fixtures, strata_out, ignored,
                                 prices, boot_draws, seed)
