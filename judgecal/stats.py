"""Agreement statistics.

The core maths is lifted from the analysis script of a registered judge-reliability study
(October 2026): raw agreement with a Wilson 95% interval, Cohen's kappa, and a square
confusion table. The bootstrap, the binomial sampler and specific agreement are additions
of this package. Standard library only.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from typing import Iterable, Optional, Sequence

Pair = tuple[str, str]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def kappa(pairs: Sequence[Pair]) -> float:
    """Cohen's kappa for two raters over the same items."""
    n = len(pairs)
    if n == 0:
        return float("nan")
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[c] * cb[c] for c in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def confusion(pairs: Iterable[Pair], classes: Sequence[str]) -> list[list[int]]:
    """Rows are rater A, columns rater B, in the order of `classes`."""
    m = Counter(pairs)
    return [[m[(a, b)] for b in classes] for a in classes]


def kappa_from_table(table: Sequence[Sequence[int]]) -> float:
    """Cohen's kappa from a square confusion table; equals kappa() on the same pairs."""
    n = sum(map(sum, table))
    if n == 0:
        return float("nan")
    po = sum(table[i][i] for i in range(len(table))) / n
    rows = [sum(r) for r in table]
    cols = [sum(c) for c in zip(*table)]
    pe = sum(r * c for r, c in zip(rows, cols)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def noise_band(retest_agreement: float) -> float:
    """Half the observed test-retest disagreement.

    The convention comes from a registered judge-reliability study, October 2026. It says how
    much judge noise to print beside a judged number.
    """
    return (1.0 - retest_agreement) / 2.0


def binomial(rng: random.Random, n: int, p: float) -> int:
    """An exact Binomial(n, p) draw that uses only rng.random().

    A copy of CPython 3.12's random.binomialvariate (Devroye's geometric method for
    n*p < 10, Hormann's BTRS above), so a seeded draw gives the same value on Python 3.9
    as on 3.14. Python Software Foundation licence.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if p <= 0.0 or p >= 1.0:
        if p == 0.0:
            return 0
        if p == 1.0:
            return n
        raise ValueError("p must be in the range 0.0 <= p <= 1.0")
    rand = rng.random
    if n == 1:
        return int(rand() < p)
    if p > 0.5:
        return n - binomial(rng, n, 1.0 - p)
    if n * p < 10.0:
        x = y = 0
        c = math.log2(1.0 - p)
        if not c:
            return x
        while True:
            try:
                y += math.floor(math.log2(rand()) / c) + 1
            except ValueError:
                continue
            if y > n:
                return x
            x += 1
    setup = False
    spq = math.sqrt(n * p * (1.0 - p))
    b = 1.15 + 2.53 * spq
    a = -0.0873 + 0.0248 * b + 0.01 * p
    c = n * p + 0.5
    vr = 0.92 - 4.2 / b
    while True:
        u = rand() - 0.5
        us = 0.5 - math.fabs(u)
        k = math.floor((2.0 * a / us + b) * u + c)
        if k < 0 or k > n:
            continue
        v = rand()
        if us >= 0.07 and v <= vr:
            return k
        if not setup:
            alpha = (2.83 + 5.1 / b) * spq
            lpq = math.log(p / (1.0 - p))
            m = math.floor((n + 1) * p)
            h = math.lgamma(m + 1) + math.lgamma(n - m + 1)
            setup = True
        v *= alpha / (a / (us * us) + b)
        if math.log(v) <= h - math.lgamma(k + 1) - math.lgamma(n - k + 1) + (k - m) * lpq:
            return k


def multinomial(rng: random.Random, n: int, weights: Sequence[int]) -> list[int]:
    """One Multinomial(n, weights / sum(weights)) draw, by conditional binomials."""
    total = sum(weights)
    out = []
    left = n
    for w in weights:
        if left == 0 or total <= 0:
            out.append(0)
            continue
        x = binomial(rng, left, min(1.0, w / total))
        out.append(x)
        left -= x
        total -= w
    return out


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (the 'type 7' rule) of an already sorted list."""
    if not sorted_values:
        return float("nan")
    h = (len(sorted_values) - 1) * q
    lo = math.floor(h)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (h - lo) * (sorted_values[hi] - sorted_values[lo])


def bootstrap_kappa(
    table: Sequence[Sequence[int]],
    draws: int = 10_000,
    seed: object = 1,
    level: float = 0.95,
) -> tuple[float, float, int]:
    """Percentile bootstrap interval for Cohen's kappa, resampling items with replacement.

    Resampling n paired items with replacement is the same as one multinomial draw over the
    cells of the confusion table, which is what this does: exact, and cheap on large n.
    Returns (lo, hi, undefined), where `undefined` counts draws whose kappa does not exist
    (every resampled verdict in one class); those draws are left out of the percentiles.
    `seed` may be any int or str; the same seed gives the same interval on every Python.
    """
    cells = [v for row in table for v in row]
    n = sum(cells)
    if n == 0 or draws <= 0:
        return (float("nan"), float("nan"), 0)
    if isinstance(seed, int):
        seed = str(seed)
    rng = random.Random(seed if isinstance(seed, str) else repr(seed))
    k = len(table)
    vals = []
    undefined = 0
    for _ in range(draws):
        x = multinomial(rng, n, cells)
        t = [x[i * k:(i + 1) * k] for i in range(k)]
        v = kappa_from_table(t)
        if v != v:
            undefined += 1
        else:
            vals.append(v)
    vals.sort()
    a = (1.0 - level) / 2.0
    return (percentile(vals, a), percentile(vals, 1.0 - a), undefined)


def specific_agreement(table: Sequence[Sequence[int]], i: int) -> Optional[float]:
    """Positive specific agreement for class i: 2 * both / (named by A + named by B).

    The share of verdicts naming the class, on either side, that the other side matched.
    None when neither side named the class.
    """
    row = sum(table[i])
    col = sum(r[i] for r in table)
    if row + col == 0:
        return None
    return 2.0 * table[i][i] / (row + col)
