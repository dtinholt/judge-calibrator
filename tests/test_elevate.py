"""Tests for the 0.2.0 additions: bootstrap kappa, per-class agreement, strata, repeat
sensitivity, concurrency and timeouts, cost, drift, the HTML summary and print sheet.
Fixtures only; no network. Run: python3 -W error -m unittest discover -s tests -v"""
from __future__ import annotations

import io
import json
import math
import os
import random
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from judgecal import (  # noqa: E402
    Calibration, ChatCompletionsJudge, Items, JudgeError, ReplayJudge, Rubric, bootstrap_kappa,
    compare, kappa, load_strata, specific_agreement,
)
from judgecal.cli import main as cli_main, parse_price  # noqa: E402
from judgecal.compare import drift_sentence, gap_between, load_session  # noqa: E402
from judgecal.items import Item  # noqa: E402
from judgecal.report import (  # noqa: E402
    drift_html, drift_text, sensitivity_note, to_html, to_text, wavers_paragraph,
)
from judgecal.stats import binomial, confusion, kappa_from_table, multinomial, percentile  # noqa: E402

FX = ROOT / "tests" / "fixtures"
STUDY = FX / "judge-study"
EL = FX / "elevate"
GATE = ["INFO", "REASON", "DONE", "BLOCKED"]
PF = Rubric(["PASS", "FAIL"])


def study(**kw):
    items = Items.load(STUDY / "items.jsonl")
    judges = [ReplayJudge(STUDY / f"{n}.jsonl", n) for n in ("frontier", "mid", "small")]
    return Calibration.run(items, judges, repeats=2, rubric=Rubric(GATE), **kw)


def run_cli(args):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = cli_main(args)
    return rc, out.getvalue(), err.getvalue()


# ---------------------------------------------------------------- (1) bootstrap

@unittest.skipUnless(STUDY.exists(), "needs the unpublished study fixture")
class TestBootstrap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = study()

    def test_binomial_edges_and_range(self):
        rng = random.Random("x")
        self.assertEqual(binomial(rng, 0, 0.3), 0)
        self.assertEqual(binomial(rng, 7, 0.0), 0)
        self.assertEqual(binomial(rng, 7, 1.0), 7)
        for n, p in ((1, 0.5), (5, 0.9), (40, 0.1), (1000, 0.3), (1000, 0.97)):
            for _ in range(200):
                self.assertTrue(0 <= binomial(rng, n, p) <= n)
        with self.assertRaises(ValueError):
            binomial(rng, -1, 0.5)
        with self.assertRaises(ValueError):
            binomial(rng, 3, 1.5)

    def test_binomial_mean(self):
        rng = random.Random("mean")
        for n, p in ((50, 0.1), (2000, 0.4)):
            m = sum(binomial(rng, n, p) for _ in range(4000)) / 4000
            self.assertAlmostEqual(m, n * p, delta=0.05 * n * p + 0.2)

    def test_multinomial_sums(self):
        rng = random.Random(3)
        for _ in range(100):
            x = multinomial(rng, 197, [53, 0, 111, 9, 14, 6, 4])
            self.assertEqual(sum(x), 197)
            self.assertEqual(x[1], 0)  # an empty cell stays empty
        self.assertEqual(multinomial(rng, 0, [1, 2]), [0, 0])

    def test_percentile(self):
        self.assertEqual(percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.5), 3.0)
        self.assertAlmostEqual(percentile([0.0, 10.0], 0.25), 2.5)
        self.assertTrue(math.isnan(percentile([], 0.5)))

    def test_kappa_from_table_matches_pairs(self):
        a = self.res.judge("frontier").headline_retest
        self.assertAlmostEqual(kappa_from_table(a.confusion), a.kappa)
        self.assertTrue(math.isnan(kappa_from_table([[0, 0], [0, 0]])))

    def test_interval_pinned_and_seeded(self):
        a = self.res.judge("frontier").headline_retest
        # pinned on 3.9.6 and 3.14.5 alike: the sampler uses only random.random()
        self.assertEqual(f"[{a.kappa_lo:.3f}, {a.kappa_hi:.3f}]", "[0.752, 0.894]")
        self.assertEqual(a.boot_draws, 10_000)
        self.assertLess(a.kappa_lo, a.kappa)
        self.assertGreater(a.kappa_hi, a.kappa)
        again = bootstrap_kappa(a.confusion, 10_000, "1|frontier#1|frontier#2")
        self.assertEqual((again[0], again[1]), (a.kappa_lo, a.kappa_hi))
        other = bootstrap_kappa(a.confusion, 2000, "another seed")
        self.assertNotEqual(other[:2], (a.kappa_lo, a.kappa_hi))

    def test_undefined_and_off(self):
        lo, hi, und = bootstrap_kappa([[5, 0], [0, 0]], 300, 1)
        self.assertEqual(und, 300)
        self.assertTrue(math.isnan(lo) and math.isnan(hi))
        self.assertTrue(math.isnan(bootstrap_kappa([[3, 1], [1, 3]], 0)[0]))
        self.assertTrue(math.isnan(bootstrap_kappa([[0, 0], [0, 0]], 100)[0]))
        res = study(boot_draws=0)
        a = res.judge("frontier").headline_retest
        self.assertEqual(a.boot_draws, 0)
        self.assertNotIn("bootstrap 95%", to_text(res).split("TEST-RETEST")[1].split("\n")[1])
        with self.assertRaises(ValueError):
            study(boot_draws=-1)

    def test_printed_beside_wilson(self):
        txt = to_text(self.res)
        self.assertIn("kappa 0.827; paired 197 / unpaired 0; kappa bootstrap 95% [0.752, 0.894]", txt)
        self.assertIn("10,000 draws, seed 1", txt)
        page = to_html(self.res)
        self.assertIn('0.827 <span class="ci">[0.752, 0.894]</span>', page)

    def test_specific_agreement(self):
        self.assertEqual(specific_agreement([[3, 1], [1, 5]], 0), 0.75)
        self.assertIsNone(specific_agreement([[0, 0], [0, 4]], 0))


# ---------------------------------------------------------------- (2) per class

@unittest.skipUnless(STUDY.exists(), "needs the unpublished study fixture")
class TestPerClass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a = study().judge("frontier").headline_retest

    def test_per_class_rows(self):
        rows = {c.cls: (c.named_a, c.named_b, c.both, c.flips) for c in self.a.per_class}
        self.assertEqual(rows, {"INFO": (53, 59, 53, 6), "REASON": (111, 124, 111, 13),
                                "DONE": (23, 14, 14, 9), "BLOCKED": (10, 0, 0, 10)})
        self.assertEqual(sum(c.flips for c in self.a.per_class), 2 * self.a.disagreements)
        self.assertEqual(f"{self.a.per_class[2].specific:.3f}", "0.757")

    def test_most_flipped_and_pair(self):
        self.assertEqual(self.a.most_flipped(), (["REASON"], 13))
        self.assertEqual(self.a.top_flip_pair(), ("REASON", "DONE", 0, 9))

    def test_wavers_paragraph_from_data(self):
        p = wavers_paragraph("frontier", self.a)
        self.assertIn("19 of 197 paired items", p)
        self.assertIn("REASON took part in 13 of those flips", p)
        self.assertIn("DONE on run 1 becoming REASON on run 2, 9 items and none in reverse", p)
        self.assertIn("0.000 for BLOCKED (10 verdicts named it)", p)
        self.assertNotIn("—", p)

    def test_wavers_edge_cases(self):
        ids = ["a", "b", "c", "d"]
        same = {"a": "P", "b": "F", "c": "P", "d": "F"}
        from judgecal.calibration import _agreement
        a0 = _agreement("x#1", "x#2", same, same, ids, ["P", "F"], runs=(1, 2))
        self.assertIn("No class wavered", wavers_paragraph("x", a0))
        self.assertEqual(a0.most_flipped(), ([], 0))
        self.assertIsNone(a0.top_flip_pair())
        empty = _agreement("x#1", "x#2", {}, {}, ids, ["P", "F"])
        self.assertIn("no item with a readable verdict", wavers_paragraph("x", empty))
        both = _agreement("x#1", "x#2", same, {"a": "F", "b": "P", "c": "P", "d": "F"}, ids, ["P", "F"])
        p = wavers_paragraph("x", both)
        self.assertIn("P and F tie for the most flips at 2 each", p)
        self.assertIn("1 went from P on run 1 to F on run 2, 1 the other way", p)

    def test_html_section(self):
        page = to_html(study())
        self.assertIn("Where the judge wavers", page)
        self.assertIn("FRONTIER, RUN 1 VS RUN 2, PER CLASS".lower(), page.lower())


# ---------------------------------------------------------------- (3) strata

class TestStrata(unittest.TestCase):
    def setUp(self):
        self.items = Items.load(EL / "items.jsonl")
        self.j = ReplayJudge(EL / "judge4.jsonl", "j")
        self.strata = load_strata(EL / "strata.jsonl")

    def test_loader(self):
        self.assertEqual(self.strata["e0"], "short")
        self.assertEqual(self.strata["ghost"], "long")
        with tempfile.TemporaryDirectory() as d:
            for body, msg in (('{"id":"a","stratum":"x"}\n{"id":"a","stratum":"y"}\n', "duplicate"),
                              ('{"id":"a"}\n', "no 'stratum'"),
                              ('{"id":"a","stratum":"  "}\n', "non-empty"),
                              ('{"stratum":"x"}\n', "id"),
                              ("", "no strata rows")):
                p = Path(d) / "s.jsonl"
                p.write_text(body)
                with self.assertRaisesRegex(ValueError, msg):
                    load_strata(p)

    def test_per_stratum_agreement(self):
        res = Calibration.run(self.items, [self.j], repeats=2, rubric=PF, strata=self.strata)
        self.assertEqual(res.strata_ignored, 1)
        got = {s.label: (s.n_items, s.agreements[0].k, s.agreements[0].n) for s in res.strata}
        self.assertEqual(got, {"short": (5, 4, 5), "long": (4, 4, 4), "(no stratum)": (1, 1, 1)})
        txt = to_text(res)
        self.assertIn("STRATA (3", txt)
        self.assertIn("1 rows name items not in this run", txt)
        page = to_html(res)
        self.assertIn("Strata", page)
        self.assertIn("never derives them", page)
        self.assertIn("lowest in short: 4 of 5", page)

    def test_strata_include_pairwise(self):
        k = ReplayJudge(EL / "judge4.jsonl", "k")
        res = Calibration.run(self.items, [self.j, k], repeats=2, rubric=PF, strata=self.strata)
        names = [(a.a, a.b) for a in res.strata[0].agreements]
        self.assertIn(("j", "k"), names)

    def test_cli(self):
        rc, out, _ = run_cli(["report", "--items", str(EL / "items.jsonl"), "--classes", "PASS,FAIL",
                              "--replay", f"j={EL / 'judge4.jsonl'}", "--strata", str(EL / "strata.jsonl"),
                              "--boot-draws", "0"])
        self.assertEqual(rc, 0)
        self.assertIn("short (5 items)", out)


# ---------------------------------------------------------------- (4) repeat sensitivity

class TestSensitivity(unittest.TestCase):
    def setUp(self):
        self.items = Items.load(EL / "items.jsonl")
        self.j = ReplayJudge(EL / "judge4.jsonl", "j")

    def test_every_pair_and_pooled_bands(self):
        res = Calibration.run(self.items, [self.j], repeats=4, rubric=PF, boot_draws=200)
        js = res.judge("j")
        self.assertEqual([(a.run_a, a.run_b, a.disagreements) for a in js.retest],
                         [(1, 2, 1), (1, 3, 1), (1, 4, 1), (2, 3, 2), (2, 4, 2), (3, 4, 2)])
        self.assertEqual([(s.runs, s.pairs, s.disagreements, s.paired) for s in js.sensitivity],
                         [(2, 1, 1, 10), (3, 3, 4, 30), (4, 6, 9, 60)])
        self.assertAlmostEqual(js.sensitivity[-1].band, 0.075)
        note = sensitivity_note(js)
        self.assertIn("moves from ±0.05 with 2 runs", note)
        self.assertIn("bands from ±0.050 to ±0.100", note)
        self.assertIn("REPEAT SENSITIVITY", to_text(res))
        self.assertIn("Repeat sensitivity", to_html(res))
        d = res.to_dict()
        self.assertAlmostEqual(d["judges"][0]["sensitivity"][1]["band"], 4 / 30 / 2)

    def test_holds(self):
        res = Calibration.run(self.items, [self.j], repeats=3, rubric=PF, boot_draws=0)
        js = res.judge("j")
        js.sensitivity[-1].disagreements = 3  # 3/30/2 = 0.05, same as 1/10/2 rounded
        self.assertIn("holds at ±0.05 from 2 runs to 3", sensitivity_note(js))

    def test_not_with_two_runs(self):
        res = Calibration.run(self.items, [self.j], repeats=2, rubric=PF, boot_draws=0)
        self.assertEqual(res.judge("j").sensitivity, [])
        self.assertIsNone(sensitivity_note(res.judge("j")))
        self.assertNotIn("REPEAT SENSITIVITY", to_text(res))


# ---------------------------------------------------------------- (5) concurrency and timeouts

class KeyedTransport:
    """Answers by item text, after a random delay, and tracks how many calls run at once."""

    def __init__(self, answer, delay=0.01, usage=None):
        self.answer, self.delay, self.usage = answer, delay, usage
        self.active = self.peak = 0
        self.lock = threading.Lock()
        self.threads = set()

    def __call__(self, url, headers, body):
        text = json.loads(body)["messages"][1]["content"]
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.threads.add(threading.current_thread().name)
        time.sleep(random.random() * self.delay)
        with self.lock:
            self.active -= 1
        msg = {"choices": [{"message": {"content": self.answer(text)}}]}
        if self.usage is not None:
            msg["usage"] = self.usage(text)
        return 200, json.dumps(msg).encode()


def ans(text):
    n = int(text.split()[1])
    return "CLASS: PASS" if n % 3 else "CLASS: FAIL"


class TestConcurrency(unittest.TestCase):
    def setUp(self):
        self.items = Items([Item(f"i{n}", f"item {n}") for n in range(40)])

    def test_order_preserved_and_bounded(self):
        t = KeyedTransport(ans, delay=0.01)
        j = ChatCompletionsJudge("http://judge.invalid/v1", "m", None, transport=t)
        res = Calibration.run(self.items, [j], repeats=2, rubric=PF, workers=3, boot_draws=0)
        self.assertLessEqual(t.peak, 3)
        self.assertGreater(t.peak, 1)
        got = [v.cls for v in res.verdicts if v.repeat == 0]
        self.assertEqual(got, ["FAIL" if n % 3 == 0 else "PASS" for n in range(40)])
        self.assertEqual([v.item for v in res.verdicts[:40]], self.items.ids())
        self.assertEqual(res.judge("m").headline_retest.k, 40)

    def test_default_four_and_sequential_one(self):
        t = KeyedTransport(ans, delay=0.005)
        j = ChatCompletionsJudge("http://judge.invalid/v1", "m", None, transport=t)
        Calibration.run(self.items, [j], repeats=1, rubric=PF, boot_draws=0)
        self.assertLessEqual(t.peak, 4)
        t1 = KeyedTransport(ans, delay=0.0)
        j1 = ChatCompletionsJudge("http://judge.invalid/v1", "m", None, transport=t1)
        Calibration.run(self.items, [j1], repeats=1, rubric=PF, workers=1, boot_draws=0)
        self.assertEqual(t1.peak, 1)
        self.assertEqual(t1.threads, {threading.main_thread().name})
        with self.assertRaises(ValueError):
            Calibration.run(self.items, [j1], rubric=PF, workers=0)

    def test_replay_judges_stay_sequential(self):
        self.assertFalse(getattr(ReplayJudge(EL / "judge4.jsonl", "j"), "concurrent", False))
        self.assertTrue(ChatCompletionsJudge.concurrent)

    def test_raise_in_pool(self):
        def t(url, headers, body):
            n = int(json.loads(body)["messages"][1]["content"].split()[1])
            return (401, b"no") if n == 5 else (200, json.dumps(
                {"choices": [{"message": {"content": "CLASS: PASS"}}]}).encode())
        j = ChatCompletionsJudge("http://judge.invalid/v1", "m", None, transport=t)
        with self.assertRaises(JudgeError):
            Calibration.run(self.items, [j], repeats=1, rubric=PF, on_error="raise", workers=4)
        res = Calibration.run(self.items, [j], repeats=1, rubric=PF, workers=4, boot_draws=0)
        self.assertEqual(res.judge("m").failed, 1)
        self.assertTrue(res.verdicts[5].failed)
        with self.assertRaises(ValueError):
            Calibration.run(self.items, [j], rubric=PF, on_error="ignore")

    def test_item_timeout_with_fake_clock(self):
        now = [0.0]
        replies = [(503, b""), (503, b""), (200, b"")]

        def t(url, headers, body):
            return replies.pop(0)

        def sleep(s):
            now[0] += s
        j = ChatCompletionsJudge("http://judge.invalid/v1", "m", None, transport=t, sleep=sleep,
                                 item_timeout=5, clock=lambda: now[0])
        with self.assertRaisesRegex(JudgeError, "item timeout"):
            j.judge(Item("x", "y"), PF, 0)
        self.assertEqual(now[0], 2.0)  # slept 2, then the 4 s backoff would cross 5 s

    def test_timeout_before_first_attempt_and_socket_timeout(self):
        now = [0.0]
        seen = []

        def t(url, headers, body, timeout=None):
            seen.append(timeout)
            now[0] += 1.5
            return 200, json.dumps({"choices": [{"message": {"content": "CLASS: PASS"}}]}).encode()
        t.accepts_timeout = True
        j = ChatCompletionsJudge("http://judge.invalid/v1", "m", None, transport=t, item_timeout=10,
                                 clock=lambda: now[0])
        self.assertEqual(j.judge(Item("x", "y"), PF, 0).cls, "PASS")
        self.assertEqual(seen, [10.0])
        j2 = ChatCompletionsJudge("http://judge.invalid/v1", "m", None, transport=t, item_timeout=1,
                                  clock=lambda: now[0])
        now[0] = 0.0
        clock_jump = iter([0.0, 5.0, 5.0, 5.0])
        j2.clock = lambda: next(clock_jump)
        with self.assertRaisesRegex(JudgeError, "item timeout"):
            j2.judge(Item("x", "y"), PF, 0)

    def test_constructor_validation(self):
        for kw in ({"temperature": float("nan")}, {"temperature": -1}, {"item_timeout": 0},
                   {"item_timeout": float("inf")}, {"retries": -1}):
            with self.assertRaises(ValueError):
                ChatCompletionsJudge("http://judge.invalid/v1", "m", None, **kw)
        j = ChatCompletionsJudge("http://judge.invalid/v1", "m", None, item_timeout=None,
                                 transport=lambda u, h, b: (200, b'{"choices":[{"message":{"content":"CLASS: FAIL"}}]}'))
        self.assertEqual(j.judge(Item("x", "y"), PF, 0).cls, "FAIL")

    def test_default_transport_accepts_timeout(self):
        from judgecal.judges import urllib_transport
        self.assertTrue(urllib_transport().accepts_timeout)


# ---------------------------------------------------------------- (6) cost

def usage(text):
    n = int(text.split()[1])
    return {"prompt_tokens": 1000 + n, "completion_tokens": 100}


class TestCost(unittest.TestCase):
    def setUp(self):
        self.items = Items([Item(f"i{n}", f"item {n}") for n in range(4)])

    def judge(self, usage_fn, name="big"):
        return ChatCompletionsJudge("http://judge.invalid/v1", "m", None, name=name,
                                    transport=KeyedTransport(ans, 0, usage_fn))

    def test_cost_from_usage(self):
        res = Calibration.run(self.items, [self.judge(usage)], repeats=2, rubric=PF,
                              prices={"big": (3.0, 15.0)}, boot_draws=0)
        js = res.judge("big")
        self.assertEqual((js.calls, js.calls_with_usage, js.input_tokens, js.output_tokens), (8, 8, 8012, 800))
        c = res.costs()[0]
        self.assertAlmostEqual(c.cost, (8012 * 3 + 800 * 15) / 1e6)
        line = res.cost_line()
        self.assertIn("big $0.0360 (8,012 in / 800 out tokens at $3 / $15 per million)", line)
        self.assertIn("Total $0.0360.", line)
        self.assertIn("cost: big $0.0360", to_text(res))
        self.assertIn("big $0.0360", to_html(res))
        self.assertAlmostEqual(res.to_dict()["costs"][0]["cost"], c.cost)

    def test_unknown_partial_and_unpriced(self):
        res = Calibration.run(self.items, [self.judge(None)], repeats=1, rubric=PF,
                              prices={"big": (3.0, 15.0)}, boot_draws=0)
        self.assertIn("big unknown (no token counts returned). Total unknown.", res.cost_line())
        half = lambda t: usage(t) if int(t.split()[1]) % 2 else {"prompt_tokens": "x"}  # noqa: E731
        both = [self.judge(half), self.judge(usage, "cheap"), self.judge(None, "free")]
        res = Calibration.run(self.items, both, repeats=1, rubric=PF, boot_draws=0,
                              prices={"big": (1, 1), "free": (1, 1)})
        line = res.cost_line()
        self.assertIn("big at least", line)
        self.assertIn("token counts missing on 2 of 4 calls", line)
        self.assertIn("cheap price not given", line)
        self.assertIn("unknown for free", line)
        self.assertIn("Total at least", line)
        res = Calibration.run(self.items, [self.judge(usage)], repeats=1, rubric=PF, boot_draws=0)
        self.assertIn("no prices given", res.cost_line())

    def test_price_validation(self):
        with self.assertRaises(ValueError):
            Calibration.run(self.items, [self.judge(usage)], rubric=PF, prices={"zz": (1, 1)})
        with self.assertRaises(ValueError):
            Calibration.run(self.items, [self.judge(usage)], rubric=PF, prices={"big": (-1, 1)})
        self.assertEqual(parse_price("big=3,15"), ("big", (3.0, 15.0)))
        for bad in ("big", "big=3", "=3,4", "big=a,b", "big=nan,1", "big=1,-2"):
            with self.assertRaises(ValueError):
                parse_price(bad)

    def test_failed_calls_not_counted(self):
        def t(url, headers, body):
            n = int(json.loads(body)["messages"][1]["content"].split()[1])
            if n == 0:
                return 400, b"bad"
            return 200, json.dumps({"choices": [{"message": {"content": "CLASS: PASS"}}],
                                    "usage": {"prompt_tokens": 10, "completion_tokens": 2}}).encode()
        j = ChatCompletionsJudge("http://judge.invalid/v1", "m", None, name="big", transport=t)
        res = Calibration.run(self.items, [j], repeats=1, rubric=PF, prices={"big": (1, 1)}, boot_draws=0)
        self.assertEqual((res.judge("big").calls, res.judge("big").failed), (3, 1))
        self.assertIn("1 failed calls not counted", res.cost_line())

    def test_tokens_survive_verdicts_out_and_replay(self):
        with tempfile.TemporaryDirectory() as d:
            items = Path(d) / "items.jsonl"
            items.write_text("".join(json.dumps({"id": f"i{n}", "text": f"item {n}"}) + "\n" for n in range(4)))
            v = Path(d) / "v.jsonl"
            # write a verdict file as --verdicts-out would, then replay it with a price
            res = Calibration.run(Items.load(items), [self.judge(usage)], repeats=1, rubric=PF, boot_draws=0)
            with open(v, "w") as f:
                for x in res.verdicts:
                    f.write(json.dumps({"item": x.item, "judge": x.judge, "repeat": x.repeat, "class": x.cls,
                                        "input_tokens": x.input_tokens, "output_tokens": x.output_tokens}) + "\n")
            rc, out, _ = run_cli(["report", "--items", str(items), "--classes", "PASS,FAIL",
                                  "--replay", f"big={v}", "--price", "big=3,15", "--repeats", "1"])
            self.assertEqual(rc, 0)
            self.assertIn("recorded counts from the original run, replayed here at no cost", out)
            self.assertIn("Total $0.", out)


# ---------------------------------------------------------------- (7) compare / drift

A = EL / "verdicts-2026-12-31.jsonl"
B = EL / "verdicts-2027-01-01.jsonl"


class TestCompare(unittest.TestCase):
    def test_drift_across_year_boundary(self):
        d = compare(A, B, rubric=PF, boot_draws=500)
        self.assertEqual((d.date_a, d.date_b, d.gap), ("2026-12-31", "2027-01-01", "1 day"))
        j = d.judges[0]
        self.assertEqual((j.across.k, j.across.n), (7, 10))
        self.assertAlmostEqual(j.band, 0.15)
        self.assertEqual((j.within_a.k, j.within_a.n), (9, 10))
        self.assertIsNone(j.within_b)
        s = drift_sentence(j)
        self.assertIn("7 of 10 paired items", s)
        self.assertIn("sits inside that interval", s)
        txt = drift_text(d)
        self.assertIn("DRIFT: test-retest across sessions", txt)
        page = drift_html(d, date="2027-01-02")
        self.assertIn("test-retest across sessions", page)
        self.assertIn("registered judge-reliability study", page)
        self.assertIn("@media print", page)
        self.assertNotIn("<script", page)
        json.dumps(d.to_dict(), allow_nan=False)

    def test_gap_rules(self):
        self.assertEqual(gap_between("2026-03-07", "2026-03-09")[0], "2 days")  # US DST starts 8 Mar
        g, _ = gap_between("2026-03-08T01:30:00-06:00", "2026-03-08T03:30:00-05:00")
        self.assertEqual(g, "1.0 hours")  # one real hour across the spring-forward jump
        g, _ = gap_between("2026-10-31T12:00:00Z", "2026-11-02T12:00:00-06:00")
        self.assertEqual(g, "2.2 days")
        g, n = gap_between("2026-01-01T00:00:00", "2026-01-02T00:00:00+00:00")
        self.assertIsNone(g)
        self.assertIn("UTC offset", n[0])
        self.assertIn("same date", gap_between("2026-10-10", "2026-10-10")[1][0])
        g, n = gap_between("2026-10-10", "2026-10-01")
        self.assertIn("B before A", g)
        self.assertIn("dated before", n[0])
        self.assertIn("not ISO 8601", gap_between("tuesday", "wednesday")[1][0])
        self.assertEqual(gap_between(None, "2026-10-10"), (None, []))
        self.assertEqual(gap_between("2026-10-09", "2026-10-10T08:00:00")[0], "1 day")

    def test_errors_and_notes(self):
        with self.assertRaisesRegex(ValueError, "not in both"):
            compare(A, B, judges=["zz"])
        with self.assertRaisesRegex(ValueError, "no judge has run 2"):
            compare(A, B, repeat=1)
        with self.assertRaises(ValueError):
            compare(A, B, repeat=-1)
        with tempfile.TemporaryDirectory() as d:
            o = Path(d) / "o.jsonl"
            o.write_text('{"item":"e0","judge":"other","class":"PASS"}\n')
            with self.assertRaisesRegex(ValueError, "share no judge"):
                compare(A, o)
            m = Path(d) / "m.jsonl"
            m.write_text('{"item":"e0","judge":"j","class":"PASS","date":"2026-01-01"}\n'
                         '{"item":"e1","judge":"j","class":"FAIL","date":"2026-01-02"}\n')
            d2 = compare(A, m)
            self.assertTrue(any("2 dates" in n for n in d2.notes))
            self.assertTrue(any("session date is missing" in n for n in d2.notes))
            self.assertEqual(d2.classes, ["PASS", "FAIL"])
            dup = Path(d) / "dup.jsonl"
            dup.write_text('{"item":"e0","judge":"j","class":"PASS"}\n{"item":"e0","judge":"j","class":"PASS"}\n')
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_session(dup)
            (Path(d) / "e.jsonl").write_text("\n")
            with self.assertRaisesRegex(ValueError, "no verdict rows"):
                load_session(Path(d) / "e.jsonl")

    def test_cli_compare(self):
        with tempfile.TemporaryDirectory() as d:
            h, js = Path(d) / "d.html", Path(d) / "d.json"
            rc, out, _ = run_cli(["compare", str(A), str(B), "--classes", "PASS,FAIL", "--html", str(h),
                                  "--json", str(js), "--boot-draws", "200", "--date", "2027-01-02"])
            self.assertEqual(rc, 0)
            self.assertIn("j A vs B: 7/10 = 0.700", out)
            self.assertIn("gap: 1 day", out)
            self.assertIn("Judge drift report", h.read_text())
            self.assertEqual(json.loads(js.read_text())["judges"][0]["across"]["k"], 7)
            rc, _, err = run_cli(["compare", str(A), str(B), "--judge", "nobody"])
            self.assertEqual(rc, 2)
            self.assertIn("judgecal: error:", err)

    @unittest.skipUnless(STUDY.exists(), "needs the unpublished study fixture")
    def test_verdicts_out_carries_date_for_compare(self):
        with tempfile.TemporaryDirectory() as d:
            v1, v2 = Path(d) / "v1.jsonl", Path(d) / "v2.jsonl"
            base = ["report", "--items", str(STUDY / "items.jsonl"), "--classes", ",".join(GATE),
                    "--replay", f"frontier={STUDY / 'frontier.jsonl'}", "--boot-draws", "0"]
            run_cli(base + ["--date", "2026-10-01", "--verdicts-out", str(v1)])
            run_cli(base + ["--date", "2026-10-08", "--verdicts-out", str(v2)])
            self.assertEqual(json.loads(v1.read_text().splitlines()[0])["date"], "2026-10-01")
            d_ = compare(v1, v2, boot_draws=0)
            self.assertEqual(d_.gap, "7 days")
            self.assertEqual(d_.judges[0].across.k, 197)  # same recorded verdicts: no drift
            self.assertEqual(d_.judges[0].within_a.k, 178)


# ---------------------------------------------------------------- (8) HTML summary and print

@unittest.skipUnless(STUDY.exists(), "needs the unpublished study fixture")
class TestHtmlSummaryPrint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.page = to_html(study(), date="2026-10-10")

    def test_print_stylesheet(self):
        self.assertIn("@media print", self.page)
        self.assertIn("break-after:page", self.page)
        self.assertIn("@page", self.page)

    def test_summary_comes_first(self):
        p = self.page
        self.assertLess(p.index('id="h-summary"'), p.index('id="h-agree"'))
        summ = p[p.index('id="h-summary"'):p.index('id="h-agree"')]
        for s in ("±0.05", "0.904", "[0.752, 0.894]", "mid matches frontier on 180 of 197",
                  "inside the reference", "REASON took part in 13", "cost:", "judge noise ±0.05"):
            self.assertIn(s, summ)

    def test_accessibility_basics(self):
        p = self.page
        self.assertIn('<html lang="en">', p)
        self.assertIn('class="skip" href="#main"', p)
        self.assertIn("<desc id=\"chart-d\">", p)
        self.assertNotIn("<th></th>", p)
        self.assertIn("<rect class=\"sq", p)  # shape as well as colour separates the row kinds


# ---------------------------------------------------------------- remaining public paths

class FakeResponse:
    def __init__(self, body):
        self.status, self._b = 200, body

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestRemainingPaths(unittest.TestCase):
    def test_cli_endpoint_through_default_transport(self):
        """The real urllib transport, with its opener patched: no socket is opened."""
        import urllib.error
        from unittest import mock
        seen = []

        def fake_urlopen(req, timeout=None):
            seen.append(timeout)
            text = json.loads(req.data)["messages"][1]["content"]
            if text == "item 2":
                raise urllib.error.HTTPError(req.full_url, 401, "no", {}, io.BytesIO(b'{"error":"key"}'))
            return FakeResponse(json.dumps({"choices": [{"message": {"content": "CLASS: PASS\nCONFIDENCE: 90"}}],
                                            "usage": {"prompt_tokens": 50, "completion_tokens": 5}}).encode())
        with tempfile.TemporaryDirectory() as d, mock.patch("judgecal.judges._open", fake_urlopen):
            items = Path(d) / "i.jsonl"
            items.write_text("".join(json.dumps({"id": f"i{n}", "text": f"item {n}"}) + "\n" for n in range(4)))
            v = Path(d) / "v.jsonl"
            rc, out, err = run_cli(["report", "--items", str(items), "--classes", "PASS,FAIL",
                                    "--endpoint", "e", "http://judge.invalid/v1", "m", "-",
                                    "--workers", "2", "--item-timeout", "30", "--price", "e=1,2",
                                    "--verdicts-out", str(v), "--boot-draws", "0", "--date", "2026-10-10"])
            self.assertEqual(rc, 0, err)
            rows = [json.loads(x) for x in v.read_text().splitlines()]
            self.assertEqual(sum(r.get("failed", False) for r in rows), 2)  # item 2, both repeats
            self.assertEqual(rows[0]["input_tokens"], 50)
            self.assertEqual(rows[0]["date"], "2026-10-10")
            self.assertIn("e $0.0004 (300 in / 30 out tokens", out)
            self.assertIn("2 failed calls not counted", out)
            self.assertTrue(all(t is not None and t <= 30 for t in seen))

    def test_keyboard_interrupt_exit_code(self):
        from unittest import mock
        with mock.patch("judgecal.cli.Items.load", side_effect=KeyboardInterrupt):
            rc, _, err = run_cli(["report", "--items", "x", "--classes", "A,B", "--replay", "a=x"])
        self.assertEqual(rc, 130)
        self.assertIn("interrupted", err)

    def test_argparse_type_messages(self):
        for flag in ("--boot-draws", "--item-timeout", "--temperature"):
            with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
                cli_main(["report", "--items", "x", "--replay", "a=x", flag, "abc"])

    @unittest.skipUnless(STUDY.exists(), "needs the unpublished study fixture")
    def test_result_lookups_and_shares(self):
        res = study(boot_draws=0)
        with self.assertRaises(KeyError):
            res.judge("zz")
        with self.assertRaises(KeyError):
            res.pair("frontier", "zz")
        pooled = res.judge("frontier").low_confidence_pooled
        self.assertEqual((pooled.k, pooled.n), (9, 394))
        self.assertTrue(math.isnan(type(pooled)(0, 0).rate))
        self.assertEqual(ReplayJudge(STUDY / "frontier.jsonl", "f").repeats_available(), 2)

    def test_run_argument_errors(self):
        items = Items.load(EL / "items.jsonl")
        j = ReplayJudge(EL / "judge4.jsonl", "j")
        for kw in ({"repeats": 0}, {"pair_repeat": 2}):
            with self.assertRaises(ValueError):
                Calibration.run(items, [j], rubric=PF, **kw)
        with self.assertRaises(ValueError):
            Calibration.run(items, [], rubric=PF)
        with self.assertRaises(ValueError):
            Calibration.run(items, [j, ReplayJudge(EL / "judge4.jsonl", "j")], rubric=PF)
        with self.assertRaises(ValueError):
            Items([Item("a"), Item("a")])

    def test_no_calls_cost(self):
        items = Items([Item("zz")])
        res = Calibration.run(items, [ReplayJudge(EL / "judge4.jsonl", "j")], rubric=PF,
                              prices={"j": (1, 1)}, boot_draws=0)
        self.assertIn("j $0 (no calls returned)", res.cost_line())

    def test_reply_shapes(self):
        t = lambda u, h, b: (200, b'{"choices":[{"message":{"content":["x"]}}]}')  # noqa: E731
        j = ChatCompletionsJudge("http://judge.invalid/v1", "m", None, transport=t)
        with self.assertRaises(JudgeError):
            j.judge(Item("x", "y"), PF, 0)
        self.assertEqual(PF.parse("{not json} CLASS: FAIL"), ("FAIL", None))
        self.assertEqual(PF.parse('["a"] CLASS: PASS'), ("PASS", None))
        with self.assertRaises(ValueError):
            Rubric(["A", " "])

    def test_replay_and_rubric_file_errors(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.jsonl"
            p.write_text('{"class":"PASS"}\n')
            with self.assertRaisesRegex(ValueError, "no item id"):
                ReplayJudge(p, "j")
            r = Path(d) / "deep.json"
            r.write_text("[" * 200000 + "]" * 200000)
            try:
                Rubric.load(r)
            except ValueError as e:
                self.assertTrue("nested too deeply" in str(e) or "'classes' list" in str(e))

    def test_compare_named_judge_and_empty_drift(self):
        d = compare(A, B, judges=["j"], boot_draws=0)
        self.assertEqual([x.judge for x in d.judges], ["j"])
        from judgecal.compare import DriftJudge
        from judgecal.calibration import _agreement
        empty = DriftJudge("j", _agreement("j A", "j B", {}, {}, ["a"], ["P", "F"]))
        self.assertIn("no item with a readable verdict", drift_sentence(empty))
        self.assertIsNone(empty.band)
        with tempfile.TemporaryDirectory() as td:
            m = Path(td) / "m.jsonl"
            m.write_text('{"item":"e0","judge":"j","class":"PASS"}\n')
            self.assertIn("note: a session date is missing", drift_text(compare(A, m, boot_draws=0)))


if __name__ == "__main__":
    unittest.main()
