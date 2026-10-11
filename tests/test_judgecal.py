"""Tests for judgecal. Run from the build folder: python3 -m unittest discover -s tests -v"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from judgecal import (  # noqa: E402
    Calibration, ChatCompletionsJudge, Items, JudgeError, ReplayJudge, Rubric, is_low, kappa,
    noise_band, wilson,
)
from judgecal.cli import main as cli_main  # noqa: E402
from judgecal.items import Item  # noqa: E402
from judgecal.report import to_html, to_text  # noqa: E402

FX = ROOT / "tests" / "fixtures"
STUDY = FX / "judge-study"
SMALL = FX / "small"
GATE = ["INFO", "REASON", "DONE", "BLOCKED"]


class TestStats(unittest.TestCase):
    """The maths, checked against the values the study's analysis script itself asserts."""

    def test_wilson_matches_study_selftest(self):
        lo, hi = wilson(99, 120)
        self.assertAlmostEqual(lo, 0.747, delta=0.002)
        self.assertAlmostEqual(hi, 0.883, delta=0.002)

    def test_wilson_empty(self):
        lo, hi = wilson(0, 0)
        self.assertNotEqual(lo, lo)  # nan

    def test_kappa_chance_and_perfect(self):
        self.assertAlmostEqual(kappa([("A", "A"), ("B", "B"), ("A", "B"), ("B", "A")]), 0.0)
        self.assertEqual(kappa([("A", "A"), ("B", "B")]), 1.0)

    def test_kappa_single_class_is_undefined(self):
        k = kappa([("A", "A"), ("A", "A")])
        self.assertNotEqual(k, k)

    def test_noise_band_convention(self):
        self.assertAlmostEqual(noise_band(178 / 197), 19 / 197 / 2)
        self.assertEqual(f"{noise_band(178 / 197):.2f}", "0.05")


@unittest.skipUnless(STUDY.exists(), "needs the unpublished study fixture")
class TestStudyFixture(unittest.TestCase):
    """Fed the study's recorded verdicts, judgecal must reproduce the figures its own analysis computed."""

    @classmethod
    def setUpClass(cls):
        cls.items = Items.load(STUDY / "items.jsonl")
        cls.judges = [ReplayJudge(STUDY / f"{n}.jsonl", n) for n in ("frontier", "mid", "small")]
        cls.res = Calibration.run(cls.items, cls.judges, repeats=2, rubric=Rubric(GATE))

    def test_fixture_has_ids_only(self):
        for f in STUDY.glob("*.jsonl"):
            for line in f.read_text().splitlines():
                row = json.loads(line)
                self.assertLessEqual(set(row), {"id", "item", "repeat", "class", "confidence"}, f.name)

    def test_frontier_retest(self):
        r = self.res.judge("frontier").headline_retest
        self.assertEqual((r.k, r.n), (178, 197))
        self.assertEqual(f"{r.rate:.3f} [{r.lo:.3f}, {r.hi:.3f}]", "0.904 [0.854, 0.937]")
        self.assertEqual(f"{r.kappa:.3f}", "0.827")

    def test_tiers_against_frontier(self):
        s = self.res.pair("frontier", "mid")
        h = self.res.pair("frontier", "small")
        self.assertEqual((s.k, f"{s.rate:.3f}", f"{s.lo:.3f}", f"{s.hi:.3f}", f"{s.kappa:.3f}"),
                         (180, "0.914", "0.866", "0.945", "0.850"))
        self.assertEqual((h.k, f"{h.rate:.3f}", f"{h.lo:.3f}", f"{h.hi:.3f}", f"{h.kappa:.3f}"),
                         (161, "0.817", "0.757", "0.865", "0.689"))

    def test_flip_pattern(self):
        # analysis.md: nine REASON<->DONE flips, ten flips into BLOCKED (six from INFO, four from REASON)
        cm = self.res.judge("frontier").headline_retest.confusion
        i = {c: n for n, c in enumerate(GATE)}
        self.assertEqual(cm[i["DONE"]][i["REASON"]] + cm[i["REASON"]][i["DONE"]], 9)
        self.assertEqual(cm[i["BLOCKED"]][i["INFO"]], 6)
        self.assertEqual(cm[i["BLOCKED"]][i["REASON"]], 4)

    def test_low_confidence_and_blocked(self):
        # analysis-raw.txt: low-confidence 9 / 4 / 0, BLOCKED chosen 10 / 3 / 1
        lows = [self.res.judge(n).low_confidence[0].k for n in ("frontier", "mid", "small")]
        blocked = [self.res.judge(n).class_mix[0].get("BLOCKED", 0) for n in ("frontier", "mid", "small")]
        self.assertEqual(lows, [9, 4, 0])
        self.assertEqual(blocked, [10, 3, 1])

    def test_band_and_line(self):
        self.assertEqual(self.res.reference, "frontier")
        self.assertAlmostEqual(self.res.band, 0.0482, places=4)
        self.assertIn("±0.05", self.res.band_line())
        self.assertIn("19/197 = 0.096", self.res.band_line())

    def test_single_run_judges_have_no_retest(self):
        self.assertIsNone(self.res.judge("mid").band)
        self.assertEqual(self.res.judge("mid").missing, [0, 197])

    def test_paired_counts_printed(self):
        txt = to_text(self.res)
        self.assertIn("178/197 = 0.904 [0.854, 0.937], kappa 0.827; paired 197 / unpaired 0", txt)
        self.assertIn("paired n = 197, unpaired 0", self.res.band_line())
        page = to_html(self.res)
        self.assertIn("paired 197 / unpaired 0", page)
        self.assertIn("<td class=\"n\">197 / 0</td>", page)

    def test_fixture_label(self):
        self.assertNotIn("Arithmetic check", to_html(self.res))
        fx = [ReplayJudge(STUDY / f"{n}.jsonl", n, fixture=True) for n in ("frontier", "mid")]
        res = Calibration.run(self.items, fx, rubric=Rubric(GATE))
        self.assertEqual(res.fixture_judges, ["frontier", "mid"])
        self.assertIn("Arithmetic check on a recorded fixture; not a finding about any judge.", to_html(res))
        self.assertIn("ARITHMETIC CHECK", to_text(res))
        self.assertIn("custom label", to_html(self.res, fixture_note="custom label"))

    def test_html_report(self):
        page = to_html(self.res, date="2026-10-09")
        self.assertIn("0.904", page)
        self.assertIn("[0.854, 0.937]", page)
        self.assertIn("<svg", page)
        self.assertIn("prefers-color-scheme: dark", page)
        self.assertNotIn("—", page)  # no em dashes
        self.assertNotIn("<script", page)

    def test_demo_report_is_synthetic(self):
        """The shipped demo is built from synthetic data and carries none of the study's figures."""
        for name in ("report.html", "drift.html"):
            demo = (ROOT / "demo" / name).read_text()
            self.assertIn("Synthetic demo data from a fixed pattern; not a measurement of any judge.", demo)
            for fig in ("/197", "0.904", "0.854, 0.937", "0.827"):
                self.assertNotIn(fig, demo, (name, fig))
        report = (ROOT / "demo" / "report.html").read_text()
        self.assertIn("55/60", report)
        self.assertIn("Repeat sensitivity", report)
        self.assertIn("long cases", report)


class TestSmallFixture(unittest.TestCase):
    def setUp(self):
        self.items = Items.load(SMALL / "items.jsonl")
        self.rubric = Rubric.load(SMALL / "rubric.json")
        self.a = ReplayJudge(SMALL / "judge_a.jsonl", "a")
        self.b = ReplayJudge([SMALL / "judge_b_run1.jsonl", SMALL / "judge_b_run2.jsonl"], "b",
                             repeat_per_file=True)
        self.res = Calibration.run(self.items, [self.a, self.b], repeats=2, rubric=self.rubric)

    def test_items_loader(self):
        texts = [i.text for i in self.items]
        self.assertTrue(all(texts))
        self.assertEqual(list(self.items)[0].meta, {"split": "dev"})

    def test_unreadable_and_low(self):
        a = self.res.judge("a")
        self.assertEqual(a.unreadable, [0, 1])
        self.assertEqual([(s.k, s.n) for s in a.low_confidence], [(2, 4), (1, 3)])

    def test_retest_skips_unpaired(self):
        a = self.res.judge("a").headline_retest
        self.assertEqual((a.k, a.n, a.unpaired), (2, 3, 1))
        b = self.res.judge("b").headline_retest
        self.assertEqual((b.k, b.n, b.unpaired), (2, 3, 1))
        self.assertAlmostEqual(self.res.band, 1 / 6)

    def test_pairwise(self):
        p = self.res.pair("a", "b")
        self.assertEqual((p.k, p.n, p.unpaired), (3, 3, 1))
        self.assertEqual(p.kappa, 1.0)
        self.assertEqual(p.confusion, [[1, 0], [0, 2]])

    def test_reference_and_pair_repeat(self):
        res = Calibration.run(self.items, [self.a, self.b], repeats=2, rubric=self.rubric,
                              reference="b", pair_repeat=1)
        self.assertEqual(res.reference, "b")
        self.assertEqual(res.pair("a", "b").n, 3)  # a4 is MAYBE in a's run 2
        with self.assertRaises(ValueError):
            Calibration.run(self.items, [self.a], rubric=self.rubric, reference="zz")

    def test_missing_third_run_is_reported(self):
        res = Calibration.run(self.items, [self.a], repeats=3, rubric=self.rubric)
        self.assertEqual(len(res.judge("a").retest), 1)  # run 3 does not exist in the file
        self.assertEqual(res.judge("a").missing[2], 4)

    def test_no_rubric_infers_classes(self):
        res = Calibration.run(self.items, [self.b], repeats=2)
        self.assertEqual(sorted(res.classes), ["FAIL", "PASS"])

    def test_one_repeat_reports_no_band(self):
        res = Calibration.run(self.items, [self.a, self.b], repeats=1, rubric=self.rubric)
        self.assertIsNone(res.band)
        self.assertIn("not measured", res.band_line())
        self.assertIn("n/a", to_html(res))
        to_text(res)

    def test_duplicate_rows_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "dup.jsonl"
            p.write_text('{"item":"a1","class":"PASS"}\n{"item":"a1","class":"FAIL"}\n')
            with self.assertRaises(ValueError):
                ReplayJudge(p, "x")


class TestRubric(unittest.TestCase):
    def setUp(self):
        self.r = Rubric(GATE, instruction="Judge the next step.", descriptions={"info": "missing evidence"})

    def test_prompt(self):
        sp = self.r.system_prompt()
        self.assertIn("Judge the next step.", sp)
        self.assertIn("- INFO: missing evidence", sp)
        self.assertIn("CLASS: <INFO|REASON|DONE|BLOCKED>", sp)

    def test_parse_lines(self):
        self.assertEqual(self.r.parse("CLASS: done\nCONFIDENCE: 35\nAll closed."), ("DONE", "35"))
        self.assertEqual(self.r.parse("**CLASS:** REASON\n**CONFIDENCE:** medium-high"), ("REASON", "medium-high"))

    def test_parse_json(self):
        self.assertEqual(self.r.parse('{"class": "blocked", "confidence": "low"}'), ("BLOCKED", "low"))

    def test_parse_garbage(self):
        self.assertEqual(self.r.parse("I think it is fine."), (None, None))

    def test_bad_rubrics(self):
        with self.assertRaises(ValueError):
            Rubric(["ONLY"])
        with self.assertRaises(ValueError):
            Rubric(["A", "a"])

    def test_is_low(self):
        self.assertTrue(is_low("low"))
        self.assertTrue(is_low("49"))
        self.assertTrue(is_low("0.3"))
        self.assertFalse(is_low("medium-high"))
        self.assertTrue(is_low("1"))  # an integer 1 is 1 of 100
        self.assertFalse(is_low("1.0"))  # a decimal 1.0 is a probability, 100 of 100
        self.assertFalse(is_low(None))


class FakeTransport:
    """Records requests and replies from a script; no network."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, url, headers, body):
        self.calls.append((url, headers, json.loads(body)))
        status, content = self.replies.pop(0)
        if isinstance(content, BaseException):
            raise content
        if status == 200 and isinstance(content, bytes):
            return 200, content
        if status != 200:
            return status, b'{"error": "busy"}'
        return 200, json.dumps({"choices": [{"message": {"role": "assistant", "content": content}}]}).encode()


class TestChatCompletionsJudge(unittest.TestCase):
    def setUp(self):
        self.rubric = Rubric(["PASS", "FAIL"], instruction="Grade it.")
        self.item = Item("x1", "The evidence.")
        os.environ["JUDGECAL_TEST_KEY"] = "sk-test"

    def tearDown(self):
        os.environ.pop("JUDGECAL_TEST_KEY", None)

    def test_request_shape_and_parse(self):
        t = FakeTransport([(200, "CLASS: FAIL\nCONFIDENCE: 80\nMissing.")])
        j = ChatCompletionsJudge("http://judge.invalid/v1/", "m-1", "JUDGECAL_TEST_KEY", transport=t, temperature=0.0)
        v = j.judge(self.item, self.rubric, 0)
        self.assertEqual((v.cls, v.confidence, v.judge), ("FAIL", "80", "m-1"))
        url, headers, body = t.calls[0]
        self.assertEqual(url, "http://judge.invalid/v1/chat/completions")
        self.assertEqual(headers["Authorization"], "Bearer sk-test")
        self.assertEqual(body["model"], "m-1")
        self.assertEqual(body["temperature"], 0.0)
        self.assertEqual([m["role"] for m in body["messages"]], ["system", "user"])
        self.assertEqual(body["messages"][1]["content"], "The evidence.")

    def test_retry_then_success(self):
        slept = []
        t = FakeTransport([(429, ""), (503, ""), (200, "CLASS: PASS")])
        j = ChatCompletionsJudge("http://judge.invalid/v1", "m", "JUDGECAL_TEST_KEY", transport=t, sleep=slept.append)
        self.assertEqual(j.judge(self.item, self.rubric, 0).cls, "PASS")
        self.assertEqual(slept, [2.0, 4.0])

    def test_client_error_raises(self):
        t = FakeTransport([(401, "")])
        j = ChatCompletionsJudge("http://judge.invalid/v1", "m", "JUDGECAL_TEST_KEY", transport=t)
        with self.assertRaises(JudgeError):
            j.judge(self.item, self.rubric, 0)

    def test_missing_key(self):
        j = ChatCompletionsJudge("http://judge.invalid/v1", "m", "JUDGECAL_UNSET_KEY", transport=FakeTransport([]))
        with self.assertRaises(JudgeError):
            j.judge(self.item, self.rubric, 0)

    def test_no_key_for_local_server(self):
        t = FakeTransport([(200, "CLASS: PASS")])
        j = ChatCompletionsJudge("http://localhost:8000/v1", "m", None, transport=t)
        j.judge(self.item, self.rubric, 0)
        self.assertNotIn("Authorization", t.calls[0][1])

    def test_calibration_end_to_end(self):
        replies = ["CLASS: PASS", "CLASS: FAIL", "CLASS: PASS", "CLASS: PASS",  # run 1
                   "CLASS: PASS", "CLASS: FAIL", "CLASS: FAIL", "garbled"]       # run 2
        t = FakeTransport([(200, r) for r in replies])
        j = ChatCompletionsJudge("http://judge.invalid/v1", "m", "JUDGECAL_TEST_KEY", transport=t)
        items = Items([Item(f"i{n}", f"text {n}") for n in range(4)])
        res = Calibration.run(items, [j], repeats=2, rubric=self.rubric, workers=1)
        r = res.judge("m").headline_retest
        self.assertEqual((r.k, r.n, r.unpaired), (2, 3, 1))
        self.assertEqual(res.judge("m").unreadable, [0, 1])

    def test_default_key_env(self):
        self.assertEqual(ChatCompletionsJudge("http://judge.invalid/v1", "m").api_key_env, "JUDGE_API_KEY")

    def test_transport_failures_are_retried(self):
        slept = []
        t = FakeTransport([(0, urllib.error.URLError("connection refused")), (0, TimeoutError("timed out")),
                           (0, ConnectionResetError("reset")), (200, "CLASS: FAIL")])
        j = ChatCompletionsJudge("http://judge.invalid/v1", "m", "JUDGECAL_TEST_KEY", transport=t, sleep=slept.append)
        self.assertEqual(j.judge(self.item, self.rubric, 0).cls, "FAIL")
        self.assertEqual(slept, [2.0, 4.0, 8.0])

    def test_transport_failure_exhausts_then_records(self):
        failing = [(0, urllib.error.URLError("down"))] * 4
        t = FakeTransport(failing + [(200, "CLASS: PASS")])
        j = ChatCompletionsJudge("http://judge.invalid/v1", "m", "JUDGECAL_TEST_KEY", transport=t, sleep=lambda _: None)
        with self.assertRaises(JudgeError):
            j.judge(self.item, self.rubric, 0)
        t2 = FakeTransport(failing + [(200, "CLASS: PASS")])
        j2 = ChatCompletionsJudge("http://judge.invalid/v1", "m", "JUDGECAL_TEST_KEY", transport=t2, sleep=lambda _: None)
        items = Items([Item("i0", "a"), Item("i1", "b")])
        res = Calibration.run(items, [j2], repeats=1, rubric=self.rubric, workers=1)
        self.assertEqual(res.judge("m").unreadable, [1])
        self.assertEqual(res.judge("m").verdicts, [1])
        self.assertTrue(res.verdicts[0].raw.startswith("error: "))

    def test_malformed_body_gets_one_retry(self):
        t = FakeTransport([(200, b"<html>gateway</html>"), (200, "CLASS: PASS")])
        j = ChatCompletionsJudge("http://judge.invalid/v1", "m", "JUDGECAL_TEST_KEY", transport=t)
        self.assertEqual(j.judge(self.item, self.rubric, 0).cls, "PASS")
        t2 = FakeTransport([(200, b"{}"), (200, b'{"choices": []}'), (200, "CLASS: PASS")])
        j2 = ChatCompletionsJudge("http://judge.invalid/v1", "m", "JUDGECAL_TEST_KEY", transport=t2)
        with self.assertRaises(JudgeError):
            j2.judge(self.item, self.rubric, 0)
        self.assertEqual(len(t2.calls), 2)

    def test_errors_recorded_by_default(self):
        t = FakeTransport([(401, ""), (200, "CLASS: PASS")])
        j = ChatCompletionsJudge("http://judge.invalid/v1", "m", "JUDGECAL_TEST_KEY", transport=t)
        items = Items([Item("i0", "a"), Item("i1", "b")])
        res = Calibration.run(items, [j], repeats=1, rubric=self.rubric, workers=1)
        self.assertEqual(res.judge("m").unreadable, [1])
        t2 = FakeTransport([(401, "")])
        j2 = ChatCompletionsJudge("http://judge.invalid/v1", "m", "JUDGECAL_TEST_KEY", transport=t2)
        with self.assertRaises(JudgeError):
            Calibration.run(items, [j2], repeats=1, rubric=self.rubric, on_error="raise", workers=1)


class TestCLI(unittest.TestCase):
    @unittest.skipUnless(STUDY.exists(), "needs the unpublished study fixture")
    def test_report_html_json_and_replayable_verdicts(self):
        with tempfile.TemporaryDirectory() as d:
            html_p, json_p, v_p = Path(d) / "r.html", Path(d) / "r.json", Path(d) / "v.jsonl"
            out = io.StringIO()
            with redirect_stdout(out), redirect_stderr(io.StringIO()):
                rc = cli_main(["report", "--items", str(STUDY / "items.jsonl"), "--classes", ",".join(GATE),
                               "--replay", f"frontier={STUDY / 'frontier.jsonl'}",
                               "--replay", f"mid={STUDY / 'mid.jsonl'}",
                               "--html", str(html_p), "--json", str(json_p), "--verdicts-out", str(v_p)])
            self.assertEqual(rc, 0)
            self.assertIn("178/197 = 0.904 [0.854, 0.937]", out.getvalue())
            self.assertIn("<svg", html_p.read_text())
            d_ = json.loads(json_p.read_text())
            self.assertAlmostEqual(d_["band"], 19 / 197 / 2)
            # the verdict dump replays to the same retest
            rj = ReplayJudge(v_p, "frontier", judge_field="frontier")
            res = Calibration.run(Items.load(STUDY / "items.jsonl"), [rj], rubric=Rubric(GATE))
            self.assertEqual(res.judge("frontier").headline_retest.k, 178)


if __name__ == "__main__":
    unittest.main()
