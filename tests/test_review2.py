"""Tests for the fixes from the second independent check: redirects, incremental verdict
saving, the alias removal, synthetic demos and packaging, the .md id rule, the narrow-screen
report, progress notes, small-n warnings and copy. Loopback sockets only (two local fake
servers for the redirect test); no outside network."""
from __future__ import annotations

import http.server
import io
import json
import os
import re
import sys
import tempfile
import threading
import unittest
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import judgecal  # noqa: E402
from judgecal import Calibration, ChatCompletionsJudge, Items, JudgeError, ReplayJudge, Rubric, compare, load_strata  # noqa: E402,E501
from judgecal.cli import main as cli_main, make_progress  # noqa: E402
from judgecal.compare import load_session  # noqa: E402
from judgecal.items import Item  # noqa: E402
from judgecal.report import CSS, drift_text, to_html, to_text  # noqa: E402

FX = ROOT / "tests" / "fixtures"
EL = FX / "elevate"
STUDY = FX / "judge-study"
PF = Rubric(["PASS", "FAIL"])


def run_cli(args):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = cli_main(args)
    return rc, out.getvalue(), err.getvalue()


# ---------------------------------------------------------------- 1. redirects

class _Server:
    """A loopback HTTP server that records requests and answers from a function."""

    def __init__(self, answer):
        rec = self.requests = []

        class H(http.server.BaseHTTPRequestHandler):
            def _any(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else b""
                rec.append((self.command, self.path, dict(self.headers), body))
                code, headers, payload = answer(self)
                self.send_response(code)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            do_GET = do_POST = _any

            def log_message(self, *a):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.t = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.t.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.t.join()


OK_BODY = json.dumps({"choices": [{"message": {"content": "CLASS: PASS"}}]}).encode()


class TestRedirects(unittest.TestCase):
    def setUp(self):
        os.environ["JUDGECAL_TEST_KEY"] = "test-key-not-real"
        self.b = _Server(lambda h: (200, {"Content-Type": "application/json"}, OK_BODY))

    def tearDown(self):
        self.b.close()
        os.environ.pop("JUDGECAL_TEST_KEY", None)

    def test_two_servers_no_follow(self):
        for code in (301, 302, 303, 307, 308):
            a = _Server(lambda h, c=code: (c, {"Location": self.b.url + "/v1/chat/completions"}, b""))
            try:
                j = ChatCompletionsJudge(a.url + "/v1", "m", "JUDGECAL_TEST_KEY", item_timeout=10)
                with self.assertRaisesRegex(JudgeError, f"HTTP {code} redirect refused"):
                    j.judge(Item("x", "y"), PF, 0)
                self.assertEqual(len(a.requests), 1)  # no retry either
                self.assertEqual(a.requests[0][0], "POST")
            finally:
                a.close()
        self.assertEqual(self.b.requests, [])  # the second origin never saw the key

    def test_control_plain_urlopen_would_leak(self):
        """Why the fix matters: urllib's default opener follows the 302 as a GET, key attached."""
        a = _Server(lambda h: (302, {"Location": self.b.url + "/elsewhere"}, b""))
        try:
            req = urllib.request.Request(a.url + "/v1/chat/completions", data=b"{}", method="POST",
                                         headers={"Authorization": "Bearer test-key-not-real"})
            with urllib.request.urlopen(req, timeout=10) as r:
                r.read()
            method, _, headers, _ = self.b.requests[0]
            self.assertEqual(method, "GET")
            self.assertEqual(headers.get("Authorization"), "Bearer test-key-not-real")
        finally:
            a.close()

    def test_fake_transport_302(self):
        calls = []

        def t(url, headers, body):
            calls.append(url)
            return 302, b""
        j = ChatCompletionsJudge("http://judge.invalid/v1", "m", None, transport=t)
        with self.assertRaisesRegex(JudgeError, "redirect refused"):
            j.judge(Item("x", "y"), PF, 0)
        self.assertEqual(len(calls), 1)

    def test_direct_success_still_works(self):
        j = ChatCompletionsJudge(self.b.url + "/v1", "m", "JUDGECAL_TEST_KEY", item_timeout=10)
        self.assertEqual(j.judge(Item("x", "y"), PF, 0).cls, "PASS")
        self.assertEqual(self.b.requests[0][2].get("Authorization"), "Bearer test-key-not-real")


# ---------------------------------------------------------------- 2. verdicts saved as they land

class TestIncrementalVerdicts(unittest.TestCase):
    def args(self, d, *extra):
        return ["report", "--items", str(EL / "items.jsonl"), "--classes", "PASS,FAIL",
                "--replay", f"j={EL / 'judge4.jsonl'}", "--repeats", "4", "--boot-draws", "0",
                "--verdicts-out", str(Path(d) / "v.jsonl"), "--date", "2026-10-10", *extra]

    def test_failed_html_keeps_every_verdict(self):
        with tempfile.TemporaryDirectory() as d:
            rc, _, err = run_cli(self.args(d, "--html", str(Path(d) / "no-such-dir" / "r.html")))
            self.assertEqual(rc, 2)
            self.assertIn("judgecal: error:", err)
            rows = [json.loads(x) for x in (Path(d) / "v.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 40)
            self.assertEqual(rows[0]["date"], "2026-10-10")

    def test_ctrl_c_keeps_calls_already_made(self):
        real = ReplayJudge.judge
        count = [0]

        def interrupting(self, item, rubric, repeat):
            count[0] += 1
            if count[0] == 5:
                raise KeyboardInterrupt
            return real(self, item, rubric, repeat)
        with tempfile.TemporaryDirectory() as d, mock.patch.object(ReplayJudge, "judge", interrupting):
            rc, _, err = run_cli(self.args(d))
            self.assertEqual(rc, 130)
            lines = (Path(d) / "v.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 4)
            self.assertEqual([json.loads(x)["item"] for x in lines], ["e0", "e1", "e2", "e3"])

    def test_verdicts_written_before_reports(self):
        order = []
        import judgecal.cli as cli
        real_write = cli.write_verdict
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(cli, "write_verdict", lambda f, v, date: (order.append("v"), real_write(f, v, date))), \
                mock.patch.object(cli, "to_html", lambda *a, **k: (order.append("html"), "<html></html>")[1]):
            rc, _, _ = run_cli(self.args(d, "--html", str(Path(d) / "r.html")))
        self.assertEqual(rc, 0)
        self.assertEqual(order[-1], "html")
        self.assertEqual(order.count("v"), 40)

    def test_concurrent_rows_are_whole(self):
        def t(url, headers, body):
            return 200, json.dumps({"choices": [{"message": {"content": "CLASS: PASS"}}],
                                    "usage": {"prompt_tokens": 3, "completion_tokens": 1}}).encode()
        got = []
        items = Items([Item(f"i{n}", f"item {n}") for n in range(50)])
        j = ChatCompletionsJudge("http://judge.invalid/v1", "m", None, transport=t)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "v.jsonl"
            from judgecal.cli import write_verdict
            with open(p, "w") as f:
                Calibration.run(items, [j], repeats=2, rubric=PF, workers=8, boot_draws=0,
                                on_verdict=lambda v: (got.append(v.item), write_verdict(f, v, "d")))
            rows = [json.loads(x) for x in p.read_text().splitlines()]
        self.assertEqual(len(rows), 100)
        self.assertEqual(sorted((r["item"], r["repeat"]) for r in rows),
                         sorted((f"i{n}", r) for n in range(50) for r in range(2)))


# ---------------------------------------------------------------- 3. alias removed

class TestAliasGone(unittest.TestCase):
    def test_no_alias(self):
        self.assertFalse(hasattr(judgecal, "OpenAICompatJudge"))
        self.assertNotIn("OpenAICompatJudge", judgecal.__all__)
        import judgecal.judges as jj
        self.assertFalse(hasattr(jj, "OpenAICompatJudge"))
        self.assertNotIn("OpenAICompatJudge", (ROOT / "README.md").read_text())


# ---------------------------------------------------------------- 4. synthetic demos and packaging

class TestShippedSurfaces(unittest.TestCase):
    def test_study_figures_absent(self):
        for f in ("README.md", "judgecal/stats.py", "judgecal/report.py",
                  "judgecal/calibration.py", "demo/report.html", "demo/drift.html"):
            text = ROOT.joinpath(f).read_text()
            for fig in ("/197", "178", "0.904", "0.827", "19 of 197"):
                self.assertNotIn(fig, text, (f, fig))

    def test_study_fixture_excluded_from_packages(self):
        py = (ROOT / "pyproject.toml").read_text()
        self.assertIn('packages = ["judgecal"]', py)
        self.assertIn("include-package-data = false", py)

    def test_synthetic_data_regenerates_identically(self):
        before = {p.name: p.read_bytes() for p in (ROOT / "demo" / "data").glob("*.json*")}
        with tempfile.TemporaryDirectory() as d:
            import shutil
            dst = Path(d) / "data"
            shutil.copytree(ROOT / "demo" / "data", dst)
            for p in dst.glob("*.json*"):
                p.unlink()
            import runpy
            runpy.run_path(str(dst / "make_synthetic.py"))
            after = {p.name: p.read_bytes() for p in dst.glob("*.json*")}
        self.assertEqual(before, after)


# ---------------------------------------------------------------- 5. one .md rule for every id

class TestMdIds(unittest.TestCase):
    def test_same_rule_everywhere(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "items.jsonl").write_text('{"id":"a1.md"}\n{"id":"a2"}\n')
            (d / "v.jsonl").write_text('{"item":"a1","repeat":0,"class":"PASS"}\n'
                                       '{"packet":"a2.md","repeat":0,"class":"FAIL"}\n'
                                       '{"item":"a1.md","repeat":1,"class":"PASS"}\n'
                                       '{"item":"a2","repeat":1,"class":"PASS"}\n')
            (d / "s.jsonl").write_text('{"id":"a1.md","stratum":"x"}\n{"id":"a2","stratum":"y"}\n')
            items = Items.load(d / "items.jsonl")
            self.assertEqual(items.ids(), ["a1", "a2"])
            res = Calibration.run(items, [ReplayJudge(d / "v.jsonl", "j")], rubric=PF, boot_draws=0,
                                  strata=load_strata(d / "s.jsonl"))
            r = res.judge("j").headline_retest
            self.assertEqual((r.k, r.n, r.unpaired), (1, 2, 0))
            self.assertEqual(res.strata_ignored, 0)
            self.assertEqual(sorted(load_session(d / "v.jsonl").order), ["a1", "a2"])
            (d / "dup.jsonl").write_text('{"id":"a"}\n{"id":"a.md"}\n')
            with self.assertRaisesRegex(ValueError, "duplicate item id 'a'"):
                Items.load(d / "dup.jsonl")
            (d / "e.jsonl").write_text('{"id":".md"}\n')
            with self.assertRaisesRegex(ValueError, "empty"):
                Items.load(d / "e.jsonl")


# ---------------------------------------------------------------- 6. narrow screens

@unittest.skipUnless(STUDY.exists(), "needs the unpublished study fixture")
class TestNarrowScreen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        items = Items.load(STUDY / "items.jsonl")
        js = [ReplayJudge(STUDY / f"{n}.jsonl", n) for n in ("frontier", "mid")]
        cls.page = to_html(Calibration.run(items, js, rubric=Rubric(["INFO", "REASON", "DONE", "BLOCKED"]),
                                           boot_draws=0))

    def test_wrappers_focusable_and_named(self):
        wraps = re.findall(r'<div class="tablewrap[^"]*"[^>]*>', self.page)
        self.assertTrue(wraps)
        for w in wraps:
            self.assertIn('tabindex="0"', w)
            self.assertIn('role="region"', w)
            self.assertRegex(w, r'aria-label="[^"]+"')
        self.assertIn('class="chartwrap" tabindex="0" role="region"', self.page)
        self.assertIn('content:"scrolls sideways"', CSS)

    def test_chart_scrolls_instead_of_shrinking(self):
        self.assertIn("min-width:560px", CSS)
        for size in re.findall(r"font-size:(\d+(?:\.\d+)?)px", CSS):
            self.assertGreaterEqual(float(size), 10, size)
        self.assertNotIn("t-val{display:none}", CSS)

    def test_labels_clear_of_gridlines(self):
        svg = self.page[self.page.index("<svg"):self.page.index("</svg>")]
        labels = [float(y) for y in re.findall(r'<text class="t-lab" x="[\d.]+" y="([\d.]+)"', svg)]
        bands = [(float(a), float(b)) for a, b in
                 re.findall(r'<line class="(?:grid|ref)" x1="[\d.]+" x2="[\d.]+" y1="([\d.]+)" y2="([\d.]+)"', svg)]
        self.assertTrue(labels and bands)
        for y in labels:
            for a, b in bands:
                # a label covers about 13 px above its baseline and 3 px of descender below it
                self.assertFalse(y + 3 > a and y - 13 < b, (y, a, b))


# ---------------------------------------------------------------- 7. progress

class TestProgress(unittest.TestCase):
    def test_quiet_then_one_line_per_repeat(self):
        now = [0.0]
        buf = io.StringIO()
        note = make_progress(buf, clock=lambda: now[0])
        note("j", 0, 3, 100)
        self.assertEqual(buf.getvalue(), "")
        now[0] = 2.5
        note("j", 1, 3, 100)
        self.assertEqual(buf.getvalue(), "judgecal: j run 2 of 3 done (100 items, 2.5 s elapsed)\n")

    def test_called_once_per_repeat(self):
        calls = []
        items = Items.load(EL / "items.jsonl")
        Calibration.run(items, [ReplayJudge(EL / "judge4.jsonl", "j"), ReplayJudge(EL / "judge4.jsonl", "k")],
                        repeats=3, rubric=PF, boot_draws=0, progress=lambda *a: calls.append(a))
        self.assertEqual(calls, [("j", 0, 3, 10), ("j", 1, 3, 10), ("j", 2, 3, 10),
                                 ("k", 0, 3, 10), ("k", 1, 3, 10), ("k", 2, 3, 10)])

    def test_changelog_sentence_fixed(self):
        log = (ROOT / "CHANGELOG.md").read_text()
        self.assertNotIn("so no progress note or chunking was added", log)


# ---------------------------------------------------------------- 8. small-n warnings

class TestSmallN(unittest.TestCase):
    def test_warnings_everywhere(self):
        items = Items.load(EL / "items.jsonl")
        res = Calibration.run(items, [ReplayJudge(EL / "judge4.jsonl", "j")], rubric=PF, boot_draws=0,
                              strata=load_strata(EL / "strata.jsonl"))
        ws = res.small_n_warnings()
        self.assertTrue(any(w.startswith("j#1 vs j#2 rests on 10 paired items (under 20)") for w in ws))
        self.assertTrue(any(w.startswith("stratum short:") for w in ws))
        self.assertIn("warning: j#1 vs j#2 rests on 10", to_text(res))
        self.assertIn("Small samples.", to_html(res))
        d = compare(EL / "verdicts-2026-12-31.jsonl", EL / "verdicts-2027-01-01.jsonl", boot_draws=0)
        self.assertEqual(len(d.small_n_warnings()), 2)
        self.assertIn("warning:", drift_text(d))
        rc, _, err = run_cli(["report", "--items", str(EL / "items.jsonl"), "--classes", "PASS,FAIL",
                              "--replay", f"j={EL / 'judge4.jsonl'}", "--boot-draws", "0"])
        self.assertIn("judgecal: warning: j#1 vs j#2 rests on 10 paired items", err)

    @unittest.skipUnless(STUDY.exists(), "needs the unpublished study fixture")
    def test_none_at_twenty_or_more(self):
        items = Items.load(STUDY / "items.jsonl")
        res = Calibration.run(items, [ReplayJudge(STUDY / "frontier.jsonl", "f")],
                              rubric=Rubric(["INFO", "REASON", "DONE", "BLOCKED"]), boot_draws=0)
        self.assertEqual(res.small_n_warnings(), [])
        self.assertNotIn("Small samples", to_html(res))


# ---------------------------------------------------------------- 9. copy

class TestCopy(unittest.TestCase):
    def test_accuracy_statement_once_in_readme_only(self):
        readme = (ROOT / "README.md").read_text()
        self.assertEqual(readme.count("ground truth"), 1)
        page = to_html(Calibration.run(Items.load(EL / "items.jsonl"), [ReplayJudge(EL / "judge4.jsonl", "j")],
                                       rubric=PF, boot_draws=0))
        for text in (page, judgecal.__doc__):
            self.assertNotIn("ground truth", text)
            self.assertNotIn("never judge accuracy", text)

    def test_phrases_gone(self):
        all_text = "".join(ROOT.joinpath(f).read_text() for f in ("README.md", "CHANGELOG.md", "judgecal/report.py",
                                                             "judgecal/compare.py"))
        for bad in ("far too faint", "plain stochasticity", "The bracket comes from the number of items. The band comes",
                    "Grey squares compare two judges", "a serving change and sampling noise",
                    "task family, source, a difficulty split", "—"):
            self.assertNotIn(bad, all_text, bad)


if __name__ == "__main__":
    unittest.main()
