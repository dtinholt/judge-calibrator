"""Hostile and extreme input. Each test is a case the stress round broke or could have broken.
Fixtures are generated in temporary folders; no network."""
from __future__ import annotations

import io
import json
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from judgecal import Calibration, Items, ReplayJudge, Rubric, is_low  # noqa: E402
from judgecal.cli import main as cli_main  # noqa: E402
from judgecal.items import read_jsonl  # noqa: E402
from judgecal.judges import _repeat  # noqa: E402
from judgecal.report import to_html, to_text  # noqa: E402

STUDY = ROOT / "tests" / "fixtures" / "judge-study"
GATE = ["INFO", "REASON", "DONE", "BLOCKED"]
XSS = '<script>document.title="PWNED"</script><img src=x onerror="document.title=\'PWNED\'">'


def cli(args):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = cli_main(args)
    return rc, out.getvalue(), err.getvalue()


class Tmp(unittest.TestCase):
    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.d = Path(self._d.name)

    def tearDown(self):
        self._d.cleanup()

    def write(self, name, text, binary=False):
        p = self.d / name
        if binary:
            p.write_bytes(text)
        else:
            p.write_bytes(text.encode("utf-8"))  # bytes, so line endings stay as written
        return p


class TestReader(Tmp):
    def test_line_endings_bom_and_unicode_separators(self):
        body = '{"id":"a","text":"one two"}\r\n{"id":"b\u0085"}\r{"id":"c"}\n\n{"id":"d"}'
        p = self.write("x.jsonl", "﻿" + body)
        # str.splitlines would cut at  , \x85 and lone \r; only \n ends a line here
        rows = [d for _, d in read_jsonl(p)]
        self.assertEqual(rows[0]["text"], "one two")
        items = Items.load(self.write("y.jsonl", '{"id":"a","text":"one two"}\r\n{"id":"b\u0085"}\r\n'))
        self.assertEqual(items.ids(), ["a", "b\u0085"])

    def test_lone_cr_line_endings(self):
        # old Mac line endings: universal-newline reading turns a lone CR into a line end
        p = self.write("cr.jsonl", '{"id":"a"}\r{"id":"b"}\r')
        self.assertEqual(Items.load(p).ids(), ["a", "b"])

    def test_malformed_names_file_and_line(self):
        p = self.write("m.jsonl", '{"id":"a"}\n{"id": \n')
        with self.assertRaisesRegex(ValueError, r"m.jsonl:2: not valid JSON"):
            Items.load(p)

    def test_deep_nesting(self):
        p = self.write("deep.jsonl", '{"id":"a","x":' + "[" * 100000 + "]" * 100000 + "}\n")
        try:
            items = Items.load(p)  # newer Pythons decode it; then it is just an item
            self.assertEqual(items.ids(), ["a"])
        except ValueError as e:
            self.assertIn("nested too deeply", str(e))

    def test_non_object_rows_and_bad_ids(self):
        for body, msg in (("5\n", "JSON object"), ('["a"]\n', "JSON object"), ('"s"\n', "JSON object"),
                          ('{"id": null}\n', "string or number"), ('{"id": {"a":1}}\n', "string or number"),
                          ('{"id": "  "}\n', "empty"), ('{"id": true}\n', "string or number"),
                          ('{"text": "no id"}\n', "no 'id'")):
            with self.assertRaisesRegex(ValueError, msg):
                Items.load(self.write("b.jsonl", body))

    def test_empty_and_blank(self):
        with self.assertRaisesRegex(ValueError, "no items"):
            Items.load(self.write("e.jsonl", ""))
        with self.assertRaisesRegex(ValueError, "no items"):
            Items.load(self.write("e2.jsonl", "\n  \r\n\t\n"))
        with self.assertRaisesRegex(ValueError, "no items"):
            Calibration.run(Items([]), [], rubric=Rubric(GATE))

    def test_duplicate_ids_name_both_lines(self):
        with self.assertRaisesRegex(ValueError, r"d.jsonl:3: duplicate item id 'a' \(first on line 1\)"):
            Items.load(self.write("d.jsonl", '{"id":"a"}\n{"id":"b"}\n{"id":"a"}\n'))

    def test_numeric_ids_and_null_text(self):
        items = Items.load(self.write("n.jsonl", '{"id": 7, "text": null}\n{"id": 7.5}\n'))
        self.assertEqual(items.ids(), ["7", "7.5"])
        self.assertEqual(list(items)[0].text, "")

    def test_five_megabyte_item(self):
        p = self.write("big.jsonl", json.dumps({"id": "big", "text": "z" * 5_000_000}) + "\n")
        t = time.perf_counter()
        items = Items.load(p)
        self.assertEqual(len(list(items)[0].text), 5_000_000)
        self.assertLess(time.perf_counter() - t, 5)


class TestReplayInput(Tmp):
    def test_repeat_values(self):
        self.assertEqual(_repeat("2", "x"), 2)
        self.assertEqual(_repeat(3.0, "x"), 3)
        for bad in ("x", -1, 1.5, float("nan"), float("inf"), None, True, [1], "1e400"):
            with self.assertRaisesRegex(ValueError, "whole number"):
                _repeat(bad, "f:1")
        p = self.write("r.jsonl", '{"item":"a","repeat":"x","class":"P"}\n')
        with self.assertRaisesRegex(ValueError, r"r.jsonl:1: repeat"):
            ReplayJudge(p, "j")

    def test_nan_and_out_of_range_confidence(self):
        for c in ("nan", "NaN", "inf", "-inf", "-5", "150", "1e309"):
            self.assertFalse(is_low(c), c)
        self.assertTrue(is_low("0"))
        self.assertTrue(is_low("0.2"))
        self.assertFalse(is_low("100"))
        items = Items.load(self.write("i.jsonl", '{"id":"a"}\n{"id":"b"}\n'))
        p = self.write("j.jsonl", '{"item":"a","class":"P","confidence":NaN}\n'
                                  '{"item":"b","class":"Q","confidence":-Infinity}\n'
                                  '{"item":"a","repeat":1,"class":"P","confidence":1e999}\n'
                                  '{"item":"b","repeat":1,"class":"Q","confidence":"-5"}\n')
        res = Calibration.run(items, [ReplayJudge(p, "j")], rubric=Rubric(["P", "Q"]), boot_draws=50)
        self.assertEqual([s.k for s in res.judge("j").low_confidence], [0, 0])
        json.dumps(res.to_dict(), allow_nan=False)  # NaN never reaches the JSON output

    def test_unknown_classes_and_huge_open_class_set(self):
        items = Items([__import__("judgecal").Item(f"i{n}") for n in range(300)])
        p = self.write("j.jsonl", "".join(json.dumps({"item": f"i{n}", "class": f"c{n % 40}"}) + "\n"
                                          for n in range(300)))
        res = Calibration.run(items, [ReplayJudge(p, "j")], repeats=1, boot_draws=0)
        self.assertEqual(len(res.classes), 40)
        to_html(res)
        to_text(res)


class TestScale(Tmp):
    def scaled(self, m):
        items = [json.loads(x) for x in (STUDY / "items.jsonl").read_text().splitlines()]
        with open(self.d / "items.jsonl", "w") as f:
            for k in range(m):
                for it in items:
                    f.write(json.dumps({"id": f"{it['id']}-{k}"}) + "\n")
        for n in ("frontier", "mid", "small"):
            rows = [json.loads(x) for x in (STUDY / f"{n}.jsonl").read_text().splitlines()]
            with open(self.d / f"{n}.jsonl", "w") as f:
                for k in range(m):
                    for r in rows:
                        f.write(json.dumps(dict(r, item=f"{r['item']}-{k}")) + "\n")

    @unittest.skipUnless(STUDY.exists(), "needs the unpublished study fixture")
    def test_ten_times_sample(self):
        self.scaled(10)
        t = time.perf_counter()
        rc, out, _ = cli(["report", "--items", str(self.d / "items.jsonl"), "--classes", ",".join(GATE),
                          "--replay", f"frontier={self.d / 'frontier.jsonl'}",
                          "--replay", f"mid={self.d / 'mid.jsonl'}", "--replay", f"small={self.d / 'small.jsonl'}",
                          "--html", str(self.d / "r.html")])
        self.assertEqual(rc, 0)
        self.assertLess(time.perf_counter() - t, 20)
        self.assertIn("1780/1970 = 0.904", out)  # same rates at ten times the items, tighter interval
        self.assertIn("[0.890, 0.916]", out)


class TestHostileText(Tmp):
    def test_payloads_render_as_text(self):
        items = self.write("i.jsonl", "\n".join(json.dumps({"id": f"a{XSS}{n}", "text": XSS}) for n in range(3)))
        rows = []
        for n in range(3):
            for r in range(2):
                rows.append(json.dumps({"item": f"a{XSS}{n}", "repeat": r, "class": XSS if (n + r) % 2 else "Q",
                                        "confidence": XSS}))
        j = self.write("j.jsonl", "\n".join(rows))
        strata = self.write("s.jsonl", json.dumps({"id": f"a{XSS}0", "stratum": XSS}))
        h = self.d / "r.html"
        rc, out, _err = cli(["report", "--items", str(items), "--replay", f"x{XSS}={j}", "--html", str(h),
                          "--title", XSS, "--notes", XSS, "--source", XSS, "--date", XSS,
                          "--fixture-note", XSS, "--strata", str(strata), "--boot-draws", "50"])
        self.assertEqual(rc, 0, _err)
        page = h.read_text()
        self.assertNotIn("<script", page)
        self.assertNotIn("<img", page)
        self.assertIn("&lt;script&gt;", page)
        self.assertIn("onerror=&quot;", page)

    def test_terminal_control_characters_are_shown_not_sent(self):
        items = self.write("i.jsonl", '{"id":"a"}\n{"id":"b"}\n')
        j = self.write("j.jsonl", '{"item":"a","class":"P"}\n{"item":"b","class":"Q"}\n')
        rc, out, _ = cli(["report", "--items", str(items), "--replay", f"evil\x1b[2J\x07={j}",
                          "--classes", "P,Q", "--repeats", "1"])
        self.assertEqual(rc, 0)
        self.assertNotIn("\x1b", out)
        self.assertNotIn("\x07", out)
        self.assertIn("evil\\x1b[2J\\x07", out)


class TestCliErrors(Tmp):
    def test_one_line_errors_exit_two(self):
        ok_items = self.write("i.jsonl", '{"id":"a"}\n')
        empty = self.write("e.jsonl", "")
        cases = [
            (["--items", str(self.d / "nope.jsonl"), "--classes", "A,B", "--replay", f"a={ok_items}"], "No such file"),
            (["--items", str(empty), "--classes", "A,B", "--replay", f"a={ok_items}"], "no items"),
            (["--items", str(ok_items), "--classes", "A", "--replay", f"a={ok_items}"], "two classes"),
            (["--items", str(ok_items), "--classes", "A,B"], "at least one"),
            (["--items", str(ok_items), "--classes", "A,B", "--replay", "nopath"], "NAME=PATH"),
            (["--items", str(ok_items), "--classes", "A,B", "--replay", f"a={ok_items}", "--pair-repeat", "3"],
             "pair_repeat"),
            (["--items", str(ok_items), "--classes", "A,B", "--replay", f"a={ok_items}", "--price", "a=x,1"],
             "numbers"),
            (["--items", str(ok_items), "--classes", "A,B", "--replay", f"a={ok_items}", "--reference", "z"],
             "reference"),
            (["--items", str(ok_items), "--replay", f"a={ok_items}", "--endpoint", "e", "http://x.invalid", "m", "-"],
             "need --rubric"),
            (["--items", str(ok_items), "--rubric", str(self.write("r.json", "{}")), "--replay", f"a={ok_items}"],
             "'classes' list"),
            (["--items", str(ok_items), "--rubric", str(self.write("r2.json", "[1,")), "--replay", f"a={ok_items}"],
             "not valid JSON"),
            (["--items", str(ok_items), "--rubric", str(self.write("r3.json", '{"classes":["A","B"],'
                                                                   '"answer_format":"{x}"}')),
              "--replay", f"a={ok_items}"], "answer_format"),
            (["--items", str(ok_items), "--rubric", str(self.write("r4.json", '{"classes":"AB"}')),
              "--replay", f"a={ok_items}"], "list of names"),
            (["--items", str(ok_items), "--rubric", str(self.write("r5.json", '{"classes":["A","B"],'
                                                                   '"descriptions":[]}')),
              "--replay", f"a={ok_items}"], "descriptions"),
        ]
        for args, msg in cases:
            rc, _, err = cli(["report"] + args)
            self.assertEqual(rc, 2, args)
            self.assertIn("judgecal: error:", err)
            self.assertIn(msg, err, args)
            self.assertNotIn("Traceback", err)

    def test_argparse_rejects_bad_numbers(self):
        ok = self.write("i.jsonl", '{"id":"a"}\n')
        for flag, val in (("--repeats", "0"), ("--workers", "0"), ("--item-timeout", "nan"),
                          ("--temperature", "nan"), ("--temperature", "-1"), ("--boot-draws", "-5"),
                          ("--item-timeout", "-3"), ("--workers", "x")):
            with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
                cli_main(["report", "--items", str(ok), "--classes", "A,B", "--replay", f"a={ok}", flag, val])

    def test_json_output_is_strict(self):
        items = self.write("i.jsonl", '{"id":"a"}\n')
        j = self.write("j.jsonl", '{"item":"a","class":"A"}\n')
        out = self.d / "r.json"
        rc, _, _ = cli(["report", "--items", str(items), "--classes", "A,B", "--replay", f"a={j}",
                        "--json", str(out)])
        self.assertEqual(rc, 0)
        self.assertNotIn("NaN", out.read_text())
        json.loads(out.read_text())


if __name__ == "__main__":
    unittest.main()
