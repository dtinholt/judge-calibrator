"""Text and HTML renderings of a CalibrationResult and of a DriftResult."""
from __future__ import annotations

import datetime as _dt
import html
import math
import re
from typing import Optional

from .calibration import Agreement, CalibrationResult, JudgeSummary
from .compare import DriftResult, drift_sentence

FIXTURE_NOTE = "Arithmetic check on a recorded fixture; not a finding about any judge."

# ---------------------------------------------------------------- shared wording


def _f(x: Optional[float]) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f  ]")


def _t(s: object) -> str:
    """User text for the terminal: control characters (ANSI escapes included) shown as \\xNN."""
    return _CTRL.sub(lambda m: f"\\x{ord(m.group()):02x}" if ord(m.group()) < 256 else f"\\u{ord(m.group()):04x}",
                     str(s))


def _pn(a: Agreement) -> str:
    return f"paired {a.n} / unpaired {a.unpaired}"


def _kci(a: Agreement) -> str:
    """The bootstrap interval for kappa as text, or '' when it was not computed."""
    if not a.boot_draws or a.kappa_lo != a.kappa_lo:
        return ""
    s = f"[{a.kappa_lo:.3f}, {a.kappa_hi:.3f}]"
    if a.kappa_undefined:
        s += f" ({a.kappa_undefined} of {a.boot_draws} draws undefined)"
    return s


def _agree_line(a: Agreement) -> str:
    k = _kci(a)
    return (f"{a.k}/{a.n} = {_f(a.rate)} [{_f(a.lo)}, {_f(a.hi)}], kappa {_f(a.kappa)}; {_pn(a)}"
            + (f"; kappa bootstrap 95% {k}" if k else ""))


def _fixture_note(res: CalibrationResult, fixture_note: Optional[str]) -> Optional[str]:
    if fixture_note:
        return fixture_note
    return FIXTURE_NOTE if res.fixture_judges else None


def _runs(a: Agreement) -> tuple[str, str]:
    return (str(a.run_a or 1), str(a.run_b or 2))


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def wavers_paragraph(name: str, a: Agreement) -> str:
    """'Where the judge wavers': a paragraph built from one test-retest table."""
    ra, rb = _runs(a)
    if a.n == 0:
        return f"{name} has no item with a readable verdict on both runs, so no class can be singled out."
    flips = a.n - a.k
    if flips == 0:
        return (f"{name} put every one of its {a.n} paired items in the same class on run {ra} "
                f"and run {rb}. No class wavered.")
    out = [f"{name} changed its class on {flips} of {a.n} paired items between run {ra} and run {rb}."]
    top, cnt = a.most_flipped()
    if len(top) == 1:
        out.append(f"{top[0]} took part in {cnt} of those flips, more than any other class.")
    else:
        out.append(f"{', '.join(top[:-1])} and {top[-1]} tie for the most flips at {cnt} each.")
    pair = a.top_flip_pair()
    if pair:
        x, y, xy, yx = pair
        t = xy + yx
        if xy and yx:
            out.append(f"The commonest move ran between {x} and {y}: {xy} went from {x} on run {ra} to "
                       f"{y} on run {rb}, {yx} the other way.")
        else:
            src, dst = (x, y) if xy else (y, x)
            out.append(f"The commonest move was {src} on run {ra} becoming {dst} on run {rb}, "
                       f"{_plural(t, 'item')} and none in reverse.")
    named = [c for c in a.per_class if c.specific is not None]
    if len(named) >= 2:
        low = min(named, key=lambda c: c.specific)
        out.append(f"Specific agreement, the share of verdicts naming a class that the other run "
                   f"matched, bottoms out at {low.specific:.3f} for {low.cls} "
                   f"({_plural(low.named_a + low.named_b, 'verdict')} named it).")
    return " ".join(out)


def sensitivity_note(j: JudgeSummary) -> Optional[str]:
    """Whether the band moves as runs are added; None with fewer than three runs measured."""
    s = j.sensitivity
    if len(s) < 2:
        return None
    first, last = s[0], s[-1]
    a, b = f"{first.band:.2f}", f"{last.band:.2f}"
    if a == b:
        out = (f"Rounded to two places, {j.name}'s band holds at ±{a} from {first.runs} runs "
               f"to {last.runs}.")
    else:
        out = (f"Rounded to two places, {j.name}'s band moves from ±{a} with {first.runs} runs "
               f"to ±{b} with {last.runs}.")
    single = [(1 - r.rate) / 2 for r in j.retest if r.n]
    if len(single) > 1:
        out += (f" Taken one pair at a time, the {len(single)} pairs give bands from "
                f"±{min(single):.3f} to ±{max(single):.3f}.")
    return out


# ---------------------------------------------------------------- text


def _confusion_text(a: Agreement, classes: list[str]) -> str:
    cl = [_t(c) for c in classes]
    w = max(7, *(len(c) for c in cl))
    head = f"{'':<{w}} | " + " | ".join(f"{c:>{w}}" for c in cl)
    rows = [f"{_t(a.a)} rows, {_t(a.b)} columns; {_pn(a)}", head]
    for c, row in zip(cl, a.confusion):
        rows.append(f"{c:<{w}} | " + " | ".join(f"{v:>{w}}" for v in row))
    return "\n".join(rows)


def to_text(res: CalibrationResult, fixture_note: Optional[str] = None) -> str:
    note = _fixture_note(res, fixture_note)
    out = [_t(note).upper(), ""] if note else []
    out += [f"judgecal: {res.n_items} items, {len(res.judges)} judges, {res.repeats} repeats",
            f"classes: {_t(', '.join(res.classes))}", ""]
    for w in res.small_n_warnings():
        out.append(f"warning: {_t(w)}")
    if res.small_n_warnings():
        out.append("")
    if res.boot_draws:
        out += [f"kappa intervals: percentile bootstrap, {res.boot_draws:,} draws, seed {_t(res.seed)}", ""]
    out.append("TEST-RETEST")
    for j in res.judges:
        if not j.retest:
            out.append(f"  {_t(j.name)}: not measured (fewer than two repeats with verdicts)")
            continue
        for a in j.retest:
            out.append(f"  {_t(a.a)} vs {_t(a.b)}: {_agree_line(a)}")
        out.append(f"  {_t(j.name)} noise band: +/-{j.band:.3f}")
    out.append("")
    out.append(f"BETWEEN JUDGES (repeat {res.pair_repeat + 1})")
    for a in res.pairwise:
        out.append(f"  {_t(a.a)} vs {_t(a.b)}: {_agree_line(a)}")
    out.append("")
    sens = [(j, sensitivity_note(j)) for j in res.judges]
    sens = [(j, n) for j, n in sens if n]
    if sens:
        out.append("REPEAT SENSITIVITY")
        for j, n in sens:
            for r in j.sensitivity:
                out.append(f"  {_t(j.name)} runs 1-{r.runs}: {_plural(r.pairs, 'pair')}, "
                           f"{r.disagreements}/{r.paired} flips, band +/-{r.band:.3f}")
            out.append(f"  {_t(n)}")
        out.append("")
    out.append("WHERE THE JUDGE WAVERS")
    for j in res.judges:
        if j.headline_retest:
            h = j.headline_retest
            out.append(f"  {_t(wavers_paragraph(j.name, h))}")
            out.append("  per class (run {} vs run {}): ".format(*_runs(h)) + "; ".join(
                f"{_t(c.cls)} specific {_f(c.specific)}, flips {c.flips}" for c in h.per_class))
    out.append("")
    out.append("LOW CONFIDENCE")
    for j in res.judges:
        parts = [f"#{i + 1} {s.k}/{s.n}" for i, s in enumerate(j.low_confidence) if s.n]
        out.append(f"  {_t(j.name)}: " + ", ".join(parts)
                   + (f"; unreadable {sum(j.unreadable)}" if sum(j.unreadable) else ""))
    out.append("")
    if res.strata:
        out.append(f"STRATA ({len(res.strata)}, from the user's strata file"
                   + (f"; {res.strata_ignored} rows name items not in this run" if res.strata_ignored else "")
                   + ")")
        for s in res.strata:
            out.append(f"  {_t(s.label)} ({_plural(s.n_items, 'item')})")
            for a in s.agreements:
                out.append(f"    {_t(a.a)} vs {_t(a.b)}: {_agree_line(a)}")
        out.append("")
    out.append("CONFUSION")
    for j in res.judges:
        if j.headline_retest:
            out.append(_confusion_text(j.headline_retest, res.classes))
            out.append("")
    for a in res.pairwise:
        out.append(_confusion_text(a, res.classes))
        out.append("")
    out.append(_t(res.cost_line()))
    out.append(_t(res.band_line()))
    return "\n".join(out)


def drift_text(d: DriftResult) -> str:
    out = [f"DRIFT: test-retest across sessions (run {d.repeat + 1} of each file)",
           f"  A: {_t(d.label_a)}, dated {_t(d.date_a or 'unknown')}",
           f"  B: {_t(d.label_b)}, dated {_t(d.date_b or 'unknown')}"]
    if d.gap:
        out.append(f"  gap: {_t(d.gap)}")
    out.append("")
    for j in d.judges:
        out.append(f"  {_t(j.judge)} A vs B: {_agree_line(j.across)}")
        if j.band is not None:
            out.append(f"    across-session band: +/-{j.band:.3f}")
        for w in (j.within_a, j.within_b):
            if w is not None:
                out.append(f"    within {_t(w.a)} vs {_t(w.b)}: {_agree_line(w)}")
        out.append(f"    {_t(drift_sentence(j))}")
        out.append("")
    for n in d.notes:
        out.append(f"note: {_t(n)}")
    for w in d.small_n_warnings():
        out.append(f"warning: {_t(w)}")
    return "\n".join(out).rstrip()


# ---------------------------------------------------------------- html

CSS = """
:root{
  --paper:#f3f0e6; --ink:#1c1d19; --graphite:#5d5f55; --rule:#d3cdb9; --ledger:#1d4a6b; --stamp:#a3361d;
  --display:"Newsreader",Georgia,"Times New Roman",serif;
  --body:"Public Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,"SF Mono",Menlo,Consolas,monospace;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){ --paper:#171814; --ink:#e7e3d5; --graphite:#a09f93; --rule:#3a3a32; --ledger:#93bcdc; --stamp:#e58d6c; }
}
:root[data-theme="dark"]{ --paper:#171814; --ink:#e7e3d5; --graphite:#a09f93; --rule:#3a3a32; --ledger:#93bcdc; --stamp:#e58d6c; }
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%;background:var(--paper)}
body{margin:0;background:var(--paper);color:var(--ink);font:400 1.0625rem/1.6 var(--body);font-variant-numeric:tabular-nums;overflow-x:hidden}
.frame{max-width:62rem;margin:0 auto;padding:0 16px}
a{color:var(--ledger);text-underline-offset:.2em}
a:focus-visible{outline:2px solid var(--ledger);outline-offset:2px}
.skip{position:absolute;left:-999px}
.skip:focus{left:16px;top:8px;background:var(--paper);padding:.25rem .5rem;z-index:2}
p{margin:0 0 1rem;max-width:66ch;overflow-wrap:anywhere}
h1,h2,h3{font-family:var(--display);font-weight:500;letter-spacing:-.01em;margin:0;overflow-wrap:anywhere}
h1{font-size:2.1rem;line-height:1.12;margin:0 0 1rem}
h2{font-size:1.35rem;line-height:1.25;margin:0 0 .75rem}
h3{font-size:1.05rem;margin:0 0 .4rem}
.mono{font-family:var(--mono);font-size:.8rem;color:var(--graphite)}
.masthead{border-top:4px solid var(--ink);border-bottom:1px solid var(--ink)}
.mast{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:baseline;gap:.2rem 2rem;padding-top:.9rem;padding-bottom:.75rem}
.brand{color:var(--ink);text-decoration:none;font-family:var(--display);font-size:1.6rem;font-weight:600;line-height:1.1;letter-spacing:-.015em}
.register{margin:0;font-family:var(--mono);font-size:.74rem;text-transform:uppercase;letter-spacing:.08em;color:var(--graphite)}
main.frame{padding-top:2rem;padding-bottom:2rem}
.kicker{font-family:var(--mono);font-size:.76rem;text-transform:uppercase;letter-spacing:.09em;color:var(--graphite);margin:0 0 .75rem}
.lede{font-size:1.15rem;line-height:1.5;max-width:60ch}
.pagehead{padding-bottom:1rem}
.summary{padding:1.25rem 0 1.5rem;border-top:1px solid var(--ink);border-bottom:3px double var(--ink)}
.summary h2{font-family:var(--mono);font-size:.76rem;text-transform:uppercase;letter-spacing:.09em;color:var(--graphite);font-weight:500;margin:0 0 .25rem}
.summary .wavers{margin-top:1rem}
.summary .cost{font-family:var(--mono);font-size:.8rem;color:var(--graphite);max-width:none}
.entry{padding:1.5rem 0;border-bottom:1px solid var(--rule)}
.figures{width:100%;border-collapse:collapse}
.figures td{border-bottom:1px solid var(--rule);padding:.85rem 0;vertical-align:baseline}
.fig-n{font-family:var(--mono);font-weight:500;font-size:1.9rem;line-height:1;letter-spacing:-.02em;text-align:right;padding-right:1rem!important;white-space:nowrap;width:1%}
.fig-l{font-size:1rem;line-height:1.4;overflow-wrap:anywhere}
.fig-s{display:block;margin-top:.2rem;font-family:var(--mono);font-size:.76rem;color:var(--graphite)}
.fixture{margin:0 0 1rem}
.stamp{display:inline-block;padding:.15rem .5rem;border:1.5px solid var(--stamp);border-radius:2px;font-family:var(--mono);font-size:.72rem;font-weight:500;text-transform:uppercase;letter-spacing:.08em;line-height:1.5;color:var(--stamp)}
.t-pn{font-family:var(--mono);font-size:11px;fill:var(--graphite)}
.printline{font-family:var(--mono);font-size:.9rem;border-left:3px solid var(--ledger);padding:.5rem .8rem;margin:1rem 0 0;max-width:46rem;overflow-wrap:anywhere}
.tablewrap,.chartwrap{overflow-x:auto;max-width:100%}
.tablewrap:focus-visible,.chartwrap:focus-visible{outline:2px solid var(--ledger);outline-offset:2px}
.warn{font-family:var(--mono);font-size:.8rem;color:var(--ink);border-left:3px solid var(--graphite);padding:.3rem .7rem;max-width:46rem}
table.data{border-collapse:collapse;width:100%;max-width:46rem;margin:0 0 .5rem;font-size:.92rem}
table.data caption{text-align:left;font-family:var(--mono);font-size:.72rem;text-transform:uppercase;letter-spacing:.09em;color:var(--graphite);padding-bottom:.4rem}
table.data th,table.data td{text-align:left;padding:.4rem .9rem .4rem 0;border-bottom:1px solid var(--rule);font-weight:400;vertical-align:baseline}
table.data th[scope=row]{overflow-wrap:break-word;min-width:6.5rem}
table.data thead th{font-family:var(--mono);font-size:.7rem;text-transform:uppercase;letter-spacing:.06em;color:var(--graphite);border-bottom:1px solid var(--ink)}
table.data td.n{font-family:var(--mono);white-space:nowrap}
table.data tr.group th{font-family:var(--display);font-size:1rem;padding-top:.9rem;border-bottom:1px solid var(--ink)}
.ci{font-family:var(--mono);font-size:.8rem;color:var(--graphite);white-space:nowrap}
.conf{display:grid;grid-template-columns:repeat(auto-fill,minmax(17rem,1fr));gap:1rem 2rem;max-width:56rem}
table.cm{border-collapse:collapse;font-family:var(--mono);font-size:.8rem;min-width:16rem}
table.cm caption{text-align:left;font-family:var(--body);font-size:.88rem;color:var(--ink);padding-bottom:.3rem;overflow-wrap:anywhere}
table.cm th,table.cm td{padding:.2rem .45rem;border-bottom:1px solid var(--rule);text-align:right;font-weight:400}
table.cm th{color:var(--graphite);font-size:.7rem}
table.cm td.diag{color:var(--ledger);font-weight:600;text-decoration:underline;text-underline-offset:.2em}
table.cm td.zero{color:var(--graphite)}
.chart{margin:.5rem 0;padding:1rem 0 .75rem;border-top:1px solid var(--ink);border-bottom:1px solid var(--rule);max-width:46rem}
.chart svg{display:block;width:100%;min-width:560px;height:auto;overflow:visible}
.chart text{font-family:var(--body)}
.t-axis{font-size:11px;fill:var(--graphite)}
.t-lab{font-size:13px;fill:var(--ink)}
.t-val{font-family:var(--mono)!important;font-size:12px;fill:var(--ink)}
.grid{stroke:var(--rule);stroke-width:1}
.ref{stroke:var(--graphite);stroke-width:1;stroke-dasharray:3 3}
.stem{stroke-width:2.5}
.m-accent{stroke:var(--ledger);fill:var(--ledger)}
.m-grey{stroke:var(--graphite);fill:var(--graphite)}
.dot{stroke:var(--paper);stroke-width:2}
.sq{stroke:var(--paper);stroke-width:2}
figcaption{margin-top:.6rem;font-size:.88rem;color:var(--graphite);max-width:66ch}
.src{display:block;margin-top:.25rem;font-family:var(--mono);font-size:.74rem;color:var(--graphite);overflow-wrap:anywhere}
ul{padding-left:1.1rem;max-width:66ch}
li{margin-bottom:.4rem}
.colophon{border-top:1px solid var(--ink);padding-top:1rem;padding-bottom:2.5rem;font-size:.88rem;color:var(--graphite)}
.colophon p{margin-bottom:.4rem}
@media (max-width:39.99rem){
  .fig-n{font-size:1.5rem}
  .tablewrap.wide::before,.chartwrap::before{content:"scrolls sideways";display:block;position:sticky;left:0;font-family:var(--mono);font-size:.7rem;letter-spacing:.06em;text-transform:uppercase;color:var(--graphite);padding:0 0 .35rem}
}
@media (min-width:40rem){ h1{font-size:2.5rem} .fig-n{font-size:2.4rem} }
@media (min-width:62rem){
  .entry{display:grid;grid-template-columns:11rem minmax(0,46rem);column-gap:2.5rem}
  .entry>*{grid-column:2}
  .entry>h2{grid-column:1;grid-row:1/span 20;font-size:1.15rem;padding-top:.1rem}
}
@media print{
  :root,:root:not([data-theme="light"]),:root[data-theme="dark"]{--paper:#fff;--ink:#111;--graphite:#444;--rule:#bbb;--ledger:#1d4a6b;--stamp:#a3361d}
  @page{margin:15mm 14mm}
  html,body{background:#fff}
  body{font-size:10pt;line-height:1.45;-webkit-print-color-adjust:exact;print-color-adjust:exact}
  .frame{max-width:none;padding:0}
  main.frame{padding-top:.8rem;padding-bottom:0}
  .skip{display:none}
  h1{font-size:20pt;margin-bottom:.5rem}
  .lede{font-size:11pt}
  .fig-n{font-size:18pt}
  .figures td{padding:.45rem 0}
  .summary{break-after:page;page-break-after:always;padding:.8rem 0 .8rem}
  .entry{display:block;padding:.9rem 0}
  .entry>h2{font-size:13pt;margin-bottom:.4rem;break-after:avoid;page-break-after:avoid}
  table,figure,.printline,tr,.judgeblock{break-inside:avoid;page-break-inside:avoid}
  h3{break-after:avoid;page-break-after:avoid}
  .tablewrap,.chartwrap{overflow:visible}
  .tablewrap::before,.chartwrap::before{display:none!important}
  .chart svg{min-width:0}
  .t-val{display:inline}
  .colophon a[href^="http"]::after{content:" (" attr(href) ")";font-family:var(--mono);font-size:8pt}
}
"""

FONTS = ("https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600"
         "&family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600"
         "&family=Public+Sans:wght@400;600&display=swap")


def _e(s: object) -> str:
    return html.escape(str(s), quote=True)


def _ci(a: Agreement) -> str:
    return f"[{_f(a.lo)}, {_f(a.hi)}]"


def _kcell(a: Agreement) -> str:
    if not a.boot_draws or a.kappa_lo != a.kappa_lo:
        return _f(a.kappa)
    out = f'{_f(a.kappa)} <span class="ci">[{a.kappa_lo:.3f}, {a.kappa_hi:.3f}]</span>'
    if a.kappa_undefined:
        out += f'<span class="fig-s">{a.kappa_undefined:,} of {a.boot_draws:,} draws undefined</span>'
    return out


def _chart(res: CalibrationResult) -> str:
    rows: list[tuple[str, Agreement, bool]] = []
    for j in res.judges:
        if j.headline_retest:
            ra, rb = _runs(j.headline_retest)
            rows.append((f"{j.name}, run {ra} vs run {rb}", j.headline_retest, True))
    for a in res.pairwise:
        rows.append((f"{a.a} vs {a.b}", a, False))
    rows = [r for r in rows if r[1].n]
    if not rows:
        return ""
    W, left, right, top, rh = 560, 16, 540, 26, 46  # inset so the end tick labels stay inside
    lo = min(r[1].lo for r in rows)
    x0 = max(0.0, math.floor((lo - 0.02) * 10) / 10)
    x1 = 1.0
    sx = lambda v: left + (v - x0) / (x1 - x0) * (right - left)  # noqa: E731
    H = top + rh * len(rows) + 8
    step = 0.05 if (x1 - x0) <= 0.4 else 0.1
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="chart-t chart-d">'
             f'<title id="chart-t">Agreement with 95% Wilson intervals</title>'
             f'<desc id="chart-d">'
             + _e("; ".join(f"{lab}: {_f(a.rate)} {_ci(a)}" for lab, a, _ in rows))
             + '</desc>']
    ticks = []
    t = x0
    while t <= x1 + 1e-9:
        ticks.append(sx(t))
        parts.append(f'<text class="t-axis" x="{sx(t):.1f}" y="{top - 12}" text-anchor="middle">{t:.2f}</text>')
        t = round(t + step, 10)
    ref = res.judge(res.reference).headline_retest if res.reference else None
    for i, (label, a, accent) in enumerate(rows):
        y = top + i * rh
        cls = "m-accent" if accent else "m-grey"
        # gridlines and the reference line run only through the bar band, below the label
        for x in ticks:
            parts.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{y + 21}" y2="{y + 39}"/>')
        if ref is not None:
            parts.append(f'<line class="ref" x1="{sx(ref.rate):.1f}" x2="{sx(ref.rate):.1f}" '
                         f'y1="{y + 21}" y2="{y + 39}"/>')
        parts.append(f'<text class="t-lab" x="{left}" y="{y + 15}">{_e(label)}'
                     f'<tspan class="t-pn"> · paired {a.n} / unpaired {a.unpaired}</tspan></text>')
        parts.append(f'<text class="t-val" x="{right}" y="{y + 15}" text-anchor="end">'
                     f'{_f(a.rate)} {_ci(a)}</text>')
        parts.append(f'<line class="stem {cls}" x1="{sx(a.lo):.1f}" x2="{sx(a.hi):.1f}" y1="{y + 30}" y2="{y + 30}"/>')
        if accent:
            parts.append(f'<circle class="dot {cls}" cx="{sx(a.rate):.1f}" cy="{y + 30}" r="5.5"/>')
        else:  # a square marker, so the two kinds of row differ in shape as well as colour
            parts.append(f'<rect class="sq {cls}" x="{sx(a.rate) - 5:.1f}" y="{y + 25}" width="10" height="10"/>')
    parts.append("</svg>")
    return "".join(parts)


def _cm_table(a: Agreement, classes: list[str]) -> str:
    head = "".join(f"<th scope=\"col\">{_e(c)}</th>" for c in classes)
    body = []
    for i, (c, row) in enumerate(zip(classes, a.confusion)):
        cells = []
        for k, v in enumerate(row):
            cl = "diag" if i == k else ("zero" if v == 0 else "")
            cells.append(f'<td class="{cl}">{v}</td>' if cl else f"<td>{v}</td>")
        body.append(f'<tr><th scope="row">{_e(c)}</th>{"".join(cells)}</tr>')
    return (f'<div class="tablewrap"><table class="cm"><caption>{_e(a.a)} (rows) by {_e(a.b)} (columns), '
            f'{a.k} of {a.n} on the diagonal (underlined); {_pn(a)}</caption><thead><tr>'
            f'<td></td>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def _per_class_table(name: str, a: Agreement) -> str:
    ra, rb = _runs(a)
    rows = "".join(
        f'<tr><th scope="row">{_e(c.cls)}</th><td class="n">{c.named_a}</td><td class="n">{c.named_b}</td>'
        f'<td class="n">{c.both}</td><td class="n">{_f(c.specific)}</td><td class="n">{c.flips}</td></tr>'
        for c in a.per_class)
    return (f'<div class="tablewrap"><table class="data"><caption>{_e(name)}, run {ra} vs run {rb}, per class'
            f'</caption><thead><tr><th scope="col">class</th><th scope="col">run {ra}</th>'
            f'<th scope="col">run {rb}</th><th scope="col">both</th><th scope="col">specific</th>'
            f'<th scope="col">flips</th></tr></thead><tbody>{rows}</tbody></table></div>')


def _warn_html(ws: list[str]) -> str:
    if not ws:
        return ""
    shown = "".join(f"<br>{_e(w)}" for w in ws[:6])
    more = f"<br>and {len(ws) - 6} more in the text report" if len(ws) > 6 else ""
    return (f'<p class="warn"><strong>Small samples.</strong> {_plural(len(ws), "figure")} rest on '
            f'fewer than 20 paired items:{shown}{more}</p>')


_TW = re.compile(r'<div class="tablewrap"><table class="(data|cm)">(\s*)<caption>(.*?)</caption>', re.S)


def _label_wrappers(body: str) -> str:
    """Make every table wrapper a focusable, named region, so a keyboard user can scroll it
    and a screen reader announces it; data tables also get the narrow-screen scroll cue."""
    def sub(m):
        label = re.sub(r"<[^>]+>", "", m.group(3))
        wide = " wide" if m.group(1) == "data" else ""
        return (f'<div class="tablewrap{wide}" tabindex="0" role="region" aria-label="{label}">'
                f'<table class="{m.group(1)}">{m.group(2)}<caption>{m.group(3)}</caption>')
    return _TW.sub(sub, body)


def _shell(title: str, register: str, body: str, date: str, footer: str) -> str:
    body = _label_wrappers(body)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(title)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="{FONTS}">
<style>{CSS}</style>
</head>
<body>
<a class="skip" href="#main">Skip to the report</a>
<header class="masthead"><div class="frame mast">
<a class="brand" href="https://dinand.com/research/">Dinand Tinholt · Research</a>
<p class="register">{_e(register)}</p>
</div></header>
<main class="frame" id="main">
{body}
</main>
<footer class="frame colophon">
<p>Generated by judgecal {_version()} on {_e(date)}. {footer}</p>
<p><a href="https://dinand.com/research/">dinand.com/research</a></p>
</footer>
</body>
</html>
"""


def to_html(
    res: CalibrationResult,
    title: str = "Judge consistency report",
    source: Optional[str] = None,
    date: Optional[str] = None,
    notes: Optional[str] = None,
    fixture_note: Optional[str] = None,
) -> str:
    """Render the report. `fixture_note` prints a stamped label at the top of the body; when
    it is omitted and any judge was replayed from a flagged fixture, FIXTURE_NOTE is used."""
    date = date or _dt.date.today().isoformat()
    fnote = _fixture_note(res, fixture_note)
    fnote_html = f'<p class="fixture"><span class="stamp">{_e(fnote)}</span></p>' if fnote else ""
    ref = res.judge(res.reference) if res.reference else None
    r = ref.headline_retest if ref else None
    band = res.band

    figs = []
    if band is not None and r is not None:
        k = _kci(r)
        figs.append(
            f'<tr><td class="fig-n">±{band:.2f}</td><td class="fig-l">Judge noise to print beside '
            f'a number this judge produces.<span class="fig-s">half of {r.disagreements}/{r.n} '
            f'= {r.disagreements / r.n:.3f} test-retest disagreement, {_e(res.reference)}; '
            f'unrounded {band:.3f}; {_pn(r)}</span></td></tr>')
        figs.append(
            f'<tr><td class="fig-n">{_f(r.rate)}</td><td class="fig-l">{_e(res.reference)} agrees with '
            f'its own second run on {r.k} of {r.n} items.<span class="fig-s">Wilson 95% {_ci(r)}, '
            f'Cohen’s κ {_f(r.kappa)}' + (f', bootstrap 95% {_e(k)}' if k else "")
            + f'; {_pn(r)}</span></td></tr>')
        others = [p for p in res.pairwise if res.reference in (p.a, p.b) and p.n]
        if others:
            best = max(others, key=lambda p: p.rate)
            other = best.b if best.a == res.reference else best.a
            inside = r.lo <= best.rate <= r.hi
            figs.append(
                f'<tr><td class="fig-n">{_f(best.rate)}</td><td class="fig-l">{_e(other)} matches '
                f'{_e(res.reference)} on {best.k} of {best.n} items, the closest of the other judges.'
                f'<span class="fig-s">Wilson 95% {_ci(best)}, κ {_f(best.kappa)}; '
                + ("inside" if inside else "outside")
                + f' the reference’s own retest interval {_ci(r)}</span></td></tr>')
    else:
        figs.append('<tr><td class="fig-n">n/a</td><td class="fig-l">No judge was run twice, so '
                    'there is no noise band. Run with two repeats.</td></tr>')

    wav = (f'<p class="wavers">{_e(wavers_paragraph(res.reference, r))}</p>'
           if r is not None else "")
    sens_ref = sensitivity_note(ref) if ref else None
    extra = []
    if sens_ref:
        extra.append(f"<p>{_e(sens_ref)}</p>")
    if res.strata:
        lows = ([(s.label, a) for s in res.strata for a in s.agreements
                 if a.n and (a.a, a.b) == (r.a, r.b)] if r is not None else [])
        if lows:
            lab, a = min(lows, key=lambda t: t[1].rate)
            extra.append(f"<p>Across {len(res.strata)} strata from your strata file, the reference "
                         f"retest is lowest in {_e(lab)}: {a.k} of {a.n} ({_f(a.rate)}, Wilson {_ci(a)}).</p>")

    retest_rows = []
    for j in res.judges:
        if not j.retest:
            retest_rows.append(f'<tr><th scope="row">{_e(j.name)}</th><td class="n" colspan="5">'
                               f'one run only, not measured</td></tr>')
            continue
        for a in j.retest:
            retest_rows.append(
                f'<tr><th scope="row">{_e(a.a)} vs {_e(a.b)}</th><td class="n">{a.k}/{a.n}</td>'
                f'<td class="n">{a.n} / {a.unpaired}</td>'
                f'<td class="n">{_f(a.rate)} <span class="ci">{_ci(a)}</span></td>'
                f'<td class="n">{_kcell(a)}</td><td class="n">±{_f((1 - a.rate) / 2)}</td></tr>')

    pair_rows = "".join(
        f'<tr><th scope="row">{_e(a.a)} vs {_e(a.b)}</th><td class="n">{a.k}/{a.n}</td>'
        f'<td class="n">{a.n} / {a.unpaired}</td>'
        f'<td class="n">{_f(a.rate)} <span class="ci">{_ci(a)}</span></td><td class="n">{_kcell(a)}</td></tr>'
        for a in res.pairwise) or '<tr><td colspan="5">Only one judge, nothing to compare.</td></tr>'

    sens_html = ""
    sens_j = [j for j in res.judges if sensitivity_note(j)]
    if sens_j:
        srows = []
        for j in sens_j:
            for s in j.sensitivity:
                srows.append(f'<tr><th scope="row">{_e(j.name)}, runs 1 to {s.runs}</th>'
                             f'<td class="n">{s.pairs}</td><td class="n">{s.disagreements}/{s.paired}</td>'
                             f'<td class="n">±{_f(s.band)}</td></tr>')
        notes_s = "".join(f"<p>{_e(sensitivity_note(j))}</p>" for j in sens_j)
        sens_html = f"""
<section class="entry" aria-labelledby="h-sens">
<h2 id="h-sens">Repeat sensitivity</h2>
<div class="tablewrap"><table class="data">
<caption>Band pooled over every pair among the first m runs</caption>
<thead><tr><th scope="col">runs</th><th scope="col">pairs</th><th scope="col">flips</th><th scope="col">band</th></tr></thead>
<tbody>{"".join(srows)}</tbody></table></div>
{notes_s}
<p>Every pair of runs also has its own row in the test-retest table. Pairs of runs share items, so
the pooled figure is a description of this run, not an independent estimate.</p>
</section>"""

    wav_all = []
    for j in res.judges:
        if j.headline_retest:
            wav_all.append(f'<div class="judgeblock"><h3>{_e(j.name)}</h3>'
                           f"<p>{_e(wavers_paragraph(j.name, j.headline_retest))}</p>"
                           + _per_class_table(j.name, j.headline_retest) + "</div>")
    wav_html = "".join(wav_all) or "<p>No judge was run twice.</p>"

    strata_html = ""
    if res.strata:
        srows = []
        for s in res.strata:
            srows.append(f'<tr class="group"><th scope="rowgroup" colspan="5">{_e(s.label)} · '
                         f'{_plural(s.n_items, "item")}</th></tr>')
            for a in s.agreements:
                srows.append(f'<tr><th scope="row">{_e(a.a)} vs {_e(a.b)}</th><td class="n">{a.k}/{a.n}</td>'
                             f'<td class="n">{a.n} / {a.unpaired}</td>'
                             f'<td class="n">{_f(a.rate)} <span class="ci">{_ci(a)}</span></td>'
                             f'<td class="n">{_f(a.kappa)}</td></tr>')
        ign = (f" {res.strata_ignored} rows in the strata file name items that are not in this run "
               f"and were left out." if res.strata_ignored else "")
        strata_html = f"""
<section class="entry" aria-labelledby="h-strata">
<h2 id="h-strata">Strata</h2>
<div class="tablewrap"><table class="data">
<caption>Agreement within each stratum from the strata file</caption>
<thead><tr><th scope="col">comparison</th><th scope="col">agree</th><th scope="col">paired / unpaired</th><th scope="col">rate [95%]</th><th scope="col">κ</th></tr></thead>
<tbody>{"".join(srows)}</tbody></table></div>
<p>The strata come from the file you supplied; judgecal never derives them from verdicts or
answers. Items the file does not mention sit under “(no stratum)”. Small strata give wide
intervals; the κ here has no bootstrap interval.{ign}</p>
</section>"""

    low_rows = []
    for j in res.judges:
        for i, s in enumerate(j.low_confidence):
            if s.n == 0 and j.missing[i]:
                continue
            mix = ", ".join(f"{c} {j.class_mix[i].get(c, 0)}" for c in res.classes)
            low_rows.append(
                f'<tr><th scope="row">{_e(j.name)} run {i + 1}</th><td class="n">{s.k}/{s.n}'
                f' <span class="ci">{_f(s.rate)}</span></td><td class="n">{j.unreadable[i]}</td>'
                f'<td class="n">{j.missing[i]}</td><td class="n">{_e(mix)}</td></tr>')

    cms = [_cm_table(j.headline_retest, res.classes) for j in res.judges if j.headline_retest]
    cms += [_cm_table(a, res.classes) for a in res.pairwise]

    line = _e(res.band_line())
    cost = _e(res.cost_line())
    src_html = f'<span class="src">{_e(source)}</span>' if source else ""
    notes_html = f"<p>{_e(notes)}</p>" if notes else ""
    judges_list = ", ".join(_e(j.name) for j in res.judges)
    boot = (f"κ intervals are percentile bootstraps over items, {res.boot_draws:,} draws, seed "
            f"{_e(res.seed)}." if res.boot_draws else "No bootstrap was run for κ.")

    body = f"""{fnote_html}
<section class="pagehead">
<p class="kicker">{_e(date)} · rubric {_e(res.rubric_name)}</p>
<h1>{_e(title)}</h1>
<p class="lede">{_plural(len(res.judges), "judge")} ({judges_list}) classified {_plural(res.n_items, "item")} into
{len(res.classes)} {"class" if len(res.classes) == 1 else "classes"} ({_e(", ".join(res.classes))}), each judge up to {res.repeats} times.</p>
{notes_html}
</section>

<section class="summary" aria-labelledby="h-summary">
<h2 id="h-summary">Summary</h2>
<table class="figures">{"".join(figs)}</table>
{wav}
{"".join(extra)}
{_warn_html(res.small_n_warnings())}
<p class="printline">{line}</p>
<p class="cost">{cost}</p>
{src_html}
</section>

<section class="entry" aria-labelledby="h-agree">
<h2 id="h-agree">Agreement</h2>
<figure class="chart" style="margin-left:0;margin-right:0">
<div class="chartwrap" tabindex="0" role="region" aria-label="Agreement chart">{_chart(res)}</div>
<figcaption>Each marker is raw agreement, with its 95% Wilson interval drawn as a bar; the tables
below hold the figures. A round ledger-blue marker means a judge was set against its own second
run, while the grey squares compare pairs of judges, all on run {res.pair_repeat + 1}. Where a second
judge lands near the dashed line, which marks the reference judge’s test-retest agreement, it
agrees with the first about as often as the first agrees with itself.{src_html}</figcaption>
</figure>
</section>

<section class="entry" aria-labelledby="h-retest">
<h2 id="h-retest">Test-retest</h2>
<div class="tablewrap"><table class="data">
<caption>Each judge against its own repeat</caption>
<thead><tr><th scope="col">runs</th><th scope="col">agree</th><th scope="col">paired / unpaired</th><th scope="col">rate [95%]</th><th scope="col">κ [bootstrap 95%]</th><th scope="col">band</th></tr></thead>
<tbody>{"".join(retest_rows)}</tbody></table></div>
<p>Agreement counts only items with a readable verdict on both runs. Unreadable replies and
missing items leave the denominator and appear as unpaired. The band is half the share of
paired items where a judge changed its class between runs; rounded to two places, it is the
figure to print beside any score this judge produced. {boot}</p>
</section>
{sens_html}
<section class="entry" aria-labelledby="h-wavers">
<h2 id="h-wavers">Where the judge wavers</h2>
{wav_html}
<p>Specific agreement for a class is twice the items both runs put in it, over the verdicts
that named it on either run. Flips counts the paired items where one run named the class
and the other did not, so each flip is counted under two classes.</p>
</section>

<section class="entry" aria-labelledby="h-pairs">
<h2 id="h-pairs">Between judges</h2>
<div class="tablewrap"><table class="data">
<caption>Pairwise, run {res.pair_repeat + 1} of each judge</caption>
<thead><tr><th scope="col">pair</th><th scope="col">agree</th><th scope="col">paired / unpaired</th><th scope="col">rate [95%]</th><th scope="col">κ [bootstrap 95%]</th></tr></thead>
<tbody>{pair_rows}</tbody></table></div>
<p>Cohen’s κ discounts the agreement two judges would reach by chance from their class mix
alone. A cheaper judge whose agreement with the reference sits inside the reference’s own
test-retest interval is, on this rubric, hard to tell apart from it.</p>
</section>
{strata_html}
<section class="entry" aria-labelledby="h-conf">
<h2 id="h-conf">Confidence</h2>
<div class="tablewrap"><table class="data">
<caption>Low-confidence verdicts and class mix per run</caption>
<thead><tr><th scope="col">run</th><th scope="col">low</th><th scope="col">unreadable</th><th scope="col">missing</th><th scope="col">class mix</th></tr></thead>
<tbody>{"".join(low_rows)}</tbody></table></div>
<p>Low means the label “low” or a numeric confidence under 50 of 100. Read this column beside
the retest table to see whether a judge flags the items it later flips.</p>
</section>

<section class="entry" aria-labelledby="h-cm">
<h2 id="h-cm">Confusion</h2>
<div class="conf">{"".join(cms)}</div>
</section>

<section class="entry" aria-labelledby="h-cost">
<h2 id="h-cost">Cost</h2>
<p class="cost mono">{cost}</p>
<p>Cost comes from the token counts the endpoint returned with each reply, at the prices you
passed in US dollars per million tokens. Calls that failed, and retried attempts whose reply
was unreadable, are left out; a replayed judge shows the counts recorded from its original run.</p>
</section>

<section class="entry" aria-labelledby="h-limits">
<h2 id="h-limits">Limits</h2>
<ul>
<li>The band follows a convention from a registered judge-reliability study: half the observed
test-retest disagreement. Print it on its own line beside the sampling interval.</li>
<li>A retest figure cannot separate sampling noise from a model that changed between runs; both
land in the same number.</li>
</ul>
</section>"""
    footer = ("Statistics after a registered judge-reliability study, October 2026: Wilson 95% "
              "intervals, Cohen’s κ, band as half the test-retest disagreement.")
    return _shell("Judge Calibrator Report", "Judge Calibrator · consistency report", body, date, footer)


def drift_html(d: DriftResult, title: str = "Judge drift report", date: Optional[str] = None,
               source: Optional[str] = None) -> str:
    """Render a DriftResult: test-retest across sessions."""
    date = date or _dt.date.today().isoformat()
    rows = []
    paras = []
    for j in d.judges:
        for a, lab in [(j.across, "A vs B")] + [(w, "within") for w in (j.within_a, j.within_b) if w]:
            rows.append(f'<tr><th scope="row">{_e(a.a)} vs {_e(a.b)}</th><td class="n">{_e(lab)}</td>'
                        f'<td class="n">{a.k}/{a.n}</td><td class="n">{a.n} / {a.unpaired}</td>'
                        f'<td class="n">{_f(a.rate)} <span class="ci">{_ci(a)}</span></td>'
                        f'<td class="n">{_kcell(a)}</td></tr>')
        paras.append(f"<p>{_e(drift_sentence(j))}</p>")
    cms = "".join(_cm_table(j.across, d.classes) for j in d.judges)
    notes = "".join(f"<li>{_e(n)}</li>" for n in d.notes)
    notes_html = f"<ul>{notes}</ul>" if notes else ""
    first = d.judges[0]
    fig = (f'<tr><td class="fig-n">{_f(first.across.rate)}</td><td class="fig-l">{_e(first.judge)} '
           f'agrees with itself across sessions on {first.across.k} of {first.across.n} items.'
           f'<span class="fig-s">Wilson 95% {_ci(first.across)}, κ {_f(first.across.kappa)}; '
           f'{_pn(first.across)}</span></td></tr>')
    src_html = f'<span class="src">{_e(source)}</span>' if source else ""
    body = f"""<section class="pagehead">
<p class="kicker">{_e(date)} · test-retest across sessions</p>
<h1>{_e(title)}</h1>
<p class="lede">Session A ({_e(d.label_a)}, dated {_e(d.date_a or "unknown")}) against session B
({_e(d.label_b)}, dated {_e(d.date_b or "unknown")}){", " + _e(d.gap) + " apart" if d.gap else ""}, run {d.repeat + 1}
of each file.</p>
</section>
<section class="summary" aria-labelledby="h-summary">
<h2 id="h-summary">Summary</h2>
<table class="figures">{fig}</table>
{"".join(paras)}
{_warn_html(d.small_n_warnings())}
{notes_html}
{src_html}
</section>
<section class="entry" aria-labelledby="h-drift">
<h2 id="h-drift">Drift</h2>
<div class="tablewrap"><table class="data">
<caption>Same judge, same items, two sessions</caption>
<thead><tr><th scope="col">runs</th><th scope="col">kind</th><th scope="col">agree</th><th scope="col">paired / unpaired</th><th scope="col">rate [95%]</th><th scope="col">κ [bootstrap 95%]</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>
<p>This is test-retest across sessions, the design a registered judge-reliability study (October
2026) used for drift. Anything that changed between the two days lands in the across-session
figure, sampling noise included. Where a file holds two runs, the within-session row shows how
much of that the judge produces on a single day.</p>
</section>
<section class="entry" aria-labelledby="h-cm">
<h2 id="h-cm">Confusion</h2>
<div class="conf">{cms}</div>
</section>"""
    return _shell("Judge Drift Report", "Judge Calibrator · drift report", body, date,
                  "Design after a registered judge-reliability study, October 2026: test-retest across sessions.")


def _version() -> str:
    from . import __version__
    return __version__
