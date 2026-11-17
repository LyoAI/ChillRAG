"""The shared page skeleton every evaluator's report is built on.

Each evaluator owns its page *content* — the overview it considers
meaningful, the table, the script that renders its verdicts — because what
a scorer should show is its own business. This module owns only the frame
they all share: visual tokens, light/dark theme, favicon, the overview
builders (KPI cards, stat strip, outcome donut, accuracy bars) and the
``crEsc`` helpers. An evaluator calls :func:`page` with its body markup and
script, and gets back one self-contained document.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
def _esc(v) -> str:
    """Escape anything renderable — html.escape insists on str first."""
    return (str(v).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))

_FAVICON = ('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
            '<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
            '<stop offset="0" stop-color="%2393a88e"/><stop offset="1" stop-color="%238aa2b0"/>'
            '</linearGradient></defs><rect width="100" height="100" rx="26" fill="url(%23g)"/>'
            '<g stroke="%23ffffff" stroke-width="9" stroke-linecap="round">'
            '<path d="M50 22v56"/><path d="M26.5 35.5l47 29"/><path d="M26.5 64.5l47-29"/>'
            "</g></svg>")


def outcome_bucket(verdict) -> str:
    """correct / wrong / abstained / fabricated — the four outcome classes a
    page's donut shows. Detail tags carry the nuance; the donut needs a class.
    """
    if verdict.ok:
        return "correct"
    if verdict.detail in ("no-value", "no-answer"):
        return "abstained"
    if verdict.detail == "hallucinated":
        return "fabricated"
    return "wrong"


def kpis(items: list[tuple[str, str, str, str]]) -> str:
    """The headline cards: ``(label, value, sub, tone)``, tone ``""/ok/mid/bad``."""
    cards = []
    for label, value, sub, tone in items:
        bar = f'<div class="accent-bar" style="background:var(--{tone or "accent"})"></div>'
        cards.append(
            f'<div class="kpi {tone}">{bar}<div class="label">{_esc(label)}</div>'
            f'<div class="value">{_esc(value)}</div>'
            + (f'<div class="sub">{_esc(sub)}</div>' if sub else "")
            + "</div>")
    return f'<div class="kpis">{"".join(cards)}</div>'


def stat_strip(items: list[tuple[str, str]]) -> str:
    """The small label/value strips under the KPI cards."""
    return '<div class="secondary">' + "".join(
        f'<div class="stat"><span class="label">{_esc(k)}</span>'
        f'<span class="value">{_esc(v)}</span></div>'
        for k, v in items) + "</div>"


_OUTCOME_COLORS = {"correct": "var(--ok)", "wrong": "var(--bad)",
                   "abstained": "var(--muted)", "fabricated": "var(--warn)"}
_OUTCOME_LABELS = {"correct": "Correct", "wrong": "Wrong",
                   "abstained": "Abstained", "fabricated": "Fabricated"}


def donut_card(title: str, buckets: dict[str, int], accuracy: float | None) -> str:
    """An outcome-distribution donut with the accuracy at its center."""
    total = sum(buckets.values())
    cx = cy = 70
    r = 54
    segs: list[str] = []
    angle = -math.pi / 2
    for key in ("correct", "wrong", "abstained", "fabricated"):
        n = buckets.get(key, 0)
        if not n or not total:
            continue
        sweep = 2 * math.pi * n / total
        a1 = min(angle + sweep, angle + 2 * math.pi)
        x0, y0 = cx + r * math.cos(angle), cy + r * math.sin(angle)
        x1, y1 = cx + r * math.cos(a1), cy + r * math.sin(a1)
        large = 1 if (a1 - angle) > math.pi else 0
        if sweep >= 2 * math.pi - 1e-9:  # a full circle needs two arcs
            segs.append(
                f'<path d="M {cx} {cy} L {x0:.2f} {y0:.2f} A {r} {r} 0 0 1 {cx + r} {cy} '
                f'A {r} {r} 0 0 1 {x0:.2f} {y0:.2f} Z" fill="{_OUTCOME_COLORS[key]}" '
                f'stroke="var(--bg-card)" stroke-width="2"/>')
        else:
            segs.append(
                f'<path d="M {cx} {cy} L {x0:.2f} {y0:.2f} A {r} {r} 0 {large} 1 {x1:.2f} {y1:.2f} Z" '
                f'fill="{_OUTCOME_COLORS[key]}" stroke="var(--bg-card)" stroke-width="2"/>')
        angle += sweep
    center = "—" if accuracy is None else f"{accuracy:.1%}"
    svg = ('<svg viewBox="0 0 140 140" class="donut" role="img" aria-label="outcome distribution">'
           + "".join(segs)
           + f'<text x="{cx}" y="{cy - 2}" text-anchor="middle" class="donut-num">{_esc(center)}</text>'
           + f'<text x="{cx}" y="{cy + 14}" text-anchor="middle" class="donut-label">Accuracy</text></svg>')
    legend = "".join(
        f'<li><span class="dot" style="background:{_OUTCOME_COLORS[key]}"></span>'
        f'{_OUTCOME_LABELS[key]}<b>{buckets.get(key, 0)}</b></li>'
        for key in ("correct", "wrong", "abstained", "fabricated") if buckets.get(key))
    return (f'<div class="chart-card"><h3>{_esc(title)}</h3>'
            f'<div class="donut-wrap">{svg}<ul class="legend">{legend}</ul></div></div>')


def bars_card(title: str, rows: list[tuple[str, int, int]]) -> str:
    """Horizontal accuracy bars: ``(name, correct, total)`` per stratum."""
    out = []
    for name, correct, total in rows:
        pct = correct / total if total else 0.0
        out.append(
            f'<div class="bar-row"><span class="bar-label" title="{_esc(name)}">{_esc(name)}</span>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{pct * 100:.1f}%"></div></div>'
            f'<span class="bar-val">{correct}/{total} · {pct:.0%}</span></div>')
    body = "".join(out) or '<p class="empty">(no data)</p>'
    return f'<div class="chart-card"><h3>{_esc(title)}</h3>{body}</div>'


def page(*, title: str, body: str, script: str, data: dict, footer: str) -> str:
    """One self-contained document: the shared frame around an evaluator's
    ``body`` markup and ``script``, with ``data`` embedded as JSON."""
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (_TEMPLATE
            .replace("__TITLE__", _esc(title))
            .replace("__FAVICON__", _FAVICON)
            .replace("__BODY__", body)
            .replace("__SCRIPT__", script)
            .replace("__FOOTER__", _esc(footer))
            .replace("__GENERATED__", generated)
            .replace("__DATA__", json.dumps(data, ensure_ascii=False, default=str)
                     .replace("</", "<\\/")))


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="icon" href='__FAVICON__'>
<style>
:root {
  --bg: #f3f5f8; --bg-card: #ffffff; --text: #101828; --muted: #667085;
  --border: #e4e7ec; --accent: #2563eb; --accent-soft: #eff4ff;
  --ok: #16a34a; --ok-soft: #e8f7ee; --bad: #dc2626; --bad-soft: #fdecec;
  --warn: #d97706; --warn-soft: #fdf3e3; --muted-soft: #eef0f3;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0b1220; --bg-card: #121b2d; --text: #e6eaf2; --muted: #8b95a8;
    --border: #22304d; --accent: #60a5fa; --accent-soft: #16233d;
    --ok: #4ade80; --ok-soft: #12291c; --bad: #f87171; --bad-soft: #331a1c;
    --warn: #fbbf24; --warn-soft: #33270f; --muted-soft: #1a2438;
  }
}
:root[data-theme="dark"] {
  --bg: #0b1220; --bg-card: #121b2d; --text: #e6eaf2; --muted: #8b95a8;
  --border: #22304d; --accent: #60a5fa; --accent-soft: #16233d;
  --ok: #4ade80; --ok-soft: #12291c; --bad: #f87171; --bad-soft: #331a1c;
  --warn: #fbbf24; --warn-soft: #33270f; --muted-soft: #1a2438;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 14px/1.55 -apple-system, "PingFang SC", "Microsoft YaHei", "Segoe UI", sans-serif; }
.wrap { max-width: 1180px; margin: 0 auto; padding: 20px 24px 48px; }
.mono { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; }

header.top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 18px; }
header.top h1 { font-size: 19px; margin: 0; }
.theme-btn { border: 1px solid var(--border); background: var(--bg-card); color: var(--text);
  border-radius: 8px; padding: 5px 12px; cursor: pointer; font-size: 13px; }

h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .08em;
  color: var(--muted); margin: 26px 2px 10px; }

.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
.kpi { background: var(--bg-card); border: 1px solid var(--border); border-radius: 14px;
  padding: 14px 18px 12px; position: relative; overflow: hidden; }
.kpi .accent-bar { position: absolute; left: 0; top: 0; bottom: 0; width: 4px; }
.kpi .label { font-size: 12px; color: var(--muted); }
.kpi .value { font-size: 30px; font-weight: 700; letter-spacing: -.02em; }
.kpi .sub { font-size: 12px; color: var(--muted); }
.kpi.ok .value { color: var(--ok); }
.kpi.bad .value { color: var(--bad); }
.kpi.mid .value { color: var(--warn); }

.secondary { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px;
  margin-top: 12px; }
.stat { background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px;
  padding: 8px 14px; display: flex; justify-content: space-between; align-items: baseline; }
.stat .label { font-size: 12px; color: var(--muted); }
.stat .value { font-size: 16px; font-weight: 650; }

.charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 12px; margin-top: 12px; }
.chart-card { background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 14px; padding: 14px 16px; }
.chart-card h3 { margin: 0 0 10px; font-size: 13px; color: var(--text); }
.donut-wrap { display: flex; align-items: center; gap: 14px; }
.donut { width: 150px; height: 150px; flex: none; }
.donut-num { font-size: 23px; font-weight: 700; fill: var(--text); }
.donut-label { font-size: 9px; fill: var(--muted); letter-spacing: .08em; }
.legend { list-style: none; margin: 0; padding: 0; font-size: 12.5px; width: 100%; }
.legend li { display: flex; align-items: center; gap: 7px; padding: 3.5px 0; color: var(--muted); }
.legend li b { margin-left: auto; color: var(--text); }
.dot { width: 9px; height: 9px; border-radius: 3px; flex: none; }
.bar-row { display: grid; grid-template-columns: minmax(74px, 130px) 1fr 92px; gap: 10px;
  align-items: center; padding: 3.5px 0; font-size: 12.5px; }
.bar-label { color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.bar-track { background: var(--muted-soft); border-radius: 5px; height: 9px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 5px;
  background: linear-gradient(90deg, var(--accent), var(--ok)); }
.bar-val { text-align: right; color: var(--muted); font-variant-numeric: tabular-nums; }
.empty { color: var(--muted); font-size: 12.5px; margin: 4px 0; }

.toolbar { display: flex; align-items: center; gap: 12px; margin: 4px 0 10px; }
.toolbar label { color: var(--muted); font-size: 13px; cursor: pointer; }
.toolbar .cnt { color: var(--muted); font-size: 12.5px; }

.table-card { background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 14px; overflow: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
thead th { position: sticky; top: 0; background: var(--bg-card); text-align: left;
  font-size: 11.5px; text-transform: uppercase; letter-spacing: .05em; color: var(--muted);
  border-bottom: 1px solid var(--border); padding: 9px 12px; white-space: nowrap; }
tbody td { padding: 8px 12px; border-bottom: 1px solid var(--border); vertical-align: top; }
tbody tr.main { cursor: pointer; }
tbody tr.main:hover { background: var(--accent-soft); }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.badge { display: inline-block; border-radius: 6px; padding: 1px 8px; font-size: 12px;
  font-weight: 600; white-space: nowrap; }
.badge.ok { background: var(--ok-soft); color: var(--ok); }
.badge.bad { background: var(--bad-soft); color: var(--bad); }

tr.expand td { background: var(--accent-soft); }
.expand-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px 24px; padding: 4px 0; }
.expand-grid .full { white-space: pre-wrap; word-break: break-word; font-size: 13px; }
.expand-grid .box { background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 10px; padding: 10px 14px; }
.kv { font-size: 12.5px; color: var(--muted); margin-top: 8px; }
.kv b { color: var(--text); }
@media (max-width: 900px) { .expand-grid { grid-template-columns: 1fr; } }

footer { margin-top: 22px; color: var(--muted); font-size: 12px; display: flex;
  justify-content: space-between; flex-wrap: wrap; gap: 8px; }
</style>
</head>
<body>
<div class="wrap">
  <header class="top">
    <h1>__TITLE__</h1>
    <button class="theme-btn" id="theme-btn">🌙 Dark</button>
  </header>

__BODY__

  <footer>
    <span>__FOOTER__</span>
    <span>generated __GENERATED__ · ChillRAG</span>
  </footer>
</div>

<script type="application/json" id="page-data">__DATA__</script>
<script>
function crEsc(v) {
  return String(v == null ? "" : v).replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}
function crClip(v, n) { v = String(v == null ? "" : v); return v.length > n ? v.slice(0, n) + "…" : v; }
var CR_DATA = JSON.parse(document.getElementById("page-data").textContent);
(function () {
  var btn = document.getElementById("theme-btn");
  function applyTheme(t) {
    document.documentElement.dataset.theme = t;
    btn.textContent = t === "dark" ? "☀️ Light" : "🌙 Dark";
    try { localStorage.setItem("chillrag-eval-theme", t); } catch (e) {}
  }
  var saved = null;
  try { saved = localStorage.getItem("chillrag-eval-theme"); } catch (e) {}
  applyTheme(saved || (window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark" : "light"));
  btn.onclick = function () {
    applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
  };
})();
</script>
<script>
__SCRIPT__
</script>
</body>
</html>
"""
