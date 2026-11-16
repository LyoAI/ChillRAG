"""HTML report for an eval run — the wrong answers, front and center.

Like the trace report, this module is a pure data → document function:
``render_eval_html(payload)`` takes the run's stats and its scored verdicts
(each joined back to its answers row) and returns one self-contained file —
inline CSS/JS, zero external assets, renders offline. The overview carries
the stat tiles; the table opens on the rows that did not match, with one
click expanding a row to the full answer, reference and note.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

_FAVICON = ('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
            '<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
            '<stop offset="0" stop-color="%2393a88e"/><stop offset="1" stop-color="%238aa2b0"/>'
            '</linearGradient></defs><rect width="100" height="100" rx="26" fill="url(%23g)"/>'
            '<g stroke="%23ffffff" stroke-width="9" stroke-linecap="round">'
            '<path d="M50 22v56"/><path d="M26.5 35.5l47 29"/><path d="M26.5 64.5l47-29"/>'
            "</g></svg>")

_KEYS_ORDER = ("scored", "accuracy", "correct", "reference-na", "accuracy-excl-na",
               "abstained", "hallucinated")
_PCT = {"accuracy", "accuracy-excl-na"}


def _tiles(stats: dict[str, Any], skipped: int) -> list[tuple[str, str]]:
    """The overview, pre-formatted so the page stays a dumb renderer."""
    tiles = [("evaluator", str(stats.get("name", "-"))),
             ("scored", f"{stats.get('scored', 0)} · skipped {skipped}")]
    for key in _KEYS_ORDER:
        if key not in stats:
            continue
        value = stats[key]
        shown = f"{value:.1%}" if key in _PCT else str(value)
        tiles.append((key, shown))
    details = stats.get("details") or {}
    if details:
        tiles.append(("details", " · ".join(f"{k} {n}" for k, n in details.items())))
    return tiles


def render_eval_html(payload: dict) -> str:
    """Render the standalone eval report for one scored answers file."""
    body = dict(payload)
    body["tiles"] = _tiles(payload.get("stats", {}), payload.get("skipped", 0))
    n_miss = sum(1 for v in payload.get("verdicts", []) if not v.get("ok"))
    footer = (f"{payload.get('answers', '')} · {payload.get('evaluator', '')} · "
              f"{n_miss} not matching")
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    data = json.dumps(body, ensure_ascii=False, default=str).replace("</", "<\\/>")
    return (_TEMPLATE
            .replace("__FAVICON__", _FAVICON)
            .replace("__FOOTER__", footer)
            .replace("__GENERATED__", generated)
            .replace("__DATA__", data))


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ChillRAG Eval</title>
<link rel="icon" href='__FAVICON__'>
<style>
:root {
  --bg: #f3f5f8; --bg-card: #ffffff; --text: #101828; --muted: #667085;
  --border: #e4e7ec; --accent: #2563eb; --accent-soft: #eff4ff;
  --ok: #16a34a; --ok-soft: #e8f7ee; --bad: #dc2626; --bad-soft: #fdecec;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0b1220; --bg-card: #121b2d; --text: #e6eaf2; --muted: #8b95a8;
    --border: #22304d; --accent: #60a5fa; --accent-soft: #16233d;
    --ok: #4ade80; --ok-soft: #12291c; --bad: #f87171; --bad-soft: #331a1c;
  }
}
:root[data-theme="dark"] {
  --bg: #0b1220; --bg-card: #121b2d; --text: #e6eaf2; --muted: #8b95a8;
  --border: #22304d; --accent: #60a5fa; --accent-soft: #16233d;
  --ok: #4ade80; --ok-soft: #12291c; --bad: #f87171; --bad-soft: #331a1c;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 14px/1.55 -apple-system, "PingFang SC", "Microsoft YaHei", "Segoe UI", sans-serif; }
.wrap { max-width: 1180px; margin: 0 auto; padding: 20px 24px 48px; }
.mono { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; }

header.top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; }
header.top h1 { font-size: 19px; margin: 0; }
.theme-btn { border: 1px solid var(--border); background: var(--bg-card); color: var(--text);
  border-radius: 8px; padding: 5px 12px; cursor: pointer; font-size: 13px; }

h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .08em;
  color: var(--muted); margin: 22px 2px 8px; }

.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 8px; }
.tile { background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px;
  padding: 10px 14px; }
.tile .v { font-size: 20px; font-weight: 700; font-variant-numeric: tabular-nums;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.tile .k { font-size: 11px; text-transform: uppercase; letter-spacing: .07em;
  color: var(--muted); margin-top: 2px; }

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
td.num { text-align: right; font-variant-numeric: tabular-nums; }
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
    <h1>☑ ChillRAG Eval</h1>
    <button class="theme-btn" id="theme-btn">🌙 Dark</button>
  </header>

  <h2>Overview</h2>
  <div class="stats" id="tiles"></div>

  <h2>Rows</h2>
  <div class="toolbar">
    <label><input type="checkbox" id="show-all"> show matching rows too</label>
    <span class="cnt" id="shown-cnt"></span>
  </div>
  <div class="table-card">
    <table id="vtable">
      <thead><tr>
        <th class="num">index</th><th>detail</th><th>question</th>
        <th>label</th><th>answer</th><th>note</th>
      </tr></thead>
      <tbody></tbody>
    </table>
  </div>

  <footer>
    <span>__FOOTER__</span>
    <span>generated __GENERATED__ · ChillRAG</span>
  </footer>
</div>

<script type="application/json" id="eval-data">__DATA__</script>
<script>
(function () {
  var DATA = JSON.parse(document.getElementById("eval-data").textContent);
  var V = DATA.verdicts || [];
  var openRows = {};

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

  function esc(v) {
    return String(v == null ? "" : v).replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  var tiles = document.getElementById("tiles");
  tiles.innerHTML = (DATA.tiles || []).map(function (t) {
    return '<div class="tile"><div class="v">' + esc(t[1]) +
      '</div><div class="k">' + esc(t[0]) + "</div></div>";
  }).join("");

  var showAll = document.getElementById("show-all");
  var tbody = document.querySelector("#vtable tbody");

  function rowHtml(v) {
    var main = "<tr class='main' data-i='" + v.index + "'>" +
      "<td class='num'>" + v.index + "</td>" +
      "<td><span class='badge " + (v.ok ? "ok" : "bad") + "'>" + esc(v.detail) + "</span></td>" +
      "<td>" + esc(String(v.question || "").slice(0, 60)) + "</td>" +
      "<td class='mono'>" + esc(String(v.label || "").slice(0, 30)) + "</td>" +
      "<td class='mono'>" + esc(String(v.answer || "").slice(0, 60)) + "</td>" +
      "<td>" + esc(String(v.note || "").slice(0, 50)) + "</td></tr>";
    if (!openRows[v.index]) return main;
    var detail = "<tr class='expand'><td colspan='6'><div class='expand-grid'>" +
      "<div><div class='kv'>question</div><div class='full'>" + esc(v.question) + "</div></div>" +
      "<div><div class='kv'>label</div><div class='full box'>" + esc(v.label) + "</div>" +
      "<div class='kv'>answer</div><div class='full box'>" + esc(v.answer) + "</div>" +
      (v.note ? "<div class='kv'>note</div><div class='full'>" + esc(v.note) + "</div>" : "") +
      "</div></td></tr>";
    return main + detail;
  }

  function render() {
    var shown = V.filter(function (v) { return showAll.checked || !v.ok; });
    document.getElementById("shown-cnt").textContent =
      shown.length + " / " + V.length + " scored";
    tbody.innerHTML = shown.map(rowHtml).join("");
  }
  showAll.onchange = render;
  tbody.onclick = function (e) {
    var tr = e.target.closest("tr.main");
    if (!tr) return;
    openRows[tr.dataset.i] = !openRows[tr.dataset.i];
    render();
  };
  render();
})();
</script>
</body>
</html>
"""
