"""HTML report for agent episode traces (``<TRACE_DIR>/<user>/<session>.jsonl``).

Renders the trace directory into one self-contained HTML file — inline CSS/JS,
zero external assets, renders offline anywhere — and opens it in the browser.
Episodes appear as a filterable, searchable, sortable table; clicking a row
expands the episode's scalars and its full message transcript. A ``TRACE``
argument deep-links straight to one episode.

    python -m chillrag.utils.trace_view                  # every session
    python -m chillrag.utils.trace_view SESSION [TRACE]  # narrowed, deep-linked
    python -m chillrag.utils.trace_view --out x.html --no-open

Like the project's standalone report, the template lives apart from any
logic: ``render_traces_html(payload)`` takes plain data and returns the whole
document, so this module stays a pure data → HTML function.
"""

from __future__ import annotations

import argparse
import json
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

from chillrag.config import TRACE_DIR


# ------------------------------------------------------------------ data --

def _read(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _session_files(pattern: str | None) -> list[Path]:
    """Every ``<user>/<session>.jsonl``, narrowed by exact-or-prefix match."""
    files = sorted(TRACE_DIR.glob("*/*.jsonl"))
    if pattern is None:
        return files
    exact = [f for f in files if f.stem == pattern]
    return exact or [f for f in files if f.stem.startswith(pattern)]


def _chain(record: dict) -> list[str]:
    """Tool names in call order, consecutive repeats collapsed to ``name×k``."""
    names = [call["function"]["name"]
             for message in record.get("messages", []) if message.get("tool_calls")
             for call in message["tool_calls"]]
    runs: list[list] = []
    for name in names:
        if runs and runs[-1][0] == name:
            runs[-1][1] += 1
        else:
            runs.append([name, 1])
    return [name if count == 1 else f"{name}×{count}" for name, count in runs]


def _build_payload(files: list[Path]) -> dict:
    sessions = []
    for path in files:
        episodes = []
        for r in _read(path):
            ts = r.get("ts", "")
            try:
                shown = datetime.fromisoformat(ts).astimezone().strftime("%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                shown = ts
            episodes.append({
                "user": path.parent.name,
                "session": path.stem,
                "trace_id": r.get("trace_id", ""),
                "ts": ts, "ts_show": shown,
                "question": r.get("question", ""),
                "answer": r.get("answer", ""),
                "num_tool_calls": r.get("num_tool_calls", 0),
                "terminated": r.get("terminated", "?"),
                "total_tokens": r.get("total_tokens") or 0,
                "duration_s": r.get("duration_s") or 0,
                "source": r.get("source", "-"),
                "mode": r.get("mode", "-"),
                "lang": r.get("lang", "-"),
                "folder": r.get("folder", "-"),
                "compacted": bool(r.get("compacted")),
                "tool_chain": _chain(r),
                "messages": r.get("messages", []),
            })
        sessions.append({
            "user": path.parent.name,
            "session": path.stem,
            "episodes": episodes,
            "n": len(episodes),
            "tokens": sum(e["total_tokens"] for e in episodes),
            "seconds": sum(e["duration_s"] for e in episodes),
            "kb": round(path.stat().st_size / 1024),
            "last": max((e["ts_show"] for e in episodes), default="-"),
        })
    return {"trace_dir": str(TRACE_DIR), "sessions": sessions}


# -------------------------------------------------------------- template --

_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ChillRAG Traces</title>
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
.wrap { max-width: 1280px; margin: 0 auto; padding: 20px 24px 48px; }
.mono { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; }
.num { font-variant-numeric: tabular-nums; }

header.top { display: flex; flex-wrap: wrap; gap: 12px; align-items: center;
  justify-content: space-between; margin-bottom: 14px; }
header.top h1 { font-size: 19px; margin: 0; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; }
.chip { background: var(--bg-card); border: 1px solid var(--border); border-radius: 999px;
  padding: 2px 10px; font-size: 12px; color: var(--muted); }
.chip b { color: var(--text); font-weight: 600; }
.theme-btn { border: 1px solid var(--border); background: var(--bg-card); color: var(--text);
  border-radius: 8px; padding: 5px 12px; cursor: pointer; font-size: 13px; }

h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .08em;
  color: var(--muted); margin: 22px 2px 8px; }

.sessions { display: flex; flex-direction: column; gap: 6px; }
.session-chip { display: flex; flex-wrap: wrap; align-items: baseline; gap: 10px;
  background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px;
  padding: 7px 14px; cursor: pointer; font-size: 13px; }
.session-chip:hover { border-color: var(--accent); }
.session-chip.active { border-color: var(--accent); background: var(--accent-soft); }
.session-chip .u { font-weight: 650; }
.session-chip .s { font-weight: 600; }
.session-chip .meta { color: var(--muted); font-size: 12px; }

.toolbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 14px 0 10px; }
.filters { display: flex; flex-wrap: wrap; gap: 6px; }
.fbtn { border: 1px solid var(--border); background: var(--bg-card); color: var(--muted);
  border-radius: 999px; padding: 3px 12px; font-size: 12.5px; cursor: pointer; }
.fbtn.active { background: var(--accent); border-color: var(--accent); color: #fff; }
.fbtn .cnt { opacity: .75; margin-left: 3px; }
.fbtn.export { background: var(--accent); border-color: var(--accent); color: #fff; font-weight: 600; }
.search { margin-left: auto; border: 1px solid var(--border); background: var(--bg-card);
  color: var(--text); border-radius: 8px; padding: 6px 12px; font-size: 13px; width: 240px; }

.table-card { background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 14px; overflow: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
thead th { position: sticky; top: 0; background: var(--bg-card); text-align: left;
  font-size: 11.5px; text-transform: uppercase; letter-spacing: .05em; color: var(--muted);
  border-bottom: 1px solid var(--border); padding: 9px 12px; cursor: pointer;
  user-select: none; white-space: nowrap; z-index: 1; }
thead th.no-sort { cursor: default; }
tbody td { padding: 8px 12px; border-bottom: 1px solid var(--border); vertical-align: top; }
tbody tr.main:hover { background: var(--accent-soft); }
tbody tr.main { cursor: pointer; }
td.num, th.num { text-align: right; }
.tid { font-weight: 650; font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px; }
.where { color: var(--muted); font-size: 12px; white-space: nowrap; }
.q-cell { max-width: 360px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.badge { display: inline-block; border-radius: 6px; padding: 1px 8px; font-size: 12px;
  font-weight: 600; white-space: nowrap; }
.badge.answer { background: var(--ok-soft); color: var(--ok); }
.badge.off { background: var(--warn-soft); color: var(--warn); }
.badge.compact { background: var(--muted-soft); color: var(--muted); }

tr.detail td { background: var(--accent-soft); border-bottom: 1px solid var(--border); }
.detail-grid { display: grid; grid-template-columns: 1fr 320px; gap: 10px 24px; padding: 6px 0; }
.detail-q, .detail-a { white-space: pre-wrap; font-size: 13px; }
.detail-a { max-height: 320px; overflow: auto; background: var(--bg-card);
  border: 1px solid var(--border); border-radius: 10px; padding: 10px 14px; }
.kv { font-size: 12.5px; color: var(--muted); }
.kv b { color: var(--text); font-weight: 600; }
.chain { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px; }

.transcript { margin-top: 4px; display: flex; flex-direction: column; gap: 8px; }
.msg { border: 1px solid var(--border); border-radius: 10px; background: var(--bg-card);
  padding: 8px 12px; font-size: 13px; }
.msg .role { font-size: 11px; text-transform: uppercase; letter-spacing: .07em;
  color: var(--muted); margin-bottom: 4px; }
.msg.user { border-left: 3px solid var(--accent); }
.msg.assistant { border-left: 3px solid var(--ok); }
.msg.tool { border-left: 3px solid var(--warn); }
.msg.system .body { color: var(--muted); }
.msg .body { white-space: pre-wrap; word-break: break-word; }
.msg pre { margin: 0; white-space: pre-wrap; word-break: break-word;
  font: 12px/1.5 ui-monospace, Menlo, Consolas, monospace; }
.call { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px;
  margin: 4px 0; }
.call summary { cursor: pointer; }
.call b { color: var(--warn); }
details.snap summary { cursor: pointer; color: var(--muted); font-size: 12px; }
footer { margin-top: 22px; color: var(--muted); font-size: 12px; display: flex;
  justify-content: space-between; flex-wrap: wrap; gap: 8px; }
@media (max-width: 980px) { .detail-grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="wrap">
  <header class="top">
    <div>
      <h1>🧾 ChillRAG Traces</h1>
      <div class="chips" id="meta-chips"></div>
    </div>
    <button class="theme-btn" id="theme-btn">🌙 Dark</button>
  </header>

  <h2>Sessions</h2>
  <div class="sessions" id="sessions"></div>

  <h2>Episodes <span id="shown-cnt"></span></h2>
  <div class="toolbar">
    <div class="filters" id="filters"></div>
    <button class="fbtn export" id="export-btn"
            title="Download the episodes currently in view as JSONL">⬇ Export JSONL</button>
    <input class="search" id="search" type="search" placeholder="Search question / answer / session…">
  </div>
  <div class="table-card">
    <table id="etable">
      <thead><tr>
        <th data-key="ts_show">time</th>
        <th data-key="user">user / session</th>
        <th data-key="trace_id" class="no-sort">trace</th>
        <th data-key="question">question</th>
        <th data-key="source" class="no-sort">source</th>
        <th data-key="num_tool_calls" class="num">tools</th>
        <th data-key="total_tokens" class="num">tokens</th>
        <th data-key="duration_s" class="num">dur</th>
        <th data-key="terminated" class="no-sort">end</th>
      </tr></thead>
      <tbody></tbody>
    </table>
  </div>

  <footer>
    <span>__FOOTER__</span>
    <span>generated __GENERATED__ · ChillRAG</span>
  </footer>
</div>

<script type="application/json" id="cr-data">__DATA__</script>
<script>
(function () {
  var DATA = JSON.parse(document.getElementById("cr-data").textContent);
  var E = [];
  DATA.sessions.forEach(function (s) {
    s.episodes.forEach(function (e) { E.push(e); });
  });
  var state = { session: null, filter: "all", query: "", sortKey: null, sortDir: 1 };
  var openRows = {};
  var view = E;

  // ---- theme ----
  var btn = document.getElementById("theme-btn");
  function applyTheme(t) {
    document.documentElement.dataset.theme = t;
    btn.textContent = t === "dark" ? "☀️ Light" : "🌙 Dark";
    try { localStorage.setItem("chillrag-trace-theme", t); } catch (e) {}
  }
  var saved = null;
  try { saved = localStorage.getItem("chillrag-trace-theme"); } catch (e) {}
  applyTheme(saved || (window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
  btn.onclick = function () {
    applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
  };

  function esc(v) {
    return String(v == null ? "" : v).replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }
  function fmtDur(s) { return s >= 1 ? s.toFixed(1) + "s" : Math.round(s * 1000) + "ms"; }

  // ---- header chips ----
  var users = {};
  E.forEach(function (e) { users[e.user] = (users[e.user] || 0) + 1; });
  var tokens = E.reduce(function (a, e) { return a + e.total_tokens; }, 0);
  var chips = [["trace dir", DATA.trace_dir], ["users", Object.keys(users).length],
               ["sessions", DATA.sessions.length], ["episodes", E.length],
               ["tokens", tokens.toLocaleString()]];
  document.getElementById("meta-chips").innerHTML = chips
    .map(function (c) { return '<span class="chip">' + esc(c[0]) + ': <b>' + esc(c[1]) + "</b></span>"; })
    .join("");

  // ---- session list ----
  var sessBox = document.getElementById("sessions");
  sessBox.innerHTML = DATA.sessions.map(function (s, i) {
    return '<div class="session-chip" data-i="' + i + '">' +
      '<span class="u">' + esc(s.user) + '</span><span class="s mono">' + esc(s.session) + "</span>" +
      '<span class="meta">' + s.n + " episodes · " + s.tokens.toLocaleString() + " tokens · " +
      Math.round(s.seconds) + "s · " + s.kb + " KB · last " + esc(s.last) + "</span></div>";
  }).join("");
  sessBox.onclick = function (e) {
    var chip = e.target.closest(".session-chip");
    if (!chip) return;
    var s = DATA.sessions[+chip.dataset.i];
    var key = s.user + "/" + s.session;
    state.session = state.session === key ? null : key;
    document.querySelectorAll(".session-chip").forEach(function (c) {
      var other = DATA.sessions[+c.dataset.i];
      c.classList.toggle("active", state.session === other.user + "/" + other.session);
    });
    render();
  };

  // ---- filters ----
  var FILTERS = [["all", "All"], ["answer", "✓ answered"], ["off", "⚠ other end"]];
  function countFor(f) {
    if (f === "all") return E.length;
    if (f === "off") return E.filter(function (e) { return e.terminated !== "answer" || e.compacted; }).length;
    return E.filter(function (e) { return e.terminated === "answer" && !e.compacted; }).length;
  }
  document.getElementById("filters").innerHTML = FILTERS.map(function (f) {
    return '<button class="fbtn' + (f[0] === "all" ? " active" : "") + '" data-f="' + f[0] + '">' +
      f[1] + '<span class="cnt">' + countFor(f[0]) + "</span></button>";
  }).join("");
  document.getElementById("filters").onclick = function (e) {
    var b = e.target.closest(".fbtn");
    if (!b) return;
    state.filter = b.dataset.f;
    document.querySelectorAll(".fbtn").forEach(function (x) { x.classList.toggle("active", x === b); });
    render();
  };
  document.getElementById("search").oninput = function (e) {
    state.query = e.target.value.trim().toLowerCase();
    render();
  };

  // ---- table ----
  var NUMERIC = { num_tool_calls: 1, total_tokens: 1, duration_s: 1 };
  var tbody = document.querySelector("#etable tbody");
  document.querySelector("#etable thead").onclick = function (e) {
    var th = e.target.closest("th");
    if (!th || th.classList.contains("no-sort") || !th.dataset.key) return;
    var key = th.dataset.key;
    if (state.sortKey === key) { state.sortDir *= -1; } else { state.sortKey = key; state.sortDir = 1; }
    render();
  };

  function transcriptHtml(msgs) {
    if (!msgs || !msgs.length) return '<div class="kv">✂ compacted — the transcript of this episode was not retained</div>';
    var names = {};
    return '<div class="transcript">' + msgs.map(function (m) {
      var role = m.role || "?";
      var inner = "";
      (m.tool_calls || []).forEach(function (c) {
        names[c.id] = c.function.name;
        inner += '<details class="call"><summary>→ <b>' + esc(c.function.name) + "</b> " +
          esc(String(c.function.arguments || "").slice(0, 120)) +
          (String(c.function.arguments || "").length > 120 ? " …" : "") + "</summary><pre>" +
          esc(c.function.arguments || "") + "</pre></details>";
      });
      var content = m.content || "";
      if (content) {
        if (role === "tool") {
          var who = names[m.tool_call_id] || "?";
          var short = content.length > 300;
          inner += '<details class="snap"' + (short ? "" : " open") + "><summary>" +
            esc(who) + " · " + esc(m.tool_call_id || "") + " · " + content.length + " chars</summary><pre>" +
            esc(content) + "</pre></details>";
        } else if (role === "system" && content.length > 200) {
          inner += '<details class="snap"><summary>system prompt · ' + content.length +
            ' chars</summary><pre>' + esc(content) + "</pre></details>";
        } else {
          inner += '<div class="body">' + esc(content) + "</div>";
        }
      }
      return '<div class="msg ' + esc(role) + '"><div class="role">' + esc(role) + "</div>" + inner + "</div>";
    }).join("") + "</div>";
  }

  function rowHtml(e, i) {
    var open = !!openRows[e.trace_id];
    var terminated = e.terminated === "answer" && !e.compacted
      ? '<span class="badge answer">answer</span>'
      : '<span class="badge off">' + esc(e.terminated) + "</span>";
    if (e.compacted) terminated += ' <span class="badge compact">✂ compacted</span>';
    var main = "<tr class='main' data-tid='" + esc(e.trace_id) + "'>" +
      "<td class='num' title='" + esc(e.ts) + "'>" + esc(e.ts_show) + "</td>" +
      "<td class='where'>" + esc(e.user) + " / <span class='mono'>" + esc(e.session) + "</span></td>" +
      "<td class='tid'>" + esc(e.trace_id.slice(0, 8)) + "</td>" +
      "<td class='q-cell' title='" + esc(e.question) + "'>" + esc(e.question) + "</td>" +
      "<td>" + esc(e.source) + "</td>" +
      "<td class='num'>" + e.num_tool_calls + "</td>" +
      "<td class='num'>" + e.total_tokens.toLocaleString() + "</td>" +
      "<td class='num'>" + fmtDur(e.duration_s) + "</td>" +
      "<td>" + terminated + "</td></tr>";
    if (!open) return main;
    var detail = "<tr class='detail'><td colspan='9'><div class='detail-grid'>" +
      "<div><div class='detail-q'>" + esc(e.question) + "</div>" + transcriptHtml(e.messages) + "</div>" +
      "<div><div class='detail-a'>" + esc(e.answer) + "</div>" +
      "<div class='kv' style='margin-top:8px'>session <b>" + esc(e.user) + "/" + esc(e.session) + "</b></div>" +
      "<div class='kv'>source <b>" + esc(e.source) + "</b> · mode <b>" + esc(e.mode) +
      "</b> · lang <b>" + esc(e.lang) + "</b></div>" +
      "<div class='kv'>folder <b>" + esc(e.folder) + "</b> · tokens <b>" +
      e.total_tokens.toLocaleString() + "</b> · " + fmtDur(e.duration_s) + "</div>" +
      "<div class='kv'>chain <span class='chain'>" +
      (e.tool_chain.length ? esc(e.tool_chain.join(" → ")) : "—") + "</span></div>" +
      "</div></div></td></tr>";
    return main + detail;
  }

  function render() {
    var key;
    var qs = E.filter(function (e) {
      if (state.session && e.user + "/" + e.session !== state.session) return false;
      if (state.filter === "off") {
        if (e.terminated === "answer" && !e.compacted) return false;
      } else if (state.filter === "answer" && (e.terminated !== "answer" || e.compacted)) return false;
      if (state.query) {
        var hay = (e.question + " " + e.answer + " " + e.user + " " + e.session + " " +
          e.source + " " + e.trace_id).toLowerCase();
        if (hay.indexOf(state.query) === -1) return false;
      }
      return true;
    });
    if (state.sortKey) {
      var k = state.sortKey, d = state.sortDir;
      qs = qs.slice().sort(function (a, b) {
        var va = a[k], vb = b[k];
        if (NUMERIC[k]) { va = +va || 0; vb = +vb || 0; return (va - vb) * d; }
        return String(va == null ? "" : va).localeCompare(String(vb == null ? "" : vb)) * d;
      });
    }
    document.getElementById("shown-cnt").textContent = "（" + qs.length + " / " + E.length + "）";
    view = qs;
    tbody.innerHTML = qs.map(rowHtml).join("");
  }

  tbody.onclick = function (e) {
    var tr = e.target.closest("tr.main");
    if (!tr) return;
    openRows[tr.dataset.tid] = !openRows[tr.dataset.tid];
    render();
  };

  // ---- export JSONL of the current view ----
  document.getElementById("export-btn").onclick = function () {
    var lines = view.map(function (e) { return JSON.stringify(e); }).join("\\n");
    var blob = new Blob([lines], { type: "application/x-ndjson;charset=utf-8" });
    var a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "chillrag-traces.jsonl";
    document.body.appendChild(a);
    a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 500);
  };

  // ---- deep link: #trace-<id8> expands and scrolls to that episode ----
  var hash = decodeURIComponent(location.hash || "");
  if (hash.indexOf("#trace-") === 0) {
    var wanted = hash.slice("#trace-".length);
    var hit = E.find(function (e) { return e.trace_id.slice(0, 8) === wanted; });
    if (hit) openRows[hit.trace_id] = true;
  }
  render();
  if (hash.indexOf("#trace-") === 0) {
    var el = document.querySelector("tr.main[data-tid^='" + hash.slice(7) + "']");
    if (el) el.scrollIntoView({ block: "start" });
  }
})();
</script>
</body>
</html>
"""


def render_traces_html(payload: dict) -> str:
    """Render the standalone trace-report document for a trace directory."""
    sessions = payload.get("sessions", [])
    n = sum(s["n"] for s in sessions)
    tokens = sum(s["tokens"] for s in sessions)
    footer = f"{len(sessions)} session(s) · {n} episode(s) · {tokens:,} tokens"
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (_TEMPLATE
            .replace("__FOOTER__", footer)
            .replace("__GENERATED__", generated)
            .replace("__DATA__", data_json))


# -------------------------------------------------------------------- cli --

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session", nargs="?", help="session name (exact or prefix)")
    parser.add_argument("trace", nargs="?", help="trace_id prefix to deep-link to")
    parser.add_argument("--out", type=Path, default=None,
                        help="report path (default: traces/report.html)")
    parser.add_argument("--no-open", action="store_true", help="write the file, don't open a browser")
    args = parser.parse_args(argv)

    files = _session_files(args.session)
    if not files:
        reason = f"no session matches {args.session!r}" if args.session else "no traces yet"
        print(f"{reason} under {TRACE_DIR}")
        return 1

    out_path = args.out or (TRACE_DIR / "report.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_traces_html(_build_payload(files)), encoding="utf-8")

    url = f"file://{out_path.resolve()}"
    if args.trace:
        url += f"#trace-{args.trace[:8]}"
    print(f"{len(files)} session(s) → {out_path}")
    if not args.no_open:
        webbrowser.open(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
