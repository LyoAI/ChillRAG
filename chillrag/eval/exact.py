"""Exact — reference and answer must be the same string, noise stripped.

Case, whitespace, punctuation and full/half-width spacing are presentation,
not content; after they go, the two must agree character for character. This
covers choice letters (``B`` / ``b`` / ``B.`` all flatten to ``b``) and short
fixed-form text alike — never fuzzy, so a near-miss fails.
"""

from __future__ import annotations

import re
from typing import ClassVar

from chillrag.eval.base import Evaluator, Verdict, abstains
from chillrag.eval.page import donut_card, kpis, outcome_bucket, page, stat_strip

_NOISE = re.compile(r"[^\w%％.・·]")


def _flat(text: str) -> str:
    """The content of a short answer: lowercased, punctuation and space gone."""
    return _NOISE.sub("", text.casefold()).rstrip(".")


class Exact(Evaluator):
    """Exact match for choice and short-form answers."""

    name: ClassVar[str] = "exact"

    def score_row(self, row: dict) -> Verdict:
        answer = str(row.get(self.answer_field) or "")
        label = str(row.get(self.label_field) or "")
        if abstains(label):
            ok = abstains(answer)
            return Verdict(0, ok, 1.0 if ok else 0.0,
                           "abstain-correct" if ok else "hallucinated",
                           "" if ok else "reference is N/A; the answer names one anyway")
        if abstains(answer):
            return Verdict(0, False, 0.0, "no-answer", "the answer names nothing")
        ok = _flat(answer) == _flat(label)
        return Verdict(0, ok, 1.0 if ok else 0.0, "exact" if ok else "mismatch",
                       "" if ok else f"{label[:40]!r} vs {answer[:40]!r}")

    def render_report(self, verdicts: list, rows: dict[int, dict],
                      *, answers: str, skipped: int) -> str:
        """The exact-match page: what was expected beside what came back."""
        stats = self.aggregate(verdicts)
        acc = stats["accuracy"]
        tone = "ok" if acc >= 0.8 else ("mid" if acc >= 0.5 else "bad")
        abstained = sum(v.detail == "no-answer" for v in verdicts)
        fabricated = sum(v.detail == "hallucinated" for v in verdicts)
        kpi_items = [("Accuracy", f"{acc:.1%}",
                      f"{stats['correct']} of {stats['scored']} rows", tone),
                     ("Abstained", str(abstained),
                      "no answer against a real reference", "mid" if abstained else ""),
                     ("Fabricated", str(fabricated),
                      "an answer against an N/A reference", "bad" if fabricated else "")]
        buckets: dict[str, int] = {}
        for v in verdicts:
            key = outcome_bucket(v)
            buckets[key] = buckets.get(key, 0) + 1
        overview = (kpis(kpi_items)
                    + stat_strip([("skipped (no label)", skipped)])
                    + '<div class="charts">'
                    + donut_card("Outcomes", buckets, acc)
                    + "</div>")

        data = {"verdicts": [{
            "index": v.index, "ok": v.ok, "detail": v.detail, "note": v.note,
            "question": str(rows.get(v.index, {}).get("question", "")),
            "label": str(rows.get(v.index, {}).get(self.label_field, "")),
            "answer": str(rows.get(v.index, {}).get(self.answer_field, "")),
        } for v in verdicts]}
        n_miss = sum(1 for v in verdicts if not v.ok)
        return page(
            title=f"ChillRAG Eval · {self.name}",
            body=overview + _BODY,
            script=_SCRIPT,
            data=data,
            footer=f"{answers} · {n_miss} not matching")


_BODY = """
  <h2>Rows</h2>
  <div class="toolbar">
    <label><input type="checkbox" id="show-all"> show matching rows too</label>
    <span class="cnt" id="shown-cnt"></span>
  </div>
  <div class="table-card">
    <table id="vtable">
      <thead><tr>
        <th class="num">index</th><th>verdict</th><th>question</th>
        <th>expected</th><th>answered</th><th>note</th>
      </tr></thead>
      <tbody></tbody>
    </table>
  </div>
"""

_SCRIPT = """
(function () {
  var V = CR_DATA.verdicts;
  var openRows = {};
  var showAll = document.getElementById("show-all");
  var tbody = document.querySelector("#vtable tbody");

  function rowHtml(v) {
    var main = "<tr class='main' data-i='" + v.index + "'>" +
      "<td class='num'>" + v.index + "</td>" +
      "<td><span class='badge " + (v.ok ? "ok" : "bad") + "'>" + (v.ok ? "✓" : "✗") +
      " " + crEsc(v.detail) + "</span></td>" +
      "<td>" + crEsc(crClip(v.question, 44)) + "</td>" +
      "<td class='mono'>" + crEsc(crClip(v.label, 26)) + "</td>" +
      "<td class='mono'>" + crEsc(crClip(v.answer, 30)) + "</td>" +
      "<td>" + crEsc(crClip(v.note, 40)) + "</td></tr>";
    if (!openRows[v.index]) return main;
    var detail = "<tr class='expand'><td colspan='6'><div class='expand-grid'>" +
      "<div><div class='kv'>question</div><div class='full'>" + crEsc(v.question) + "</div></div>" +
      "<div><div class='kv'>expected</div><div class='full box'>" + crEsc(v.label) + "</div>" +
      "<div class='kv'>answered</div><div class='full box'>" + crEsc(v.answer) + "</div>" +
      (v.note ? "<div class='kv'>note</div><div class='full'>" + crEsc(v.note) + "</div>" : "") +
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
"""
