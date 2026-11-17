"""FinNumeric — value comparison for answers that name a number.

The reference (``label``, plus an optional ``unit`` column) and the model's
answer are both parsed as a value with a unit word (``92,599,365.5万元``,
``19.3%``, ``(1.5)``, ``3.51亿元``), then matched in three widening steps:

1. **Coarser precision**: both values quantize to the coarser of the two
   precisions they claim (``3.51亿元`` claims ±0.005亿 = ±50万, so it matches
   a ``35142万元`` reference) — Decimal arithmetic, half-even.
2. **Round tolerance**: the answer rounded to the reference's precision within
   ``EVAL_ROUND_TOL``.
3. **Relative bound**, cross-unit only: money 1% / rate 5% — a materially
   different value never slips through.

Units carry a dimension: money scales against 元 (十/百/千/万/亿/兆 and
thousand/million/billion, compound units multiply, a bare currency is base 1,
foreign currency is a different dimension and never converts); rates against
a pure ratio (``%`` is ×0.01, ``倍`` is ×1). An answer without a unit word
implies the reference's unit and is never converted; incompatible dimensions
(``万元`` vs ``%``) fail outright.

A bare reference is read through the answer's dimension: money in 元
(``85,247,150千元`` matches ``85247150000``), rates under **both** readings
datasets use — as a ratio (``8.6%`` matches ``0.086``) and as a percent
number (``12.11%`` matches ``12.11``, ``-1387%`` matches ``-1386.98``) —
whichever one agrees at the claimed precision. The ``unit`` column exists
for labels written as bare numbers in a fixed non-base unit; a unit written
inside the label always wins.

A reference marked N/A requires abstention: no value in the answer, or the
row counts as fabricated. Unit-bearing numbers in an answer outrank bare
ones, so years and ids mentioned in prose don't capture the value.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from math import log10
from typing import ClassVar

from chillrag.config import EVAL_MONEY_REL_TOL, EVAL_RATE_REL_TOL, EVAL_ROUND_TOL
from chillrag.eval.base import Evaluator, Verdict, abstains
from chillrag.eval.page import bars_card, donut_card, kpis, outcome_bucket, page, stat_strip

# A number, including thousand-grouped ones ("92,599,365.5" is one number,
# not three); the grouped alternative must come first or finditer splits it.
_NUM_RE = re.compile(
    r"-?\d{1,3}(?:[,，]\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
# Chars a unit word may be built from, once the number itself has ended.
_UNIT_CHARS = re.compile(r"[一-鿿A-Za-z%％×]+")
_OPENERS = re.compile(r"[\s（([【\"“'':：,，、]")

_CN_SCALE = {"十": 1e1, "百": 1e2, "佰": 1e2, "仟": 1e3, "千": 1e3, "万": 1e4, "亿": 1e8, "兆": 1e12}
_EN_SCALE = (
    ("trillion", 1e12), ("billion", 1e9), ("million", 1e6), ("thousand", 1e3),
    ("hundred", 1e2), ("bn", 1e9), ("mn", 1e6), ("k", 1e3),
)
_BASE_CURRENCY = ("元", "圆", "块", "人民币", "yuan", "rmb", "cny", "renminbi")
_FOREIGN_CURRENCY = ("美元", "美金", "港币", "港元", "欧元", "日元", "英镑",
                     "usd", "hkd", "eur", "jpy", "gbp", "dollar", "euro")
_PERCENT = {"%", "％", "percent", "pct"}
_MULTIPLES = {"倍", "倍数", "倍率", "times", "time", "x", "×"}


def _meaningful_precision(literal: str) -> int:
    """Decimal places a written number claims — trailing zeros are noise
    (``11`` / ``11.0`` → 0, ``11.44`` → 2, ``98240271.5`` → 1)."""
    if "." not in literal:
        return 0
    return len(literal.split(".", 1)[1].rstrip("0"))


def _parse_value(text: str) -> tuple[float, str, str] | None:
    """``(value, unit, literal)`` of the most answer-like number in ``text``.

    Unit-bearing numbers win over bare ones — prose is full of years and ids,
    and only the value travels with its unit. ``None`` when nothing parses.
    """
    bare: tuple[float, str, str] | None = None
    for m in _NUM_RE.finditer(text):
        i = m.end()
        while i < len(text) and _OPENERS.match(text[i]):
            i += 1
        unit = _UNIT_CHARS.match(text[i:])
        unit_word = (unit.group(0) if unit else "")[:12]
        literal = m.group(0)
        plain = literal.replace(",", "").replace("，", "")
        value = -float(plain) if _wrapped_negative(text, m) else float(plain)
        # A known unit (money, rate, even foreign currency) marks this number
        # as the answer; bare ones are prose and only ever a fallback.
        if unit_word and _known_unit(unit_word):
            return value, unit_word, literal
        if bare is None:
            bare = (value, "", literal)
    return bare


def _known_unit(unit: str) -> bool:
    """Whether a unit word names a dimension, even an incompatible one."""
    return (_money_scale(unit) is not None or _rate_scale(unit) is not None
            or any(w in unit.lower() for w in _FOREIGN_CURRENCY))


def _wrapped_negative(text: str, m: re.Match) -> bool:
    """Whether the match sits in accounting parentheses: ``（1.5）`` / ``(1.5)``."""
    before, after = text[: m.start()].rstrip(), text[m.end():].lstrip()
    return before.endswith(("(", "（")) and after.startswith((")", "）"))


def _money_scale(unit: str) -> float | None:
    """Scale of a money-dimension unit against 元, or ``None`` if not money."""
    u = unit.lower()
    if any(w in u for w in _FOREIGN_CURRENCY):
        return None
    scale, found = 1.0, False
    for word, s in _EN_SCALE:
        if re.search(rf"(?<![a-z]){word}(?![a-z])", u):
            scale *= s
            found = True
            u = re.sub(rf"(?<![a-z]){word}(?![a-z])", " ", u)  # so "bn" can't re-count
    for ch, s in _CN_SCALE.items():
        if ch in u:
            scale *= s
            found = True
    if not found and any(w in u for w in _BASE_CURRENCY):
        return 1.0
    return scale if found else None


def _rate_scale(unit: str) -> float | None:
    """Scale of a rate unit against a pure ratio (``%`` → 0.01, ``倍`` → 1)."""
    u = unit.strip().lower()
    if u in _PERCENT:
        return 0.01
    if u in _MULTIPLES:
        return 1.0
    return None


def _close(ref: tuple, ans: tuple, dimension: str) -> tuple[bool, str]:
    """One reading pair, at three widening steps: quantize both to the coarser
    of their precisions-in-base, then the reference's round tolerance (in its
    own writing's scale), then — dimensional pairs only — the relative bound.
    """
    try:
        ref_value, ref_dp, ref_tol = ref
        ans_value, ans_dp, _ = ans
        coarse = Decimal(1).scaleb(-min(ref_dp, ans_dp))
        if (Decimal(str(ref_value)).quantize(coarse, rounding=ROUND_HALF_EVEN)
                == Decimal(str(ans_value)).quantize(coarse, rounding=ROUND_HALF_EVEN)):
            return True, "exact"
        ref_step = Decimal(1).scaleb(-ref_dp)
        if abs(Decimal(str(ans_value)).quantize(ref_step, rounding=ROUND_HALF_EVEN)
               - Decimal(str(ref_value))) <= Decimal(str(ref_tol)):
            return True, "round-tol"
        rel = {"rate": EVAL_RATE_REL_TOL, "money": EVAL_MONEY_REL_TOL}.get(dimension)
        if rel is not None and (abs(Decimal(str(ans_value)) - Decimal(str(ref_value)))
                                <= Decimal(str(rel)) * abs(Decimal(str(ref_value)))):
            return True, f"rel-{dimension}"
        return False, "mismatch"
    except InvalidOperation:
        return False, "overflow"


def _readings(value: float, unit: str, precision: int,
              dimension: str) -> list[tuple[float, int, float]] | None:
    """How one side reads in the dimension's base: ``(value, dp, round_tol)``.

    A unitful side has exactly one reading — its value scaled into base, with
    its precision carried over (``35142万元`` reads as 351,420,000 元 with the
    slop of 0.1万元). A bare side reads literally, and — rates only — a second
    time as a percent number (``12.11`` also reads as ``12.11%``), since
    datasets write rates both ways. ``None`` = this unit has no reading in
    this dimension (foreign currency in the money dimension).
    """
    tol = EVAL_ROUND_TOL  # the slop is 0.1 in each reading's own writing space
    if not unit:
        out = [(value, precision, tol)]
        if dimension == "rate":
            out.append((value * 0.01, precision + 2, tol * 0.01))
        return out
    scale = _money_scale(unit) if dimension == "money" else _rate_scale(unit)
    if scale is None:
        return None
    return [(value * scale, precision - round(log10(scale)), tol * scale)]


def _compare(ref_value: float, ref_unit: str, ref_precision: int,
             value: float, unit: str, precision: int) -> tuple[bool, str, str, str]:
    """Match two values under their units; ``(ok, detail, note, dimension)``.

    The dimension comes from whichever side carries a unit. Each side is then
    read into that dimension's base — 元 for money, a pure ratio for rates —
    and every reading combination is tried: a bare rate reference matches a
    percent answer as a ratio (``8.6%`` = ``0.086``) or as a percent number
    (``12.11%`` = ``12.11``). Only when neither side carries a unit does the
    comparison fall back to literal digit equality; an incompatible pair
    (``万元`` vs ``%``, foreign currency) fails outright.
    """
    def dim_of(u: str) -> str:
        if _money_scale(u) is not None:
            return "money"
        if _rate_scale(u) is not None:
            return "rate"
        return ""

    if unit and ref_unit and dim_of(unit) and dim_of(ref_unit) \
            and dim_of(unit) != dim_of(ref_unit):
        return False, "unit-mismatch", f"{unit!r} and {ref_unit!r} are different dimensions", ""
    dimension = dim_of(unit) or dim_of(ref_unit)
    if dimension and unit and not dim_of(unit):
        return False, "unit-mismatch", f"{unit!r} converts to no reference", ""
    if dimension and ref_unit and not dim_of(ref_unit):
        return False, "unit-mismatch", f"{ref_unit!r} converts to no reference", ""
    if not dimension and (unit or ref_unit) and (
            (unit and _known_unit(unit)) or (ref_unit and _known_unit(ref_unit))):
        return False, "unit-mismatch", "the unit converts to no bare reference", ""

    ref_readings = _readings(ref_value, ref_unit, ref_precision, dimension) \
        if dimension else [(ref_value, ref_precision, EVAL_ROUND_TOL * 10.0 ** (-ref_precision))]
    ans_readings = _readings(value, unit, precision, dimension) \
        if dimension else [(value, precision, 0.0)]
    # The relative bound is a cross-conversion fallback: identical units mean
    # no conversion happened, and 1% slop must not excuse a wrong digit run.
    allow_rel = bool(dimension) and (not unit or not ref_unit or unit != ref_unit)

    notes: list[str] = []
    for r in ref_readings:
        for a in ans_readings:
            ok, detail = _close(r, a, dimension if allow_rel else "")
            if ok:
                note = (f"base ({dimension}): {r[0]:,.6g} vs {a[0]:,.6g}"
                        if dimension and (unit or ref_unit) else "")
                return True, detail, note, dimension
            notes.append(f"{r[0]:,.6g} vs {a[0]:,.6g}")
    note = f"base ({dimension}): " + " · ".join(notes) if dimension else ""
    return False, "mismatch", note, dimension


class FinNumeric(Evaluator):
    """Value accuracy with unit conversion and claimed-precision matching."""

    name: ClassVar[str] = "fin-numeric"
    unit_field: ClassVar[str] = "unit"

    def score_row(self, row: dict) -> Verdict:
        answer = str(row.get(self.answer_field) or "")
        label = str(row.get(self.label_field) or "")

        if abstains(label):
            ok = abstains(answer)
            return Verdict(0, ok, 1.0 if ok else 0.0,
                           "abstain-correct" if ok else "hallucinated",
                           "" if ok else "reference is N/A; the answer names a value anyway",
                           {"reference-na": True})
        if abstains(answer):
            return Verdict(0, False, 0.0, "no-value", "the answer names no value")

        parsed = _parse_value(answer)
        if parsed is None:
            return Verdict(0, False, 0.0, "no-value", "the answer carries no number")
        value, unit, literal = parsed
        reference = _parse_value(label)
        if reference is None:
            return Verdict(0, False, 0.0, "label-not-numeric",
                           f"label {label[:40]!r} parses to no value")
        ref_value, ref_unit, ref_literal = reference
        # The label's own unit wins; the unit column only gives a bare-number
        # label ("19.3" + unit "%") its dimension, never overwrites one.
        ref_unit = ref_unit or str(row.get(self.unit_field) or "").strip()
        ref_precision = _meaningful_precision(ref_literal)
        precision = _meaningful_precision(literal)
        ok, detail, note, dimension = _compare(
            ref_value, ref_unit, ref_precision, value, unit, precision)
        data = {"ref_value": f"{ref_value:,.6g}", "ref_unit": ref_unit,
                "value": f"{value:,.6g}", "unit": unit,
                "ref_precision": ref_precision, "precision": precision,
                "dimension": dimension}
        return Verdict(0, ok, 1.0 if ok else 0.0, detail,
                       note or ("" if ok else f"{value:,.6g}{unit} vs {ref_value:,.6g}{ref_unit}"),
                       data)

    def aggregate(self, verdicts: list) -> dict:
        """Base accuracy plus the abstention view and the fallback histogram."""
        stats = super().aggregate(verdicts)
        na_rows = sum(v.detail in ("abstain-correct", "hallucinated") for v in verdicts)
        stats["reference-na"] = na_rows
        if na_rows and len(verdicts) > na_rows:
            core = [v for v in verdicts if v.detail not in ("abstain-correct", "hallucinated")]
            stats["accuracy-excl-na"] = sum(v.ok for v in core) / len(core)
        stats["abstained"] = sum(v.detail == "no-value" for v in verdicts)
        stats["hallucinated"] = sum(v.detail == "hallucinated" for v in verdicts)
        return stats

    # ------------------------------------------------------------- report --

    def render_report(self, verdicts: list, rows: dict[int, dict],
                      *, answers: str, skipped: int) -> str:
        """The numeric page: every question marked right or wrong, the
        reference against the answered value with its unit and the rule that
        decided."""
        stats = self.aggregate(verdicts)
        acc = stats["accuracy"]
        tone = "ok" if acc >= 0.8 else ("mid" if acc >= 0.5 else "bad")
        kpi_items = [("Accuracy", f"{acc:.1%}",
                      f"{stats['correct']} of {stats['scored']} rows", tone)]
        if "accuracy-excl-na" in stats:
            kpi_items.append(("Accuracy excl. N/A", f"{stats['accuracy-excl-na']:.1%}",
                              f"{stats['reference-na']} N/A reference(s) excluded", ""))
        abstained, fabricated = stats.get("abstained", 0), stats.get("hallucinated", 0)
        kpi_items.append(("Abstained", str(abstained),
                          "no value against a real reference", "mid" if abstained else ""))
        kpi_items.append(("Fabricated", str(fabricated),
                          "a value against an N/A reference", "bad" if fabricated else ""))

        buckets: dict[str, int] = {}
        for v in verdicts:
            key = outcome_bucket(v)
            buckets[key] = buckets.get(key, 0) + 1
        groups: dict[str, list] = {}
        for v in verdicts:
            folder = str(rows.get(v.index, {}).get("folder") or "-")
            groups.setdefault(folder, []).append(v)
        overview = (kpis(kpi_items)
                    + stat_strip([("skipped (no label)", skipped),
                                  ("reference-na", stats.get("reference-na", 0))])
                    + '<div class="charts">'
                    + donut_card("Outcomes", buckets, acc)
                    + bars_card("Accuracy by folder",
                                [(name, sum(v.ok for v in vs), len(vs))
                                 for name, vs in sorted(groups.items())])
                    + "</div>")

        data = {"verdicts": [{
            "index": v.index, "ok": v.ok, "detail": v.detail, "note": v.note,
            "question": str(rows.get(v.index, {}).get("question", "")),
            "label": str(rows.get(v.index, {}).get(self.label_field, "")),
            "answer": str(rows.get(v.index, {}).get(self.answer_field, "")),
            "data": v.data,
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
    <label><input type="checkbox" id="only-bad"> only wrong answers</label>
    <span class="cnt" id="shown-cnt"></span>
  </div>
  <div class="table-card">
    <table id="vtable">
      <thead><tr>
        <th class="num">index</th><th>verdict</th><th>question</th>
        <th>reference</th><th>answered</th><th>note</th>
      </tr></thead>
      <tbody></tbody>
    </table>
  </div>
"""

_SCRIPT = """
(function () {
  var V = CR_DATA.verdicts;
  var openRows = {};
  var onlyBad = document.getElementById("only-bad");
  var tbody = document.querySelector("#vtable tbody");

  function answeredCell(v) {
    var d = v.data || {};
    if (d.value !== undefined) return d.value + " " + (d.unit || "(bare)");
    return crClip(v.answer, 30);
  }

  function rowHtml(v) {
    var main = "<tr class='main' data-i='" + v.index + "'>" +
      "<td class='num'>" + v.index + "</td>" +
      "<td><span class='badge " + (v.ok ? "ok" : "bad") + "'>" + (v.ok ? "✓" : "✗") +
      " " + crEsc(v.detail) + "</span></td>" +
      "<td>" + crEsc(crClip(v.question, 44)) + "</td>" +
      "<td class='mono'>" + crEsc(crClip(v.label, 26)) + "</td>" +
      "<td class='mono'>" + crEsc(crClip(answeredCell(v), 30)) + "</td>" +
      "<td>" + crEsc(crClip(v.note, 40)) + "</td></tr>";
    if (!openRows[v.index]) return main;
    var d = v.data || {};
    var compare = "";
    if (d.ref_value !== undefined) {
      compare = "reference <b>" + crEsc(d.ref_value) + " " + crEsc(d.ref_unit || "(bare)") +
        "</b> (" + d.ref_precision + " dp) · answered <b>" + crEsc(d.value) + " " +
        crEsc(d.unit || "(bare)") + "</b> (" + d.precision + " dp)" +
        (d.dimension ? " · " + crEsc(d.dimension) + " conversion" : "");
    }
    var detail = "<tr class='expand'><td colspan='6'><div class='expand-grid'>" +
      "<div><div class='kv'>question</div><div class='full'>" + crEsc(v.question) + "</div>" +
      (compare ? "<div class='kv'>comparison</div><div class='full'>" + compare + "</div>" : "") +
      "</div>" +
      "<div><div class='kv'>reference</div><div class='full box'>" + crEsc(v.label) + "</div>" +
      "<div class='kv'>answered</div><div class='full box'>" + crEsc(v.answer) + "</div>" +
      (v.note ? "<div class='kv'>note</div><div class='full'>" + crEsc(v.note) + "</div>" : "") +
      "</div></div></td></tr>";
    return main + detail;
  }

  function render() {
    var shown = V.filter(function (v) { return !onlyBad.checked || !v.ok; });
    document.getElementById("shown-cnt").textContent =
      shown.length + " / " + V.length + " scored";
    tbody.innerHTML = shown.map(rowHtml).join("");
  }
  onlyBad.onchange = render;
  tbody.onclick = function (e) {
    var tr = e.target.closest("tr.main");
    if (!tr) return;
    openRows[tr.dataset.i] = !openRows[tr.dataset.i];
    render();
  };
  render();
})();
"""
