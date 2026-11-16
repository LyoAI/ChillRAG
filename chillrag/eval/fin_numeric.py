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
(``万元`` vs ``%``) fail outright. The ``unit`` column exists for labels
written as bare numbers — a unit written inside the label always wins.

A reference marked N/A requires abstention: no value in the answer, or the
row counts as fabricated. Unit-bearing numbers in an answer outrank bare
ones, so years and ids mentioned in prose don't capture the value.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from typing import ClassVar

from chillrag.config import EVAL_MONEY_REL_TOL, EVAL_RATE_REL_TOL, EVAL_ROUND_TOL
from chillrag.eval.base import Evaluator, Verdict, abstains

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


def _compare(ref_value: float, ref_unit: str, ref_precision: int,
             value: float, unit: str, precision: int) -> tuple[bool, str, str]:
    """Match two values under their units; ``(ok, detail, note)``.

    Units convert first — money dimension into money, rate into rate, an
    incompatible pair fails. The converted value then quantizes to the
    coarser of the two claimed precisions (the answer's precision, expressed
    in the reference's unit, scales by the conversion factor), with the
    round-tolerance and cross-unit relative bounds as fallbacks.
    """
    factor = Decimal(1)
    note = ""
    dimension = ""
    if unit and ref_unit and unit != ref_unit:
        ref_scale, ans_scale = _money_scale(ref_unit), _money_scale(unit)
        if ref_scale is not None and ans_scale is not None:
            dimension = "money"
        else:
            ref_scale, ans_scale = _rate_scale(ref_unit), _rate_scale(unit)
            if ref_scale is None or ans_scale is None:
                return False, "unit-mismatch", f"{unit!r} and {ref_unit!r} are different dimensions"
            dimension = "rate"
        value = value * ans_scale / ref_scale
        factor = Decimal(str(ans_scale)) / Decimal(str(ref_scale))
        note = f"{unit} → {ref_unit}: {value:,.6g} vs {ref_value:,.6g}"
    try:
        # Steps are powers of ten, so the coarser of the two claimed precisions
        # is an exponent comparison — multiplying Decimals here would drag the
        # factor's fractional exponent in and silently quantize too fine.
        from math import log10

        factor_exp = round(log10(factor))
        coarse = Decimal(1).scaleb(max(-ref_precision, -precision + factor_exp))
        if (Decimal(str(value)).quantize(coarse, rounding=ROUND_HALF_EVEN)
                == Decimal(str(ref_value)).quantize(coarse, rounding=ROUND_HALF_EVEN)):
            return True, "exact", note
        ref_step = Decimal(1).scaleb(-ref_precision)
        if (abs(Decimal(str(value)).quantize(ref_step, rounding=ROUND_HALF_EVEN)
                - Decimal(str(ref_value))) <= Decimal(str(EVAL_ROUND_TOL))):
            return True, "round-tol", note
        rel = {"rate": EVAL_RATE_REL_TOL, "money": EVAL_MONEY_REL_TOL}.get(dimension)
        if rel is not None and (abs(Decimal(str(value)) - Decimal(str(ref_value)))
                                <= Decimal(str(rel)) * abs(Decimal(str(ref_value)))):
            return True, f"rel-{dimension}", note
        return False, "mismatch", note
    except InvalidOperation:
        return False, "overflow", note


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
                           "" if ok else "reference is N/A; the answer names a value anyway")
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
        ok, detail, note = _compare(
            ref_value, ref_unit, _meaningful_precision(ref_literal),
            value, unit, _meaningful_precision(literal))
        return Verdict(0, ok, 1.0 if ok else 0.0, detail,
                       note or ("" if ok else f"{value:,.6g}{unit} vs {ref_value:,.6g}{ref_unit}"))

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
