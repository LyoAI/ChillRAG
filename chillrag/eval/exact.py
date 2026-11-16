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
