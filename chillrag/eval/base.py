"""Evaluator framework: score the answers file a batch run produced.

An :class:`Evaluator` turns one answers row into one :class:`Verdict`. The
base class owns nothing but the contract — score a row, aggregate verdicts —
so a new evaluator is a subclass with ``score_row`` and nothing else. Rows
without a ``label`` (the reference answer) are *skipped*, never failed: mixed
batches stay scoreable, and the skip is reported, not buried.

Aggregates are plain data (``label -> number``); rendering stays in the CLI.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar


#: Strings a free-text answer or reference uses to say it found nothing.
_ABSTAIN = ("n/a", "na", "nan", "none", "null", "无", "未找到", "未提及",
            "不存在", "无法确定", "未知")


def abstains(text: str) -> bool:
    """Whether an answer or reference declares nothing found rather than a value.

    A reference that abstains flips the row's contract: the answer must
    abstain too, and any value it names counts as fabricated.
    """
    return not text.strip() or text.strip().lower() in _ABSTAIN


@dataclass
class Verdict:
    """One row's score, with the reason a human can read in the table."""

    index: int                    # the answers row's index, joining back to its trace
    ok: bool
    score: float = 1.0            # 0..1; partial credit is an evaluator's choice
    detail: str = ""              # short machine tag, e.g. "exact", "round-tol"
    note: str = ""                # human explanation, e.g. "3.51亿 ≈ 35142万"


class Evaluator(ABC):
    """Scores answers rows against their reference (``label``) values.

    Subclass and override :meth:`score_row`; override :meth:`aggregate` to add
    evaluator-specific statistics alongside the base accuracy.
    """

    #: Registry / CLI name.
    name: ClassVar[str]
    #: Answers-file columns: the model's answer, and the reference it must match.
    answer_field: ClassVar[str] = "answer"
    label_field: ClassVar[str] = "label"

    @abstractmethod
    def score_row(self, row: dict[str, Any]) -> Verdict:
        """Score one answers row. The row carries the model ``answer``, the
        ``label`` and any metadata the batch run passed through."""

    def aggregate(self, verdicts: list[Verdict]) -> dict[str, Any]:
        """Plain-number summary of a scored run; subclasses extend this."""
        n = len(verdicts)
        hits = sum(v.ok for v in verdicts)
        details: dict[str, int] = {}
        for v in verdicts:
            details[v.detail or "-"] = details.get(v.detail or "-", 0) + 1
        return {
            "scored": n,
            "accuracy": hits / n if n else 0.0,
            "correct": hits,
            "details": details,
        }


def evaluate(rows: list[dict[str, Any]], evaluator: Evaluator,
             ) -> tuple[list[Verdict], int]:
    """Score every row that names a reference; return verdicts and skip count.

    Verdicts keep input order, and each carries the row's own ``index`` when
    it has one, so results join back to the answers file and its traces.
    """
    verdicts: list[Verdict] = []
    skipped = 0
    for position, row in enumerate(rows):
        label = row.get(evaluator.label_field)
        if label is None or not str(label).strip():
            skipped += 1
            continue
        verdict = evaluator.score_row(row)
        try:
            verdict.index = int(row.get("index", position))
        except (TypeError, ValueError):
            verdict.index = position
        verdicts.append(verdict)
    return verdicts, skipped
