"""Answer evaluation for batch runs — score an answers file against the
reference values its rows carry (the ``label`` column, passed through by
``batch_request``).

The framework is one contract (:class:`Evaluator.score_row` -> Verdict) with
aggregation and skipping in the base; each concrete evaluator is a subclass.
New evaluators register here and become CLI choices automatically.
"""

from chillrag.eval.base import Evaluator, Verdict, evaluate
from chillrag.eval.exact import Exact
from chillrag.eval.fin_numeric import FinNumeric

EVALUATORS: dict[str, type[Evaluator]] = {cls.name: cls for cls in (FinNumeric, Exact)}

__all__ = ["Evaluator", "Verdict", "evaluate", "FinNumeric", "Exact", "EVALUATORS"]
