"""Batch driver: run a file of questions through the agent, one episode each.

Input is a JSONL file (one JSON object per line; a bare JSON string counts as
the question) or an Excel sheet (first sheet, header row). Only the question
field is required; every other field is optional metadata that resolves in
row → CLI-flag → default order:

=============== ===========  ============================================
field           flag         default
=============== ===========  ============================================
question        --question-field (defaults to ``question``)   required
folder          --folder     — error names the prepared corpora
user            --user       ``batch`` (must be a filesystem-safe name)
source          --source     ``batch``
session_id      — (row only) ``batch-<run>-<row>``, one per query
lang / mode / instructions / output_instruction  — row overrides flag
--------------- -----------  --------------------------------------------
any other field —            rides along: bound as episode context and
                           echoed into the output rows
=============== ===========  ============================================

There is no ``--session-id``: a session is a multi-turn conversation, and a
batch is not one. Every row is a single independent question, so it gets its
own session file; a row that names a ``session_id`` names its own file with
it, never merging rows into a shared conversation.

::

    python -m chillrag.batch_request questions.jsonl --folder 天津银行 -c 4
    python -m chillrag.batch_request queries.xlsx --question-field query --user kainan

Questions run through ``answer_by_question`` — one ReAct episode each, many
model calls apiece — behind a semaphore, so concurrency stays modest by
default. Each query traces as its own session, named after the run and the
row, so every episode gets its own file while a run's sessions stay together
in the directory. Results stream to ``<input>.answers.jsonl`` (or ``--out``,
``.xlsx`` honored) as they complete. Each answers row carries the
``trace_id`` of the episode that produced it, so a result joins back to its
transcript; a cache hit ran no episode and has no id. Corpora load from their
prepared cache — a batch run answers, never re-parses.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from chillrag.corpus import CorpusManager
from chillrag.rag_tool import RAGTool
from chillrag.tracing import episode, last_trace_id, new_session_id
from chillrag.utils import get_logger

logger = get_logger()
console = Console()

# Row fields that map onto answer_by_question kwargs; everything unrecognized
# is passthrough context.
CALL_FIELDS = ("instructions", "output_instruction", "lang", "mode")


@dataclass
class Job:
    index: int
    question: str
    folder: str
    context: dict[str, Any]      # episode binding: identity + passthrough fields
    call_kwargs: dict[str, Any]  # answer_by_question kwargs


@dataclass
class Outcome:
    index: int
    status: str                  # ok | empty | error
    answer: str
    duration_s: float
    error: str = ""
    trace_id: str | None = None  # None when no episode ran (cache hit) or it failed


# ------------------------------------------------------------------ input --

def _read_rows(path: Path) -> list[Any]:
    if path.suffix.lower() == ".xlsx":
        import openpyxl

        book = openpyxl.load_workbook(path, read_only=True, data_only=True)
        rows = list(book.active.iter_rows(values_only=True))
        if not rows:
            return []
        header = [str(cell).strip() if cell is not None else "" for cell in rows[0]]
        return [dict(zip(header, row)) for row in rows[1:]
                if any(cell is not None for cell in row)]

    if path.suffix.lower() == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append({"question": obj} if isinstance(obj, str) else obj)
        return rows

    raise SystemExit(f"unsupported input {path.name!r}: use .jsonl or .xlsx")


def _prepared_folders() -> list[str]:
    cache = CorpusManager.instance().cache_dir
    return sorted(p.name for p in cache.iterdir() if p.is_dir() and not p.name.startswith(".")
                  and CorpusManager.instance().is_prepared(p.name))


def _plan(rows: list[dict], args: argparse.Namespace) -> list[Job]:
    run_id = new_session_id()
    jobs: list[Job] = []
    for index, row in enumerate(rows):
        raw = row.get(args.question_field)
        question = "" if raw is None else str(raw).strip()
        if not question:
            raise SystemExit(f"row {index + 1}: no question in field {args.question_field!r}")
        folder = row.get("folder") or args.folder
        if not folder:
            prepared = ", ".join(_prepared_folders()) or "(none)"
            raise SystemExit(f"row {index + 1}: no folder — add a folder field or pass "
                             f"--folder (prepared: {prepared})")

        call_kwargs: dict[str, Any] = {}
        for name in CALL_FIELDS:
            value = row.get(name)
            if value in (None, ""):
                value = getattr(args, name)
            if value not in (None, ""):
                call_kwargs[name] = value

        # Only the question field is consumed; everything else — identity,
        # call fields and unknown columns — travels as episode context and is
        # echoed into the output. Excel cells may hold dates and other objects,
        # which would break the trace record's JSON, so non-primitives are
        # stringified at the boundary.
        context = {}
        for key, value in row.items():
            if key in (args.question_field, "folder") or value is None:
                continue
            context[key] = value if isinstance(value, (str, int, float, bool)) else str(value)
        context.setdefault("user", "batch")
        context["user"] = str(context["user"])
        # One session per query: a batch is a set of independent single-turn
        # questions, so each episode's trace is its own file. The run id in
        # the name keeps a run's sessions together in a listing.
        context.setdefault("session_id", f"batch-{run_id}-{index + 1:04d}")
        context["session_id"] = str(context["session_id"])
        context.setdefault("source", "batch")
        context["folder"] = str(folder)
        context.update(call_kwargs)
        jobs.append(Job(index, question, str(folder), context, call_kwargs))
    return jobs


# ----------------------------------------------------------------- engine --

def _load_tool(folders: set[str]) -> RAGTool:
    """One handle over every corpus the batch needs, from the prepared cache.

    ``get`` rather than ``ensure``: a batch run must answer, never re-parse —
    a drifted corpus is an ingest concern, out of scope here.
    """
    manager = CorpusManager.instance()
    tool = RAGTool()
    for folder in sorted(folders):
        try:
            bundle = manager.get(folder)
        except FileNotFoundError as exc:
            prepared = ", ".join(_prepared_folders()) or "(none)"
            raise SystemExit(f"corpus {folder!r} is not prepared — ingest it first "
                             f"(prepared: {prepared})") from exc
        tool._attach(bundle)
    return tool


def _answer(tool: RAGTool, job: Job) -> tuple[str, str | None]:
    """The thread body: bind the row's identity, then one agent episode.

    Returns the answer and the id of the trace that episode wrote, so the
    answers row can be joined back to its transcript.
    """
    with episode(**job.context):
        answer = tool.answer_by_question(job.question, **job.call_kwargs)
        return answer, last_trace_id()


async def run_batch(
    jobs: list[Job],
    tool: RAGTool,
    *,
    concurrency: int,
    on_outcome=None,
) -> list[Outcome]:
    """Run jobs through a semaphore-bounded task queue, preserving input order."""
    semaphore = asyncio.Semaphore(concurrency)

    async def one(job: Job) -> Outcome:
        async with semaphore:
            start = time.perf_counter()
            try:
                answer, trace_id = await asyncio.to_thread(_answer, tool, job)
            except Exception as exc:  # one bad question must not sink the batch
                outcome = Outcome(job.index, "error", "",
                                  time.perf_counter() - start,
                                  f"{type(exc).__name__}: {exc}")
            else:
                status = "ok" if answer.strip() else "empty"
                outcome = Outcome(job.index, status, answer,
                                  time.perf_counter() - start, trace_id=trace_id)
            logger.info("batch row %d: %s (%.1fs)", job.index + 1, outcome.status,
                        outcome.duration_s)
            if on_outcome:
                on_outcome(outcome)
            return outcome

    return list(await asyncio.gather(*(one(job) for job in jobs)))


# ----------------------------------------------------------------- output --

def _record(job: Job, outcome: Outcome) -> dict[str, Any]:
    record = {"index": job.index, **job.context, "question": job.question,
              "answer": outcome.answer, "status": outcome.status,
              "duration_s": round(outcome.duration_s, 3)}
    if outcome.trace_id:
        record["trace_id"] = outcome.trace_id
    if outcome.error:
        record["error"] = outcome.error
    return record


def _write_xlsx(path: Path, jobs: list[Job], outcomes: list[Outcome]) -> None:
    import openpyxl

    records = [_record(job, outcome) for job, outcome in zip(jobs, outcomes)]
    header: list[str] = []
    for record in records:
        header += [k for k in record if k not in header]
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append(header)
    for record in records:
        sheet.append([_cell(record.get(k)) for k in header])
    book.save(path)


def _cell(value: Any) -> Any:
    if isinstance(value, str) and len(value) > 32_000:  # Excel's cell limit
        return value[:32_000] + " …"
    return value


# -------------------------------------------------------------------- cli --

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="questions: .jsonl or .xlsx")
    parser.add_argument("--folder", help="corpus when a row has no folder field")
    parser.add_argument("--out", type=Path, help="results file (default: alongside input)")
    parser.add_argument("-c", "--concurrency", type=int, default=2,
                        help="parallel episodes; each one is many model calls (default 2)")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--user", help="identity when a row has no user field")
    parser.add_argument("--source", default="batch")
    parser.add_argument("--lang", help="prompt language when a row has none")
    parser.add_argument("--mode", choices=("agent", "oneshot"),
                        help="answer mode when a row has none")
    parser.add_argument("--instructions", help="system-prompt requirements when a row has none")
    parser.add_argument("--output-instruction",
                        help="per-question output contract when a row has none")
    parser.add_argument("--limit", type=int, help="run only the first N rows")
    args = parser.parse_args(argv)

    rows = _read_rows(args.input)
    if not rows:
        raise SystemExit(f"no rows in {args.input}")
    if args.limit:
        rows = rows[: args.limit]
    jobs = _plan(rows, args)

    out_path = args.out or args.input.with_name(args.input.stem + ".answers.jsonl")
    xlsx_out = out_path.suffix.lower() == ".xlsx"
    if not xlsx_out:
        out_file = out_path.open("w", encoding="utf-8")

    progress = Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                        BarColumn(), TextColumn("{task.completed}/{task.total}"),
                        TimeElapsedColumn(), console=console) if console.is_terminal else None
    task_id = None
    if progress:
        progress.start()
        task_id = progress.add_task("answering", total=len(jobs))

    def on_outcome(outcome: Outcome) -> None:
        if not xlsx_out:
            job = jobs[outcome.index]
            out_file.write(json.dumps(_record(job, outcome), ensure_ascii=False,
                                      default=str) + "\n")
            out_file.flush()
        if task_id is not None:
            # One tick per finished episode. Rows are slow (many model calls
            # each), so this is the only progress signal the run has.
            progress.advance(task_id)

    try:
        tool = _load_tool({job.folder for job in jobs})
        outcomes = asyncio.run(run_batch(jobs, tool, concurrency=args.concurrency,
                                         on_outcome=on_outcome))
    finally:
        if progress:
            progress.stop()
        if not xlsx_out:
            out_file.close()
    if xlsx_out:
        _write_xlsx(out_path, jobs, outcomes)

    counts = {status: sum(o.status == status for o in outcomes)
              for status in ("ok", "empty", "error")}
    summary = Table.grid(pad_edge=False)
    summary.add_column(style="dim", width=12)
    summary.add_column()
    summary.add_row("results", f"{counts['ok']} ok · {counts['empty']} empty · "
                               f"{counts['error']} error → {out_path}")
    seconds = sum(o.duration_s for o in outcomes)
    summary.add_row("usage", f"{seconds:.1f}s episode time across "
                             f"{args.concurrency} worker(s)")
    console.print(summary)
    for job, outcome in zip(jobs, outcomes):
        if outcome.status == "error":
            console.print(f"[yellow]row {job.index + 1} "
                          f"({job.question[:40]}…): {outcome.error}[/]")
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
