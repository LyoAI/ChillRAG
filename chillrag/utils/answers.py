"""Answers-file IO shared by the batch tooling — one row, one dict.

``read_answers`` parses a batch answers file (``.jsonl``, one object per line,
or ``.xlsx``, first sheet with a header row) into plain dicts.
``write_answers`` rewrites one in place, extension-driven and atomic: the full
content lands in a temp file first, then a single replace — a crash mid-write
never corrupts the artifact both the answers reader and the retry tool count
on.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def cell(value: Any) -> Any:
    """A value as an Excel cell — strings past Excel's cap get truncated."""
    if isinstance(value, str) and len(value) > 32_000:
        return value[:32_000] + " …"
    return value


def read_answers(path: Path) -> list[dict]:
    """Every answers record, in file order."""
    if path.suffix.lower() == ".xlsx":
        import openpyxl

        book = openpyxl.load_workbook(path, read_only=True, data_only=True)
        rows = list(book.active.iter_rows(values_only=True))
        if not rows:
            return []
        header = [str(c).strip() if c is not None else "" for c in rows[0]]
        return [dict(zip(header, row)) for row in rows[1:]
                if any(c is not None for c in row)]
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    raise SystemExit(f"expected a .jsonl or .xlsx answers file, got {path.name!r}")


def write_answers(path: Path, records: list[dict]) -> None:
    """Rewrite ``path`` with ``records`` in order; jsonl or xlsx by suffix.

    The header is the union of the records' keys, first-seen order, so columns
    a run gained mid-flight (``trace_id``, ``attempts``) keep their place.
    """
    tmp = path.with_name(path.name + ".tmp")
    if path.suffix.lower() == ".xlsx":
        import openpyxl

        header: list[str] = []
        for record in records:
            header += [k for k in record if k not in header]
        book = openpyxl.Workbook()
        sheet = book.active
        sheet.append(header)
        for record in records:
            sheet.append([cell(record.get(k)) for k in header])
        book.save(tmp)
    else:
        tmp.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in records) + "\n",
            encoding="utf-8")
    os.replace(tmp, path)
