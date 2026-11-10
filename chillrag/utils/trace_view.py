"""Terminal viewer for agent episode traces (``<TRACE_DIR>/<user>/<session>.jsonl``).

Run it as a module::

    python -m chillrag.trace_view                 # overview: every user → session
    python -m chillrag.trace_view SESSION         # one session's episodes as a table
    python -m chillrag.trace_view SESSION TRACE   # one episode: scalars + transcript

``SESSION``/``TRACE`` may be unambiguous prefixes. ``--full`` disables message
truncation, for copy-pasting a whole transcript.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from rich.box import ROUNDED
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from chillrag.config import TRACE_DIR

console = Console()

ROLE_STYLES = {"system": "dim", "user": "cyan", "assistant": "green", "tool": "yellow"}
# Message bodies are clipped per role unless --full; tool results are the big ones.
_FULL_LIMIT = 10_000
_ROLE_LIMITS = {"system": 150, "user": 600, "assistant": 2_000, "tool": 200}


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


# ------------------------------------------------------------ formatting --

def _n(value: float) -> str:
    return f"{value:,}"


def _dur(seconds: float) -> str:
    return f"{seconds:.1f}s" if seconds >= 1 else f"{seconds * 1000:.0f}ms"


def _when(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).astimezone().strftime("%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return str(iso)


def _chain(record: dict) -> str:
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
    return " → ".join(name if count == 1 else f"{name}×{count}" for name, count in runs)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else f"{text[:limit]} … ({_n(len(text))} chars)"


# ----------------------------------------------------------------- views --

def _overview(files: list[Path]) -> None:
    users: dict[str, list[Path]] = {}
    for path in files:
        users.setdefault(path.parent.name, []).append(path)

    tree = Tree(f"[bold]traces[/]  {TRACE_DIR}")
    for user, paths in users.items():
        branch = tree.add(f"[bold cyan]{user}[/]")
        for path in paths:
            records = _read(path)
            tokens = sum(r.get("total_tokens") or 0 for r in records)
            seconds = sum(r.get("duration_s") or 0 for r in records)
            last = max((_when(r.get("ts", ""))[:11] for r in records), default="-")
            kb = path.stat().st_size / 1024
            branch.add(
                f"[bold]{path.stem}[/]  {len(records)} episodes · "
                f"{_n(tokens)} tokens · {seconds:.1f}s · {kb:.0f} KB · last {last}")
    console.print(tree)


def _table(path: Path) -> None:
    records = _read(path)
    table = Table(
        title=f"[bold]{path.parent.name}/{path.stem}[/]",
        caption=f"{len(records)} episode(s) · {path.stat().st_size / 1024:.0f} KB",
        box=ROUNDED, expand=True,
    )
    table.add_column("#", justify="right", style="dim")
    table.add_column("time", no_wrap=True)
    table.add_column("trace", no_wrap=True, style="dim")
    table.add_column("question", ratio=1, overflow="fold")
    table.add_column("tools", justify="right")
    table.add_column("tokens", justify="right")
    table.add_column("dur", justify="right", no_wrap=True)
    flagged = any(r.get("compacted") or r.get("terminated") != "answer" for r in records)
    if flagged:
        table.add_column(no_wrap=True)
    for i, r in enumerate(records, 1):
        terminated = r.get("terminated", "?")
        flags = " ".join(filter(None, (
            f"[yellow]{terminated}[/]" if flagged and terminated != "answer" else "",
            "✂" if r.get("compacted") else "",
        )))
        row = [
            str(i), _when(r.get("ts", ""))[6:], r.get("trace_id", "?")[:8],
            r.get("question", "?"), str(r.get("num_tool_calls", 0)),
            _n(r.get("total_tokens") or 0), _dur(r.get("duration_s") or 0),
        ]
        if flagged:
            row.append(flags)
        table.add_row(*row)
    console.print(table)


def _detail(record: dict, full: bool) -> None:
    limit = _FULL_LIMIT if full else lambda role: _ROLE_LIMITS.get(role, 200)
    names: dict[str, str] = {}

    def body(message: dict) -> Text:
        role = message.get("role", "?")
        cap = _FULL_LIMIT if full else _ROLE_LIMITS.get(role, 200)
        text = Text()
        if role == "assistant" and message.get("tool_calls"):
            for call in message["tool_calls"]:
                names[call.get("id", "")] = call["function"]["name"]
                text.append("→ ", style="dim")
                text.append(call["function"]["name"], style="yellow")
                args = call["function"].get("arguments", "")
                text.append(f" {_clip(args, 10_000 if full else 200)}\n", style="dim")
        content = message.get("content") or ""
        if content:
            if role == "tool":
                text.append(f"{names.get(message.get('tool_call_id'), '?')} · ", style="yellow")
            text.append(_clip(content, cap), style=ROLE_STYLES.get(role, ""))
        return text

    scalars = Table.grid(pad_edge=False, expand=True)
    scalars.add_column(width=10, style="dim")
    scalars.add_column(ratio=1)
    scalars.add_row("question", Text(record.get("question", "?")))
    scalars.add_row("answer", Text(_clip(record.get("answer", ""), _FULL_LIMIT if full else 5_000)))
    tools = (f"{_chain(record)}  ({record.get('num_tool_calls', 0)} calls)"
             if record.get("num_tool_calls") else "-")
    scalars.add_row("tools", Text(tools))
    scalars.add_row("usage", Text(
        f"{_n(record.get('total_tokens') or 0)} tokens · "
        f"{_dur(record.get('duration_s') or 0)} · terminated {record.get('terminated', '?')}"))
    scalars.add_row("session", Text(
        f"{record.get('user', '?')}/{record.get('session_id', '?')} · "
        f"source {record.get('source', '-')} · mode {record.get('mode', '-')} · "
        f"lang {record.get('lang', '-')}"))
    console.print(Panel(
        scalars, box=ROUNDED, expand=False,
        title=f"trace [bold]{record.get('trace_id', '?')[:8]}[/]",
        subtitle=_when(record.get("ts", "")), subtitle_align="right"))

    if not record.get("messages"):
        console.print("[yellow]✂ compacted — this episode's transcript was not retained[/]")
        return

    transcript = Table.grid(pad_edge=False, expand=True)
    transcript.add_column(width=10)
    transcript.add_column(ratio=1)
    for message in record["messages"]:
        role = message.get("role", "?")
        transcript.add_row(
            Text(role.ljust(9), style=ROLE_STYLES.get(role, "")), body(message))
    console.print(Group("[bold]transcript[/]", transcript))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session", nargs="?", help="session name (exact or prefix)")
    parser.add_argument("trace", nargs="?", help="trace_id, or its first characters")
    parser.add_argument("--full", action="store_true", help="do not truncate message contents")
    args = parser.parse_args(argv)

    files = _session_files(args.session)
    if not files:
        reason = f"no session matches {args.session!r}" if args.session else "no traces yet"
        console.print(f"{reason} under {TRACE_DIR}")
        return 1

    if args.trace is None:
        if args.session is None:
            _overview(files)
        else:
            for path in files:
                _table(path)
        return 0

    matches = [(path, r) for path in files for r in _read(path)
               if r.get("trace_id", "").startswith(args.trace)]
    if not matches:
        console.print(f"no trace id starts with {args.trace!r} in the matched session(s)")
        return 1
    if len(matches) > 1:
        console.print("trace prefix is ambiguous:")
        for path, r in matches:
            console.print(f"  {r['trace_id'][:8]}  {path.parent.name}/{path.stem}  "
                          f"{r.get('question', '?')[:40]}")
        return 1
    _detail(matches[0][1], args.full)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
