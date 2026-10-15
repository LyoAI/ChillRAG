"""Ingest a materials folder from the command line.

    uv run python -m app.ingest 天津银行
    uv run python -m app.ingest 天津银行 年报 --rebuild
    uv run python -m app.ingest --all

Does what a web upload does — parse, chunk, embed, index — and then registers
the folder so the service and its HTTP clients can reach it by source id. Drop
the documents in ``materials/<folder>/`` first; the derived artifacts land in
``.cache/<folder>/``, which is where the service reads them from, so a folder
ingested here is not re-parsed when it is first queried.

The service must not be ingesting the same folder at the same time: the ingest
locks and the loaded embedder are per process, so two processes writing one
folder's cache can interleave. A second process also loads its own copy of the
embedding model and its own MinerU; on a small GPU, run this as

    EMBED_DEVICE=cpu MINERU_DEVICE_MODE=cpu uv run python -m app.ingest <folder>
"""

from __future__ import annotations

import argparse
import sys
from time import perf_counter

from app import registry
from chillrag.config import CACHE_DIR, MATERIALS_DIR
from chillrag.corpus import CorpusBundle, CorpusManager
from chillrag.rag_tool import RAGTool, file_id_for, source_id_for
from chillrag.utils import get_logger

#: What the ingest path can decide to redo is derived from the source files'
#: name, size and mtime — no change to the processing itself shows up there, so
#: a rebuild is the only way to pick one up. See the module docstring.
_REBUILD_NOTE = "discard the derived cache and redo the whole pipeline"


def _folders(args: argparse.Namespace) -> list[str]:
    """The folders to ingest, from the arguments or from ``--all``."""
    if args.folders and args.all:
        raise RuntimeError("give folder names or --all, not both")
    if args.folders:
        return list(args.folders)
    if not args.all:
        raise RuntimeError("name at least one folder, or pass --all")
    if not MATERIALS_DIR.is_dir():
        raise RuntimeError(f"no materials directory at {MATERIALS_DIR}")
    found = sorted(p.name for p in MATERIALS_DIR.iterdir() if p.is_dir() and not p.name.startswith("."))
    if not found:
        raise RuntimeError(f"no folders under {MATERIALS_DIR}")
    return found


def _ingest(folder: str, rebuild: bool) -> tuple[CorpusBundle, float]:
    """Ingest one folder. ``(bundle, seconds)``."""
    start = perf_counter()
    RAGTool().upload_file_complete(folder, rebuild=rebuild)
    # A cache hit: upload_file_complete has just loaded this same bundle.
    return CorpusManager.instance().get(folder), perf_counter() - start


def _warn_on_collisions(folder: str, source_id: int) -> None:
    """Report documents whose id already belongs to another folder.

    A file id is a hash of the file *name*, so ``报告.pdf`` in two folders is one
    id, and the later registration takes the earlier folder's record with it.
    Not detectable automatically without changing the id scheme, which would
    invalidate every id a client has cached — so say so instead.
    """
    manager = CorpusManager.instance()
    for path in manager.source_docs(folder):
        prior = registry.file_of(file_id_for(path.name))
        if prior and prior["source"] != source_id:
            print(
                f"warning: {path.name!r} shares a file id with "
                f"{registry.folder_of(prior['source'])!r}; the id now resolves to {folder!r}",
                file=sys.stderr,
            )


def _register(folder: str) -> int:
    """Register ``folder`` and its documents; return the source id."""
    names = [p.name for p in CorpusManager.instance().source_docs(folder)]
    source_id = source_id_for(folder)
    _warn_on_collisions(folder, source_id)
    registry.register_folder(folder, names)
    return source_id


def _report(folder: str, bundle: CorpusBundle, source_id: int, elapsed: float) -> None:
    docs = bundle.doc_ids()
    chunks = bundle.store.all_chunks()
    noun = "s" if len(docs) != 1 else ""
    print(f"{folder}  ({len(docs)} document{noun}, {len(chunks)} chunks, {elapsed:.1f}s)")
    # Names are not padded: padding counts characters, so a Chinese document
    # list would be ragged rather than aligned.
    print(f"  {'source_id':<9}  {source_id}")
    for doc_id in docs:
        print(f"  {doc_id}  ({len(bundle.chunks_of(doc_id))} chunks)")
    print(f"  {'materials':<9}  {MATERIALS_DIR / folder}")
    print(f"  {'cache':<9}  {CACHE_DIR / folder}")


def _run(args: argparse.Namespace) -> int:
    if args.quiet:
        # The chillrag logger is attached on import and writes INFO to stdout,
        # which would interleave with the summary below.
        get_logger("WARNING")

    folders = _folders(args)
    failed: list[str] = []
    documents = chunks = 0
    start = perf_counter()

    for folder in folders:
        try:
            bundle, elapsed = _ingest(folder, args.rebuild)
            source_id = _register(folder)
        except (OSError, ValueError) as exc:  # empty folder, path-shaped name, unreadable file
            print(f"error: {folder}: {exc}", file=sys.stderr)
            failed.append(folder)
            continue
        _report(folder, bundle, source_id, elapsed)
        documents += len(bundle.doc_ids())
        chunks += len(bundle.store.all_chunks())
        if len(folders) > 1:
            print()

    if len(folders) > 1:
        noun = "s" if len(folders) != 1 else ""
        print(f"ingested {len(folders) - len(failed)} folder{noun}, {documents} documents, "
              f"{chunks} chunks, {perf_counter() - start:.1f}s")
    # Failures are reported but do not stop the batch: one empty folder must not
    # abandon the work already done on the others.
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.ingest", description=__doc__.splitlines()[0])
    parser.add_argument("folders", nargs="*", help="folder names under the materials root")
    parser.add_argument("--all", action="store_true", help="every folder under the materials root")
    parser.add_argument("--rebuild", action="store_true", help=_REBUILD_NOTE)
    parser.add_argument("-q", "--quiet", action="store_true", help="only the summary, no ingest log")
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except (OSError, ValueError, RuntimeError) as exc:  # bad arguments, unusable tree
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
