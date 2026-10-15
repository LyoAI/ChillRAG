"""The ingest CLI: what it writes, what it prints, and what it refuses.

Runs against a corpus manager and a registry of our own, with a fake tool that
stands in for parsing and embedding — the same bargain the rest of the suite
makes. What is under test is the CLI's contract with the two people who use it:
the developer reading the summary, and the service reading the registry.
"""

import logging

import pytest

from app import ingest, registry
from chillrag.corpus import CorpusBundle, CorpusManager
from chillrag.rag_tool import _folder_name, file_id_for, source_id_for
from chillrag.tools.searcher import Chunk, DocumentStore

MD = "营业收入为 1234 万元。\n"


class FakeTool:
    """Mimics the parts of ``upload_file_complete`` the CLI depends on."""

    calls: list = []

    def __init__(self) -> None:
        self._manager = CorpusManager.instance()

    def upload_file_complete(self, folder, *, rebuild=False):
        # The name check lives in the tool, not the CLI, so the fake reuses the
        # real one rather than restating the rule and drifting from it.
        folder = _folder_name(folder)
        FakeTool.calls.append((folder, rebuild))
        paths = self._manager.source_docs(folder)
        if not paths:
            raise FileNotFoundError(f"{folder!r} holds no documents to ingest")
        store = DocumentStore(name=folder)
        for path in paths:
            store.add_chunk(Chunk(
                chunk_id=f"{path.name}#0", doc_id=path.name, order=0, text=MD,
                meta={"doc": path.name, "page": 1, "section": None, "content_types": ["text"]},
            ))
        self._manager._bundles[folder] = CorpusBundle(
            folder=folder, store=store, signature=self._manager.signature(folder),
            cache_dir=self._manager.cache_dir_for(folder),
        )
        return None


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A materials tree, a cache tree and a registry, none of them the real ones."""
    materials = tmp_path / "materials"
    cache = tmp_path / "cache"
    materials.mkdir()
    cache.mkdir()
    monkeypatch.setattr(ingest, "MATERIALS_DIR", materials)
    monkeypatch.setattr(ingest, "CACHE_DIR", cache)
    monkeypatch.setattr(registry, "_PATH", cache / "service_registry.json")
    monkeypatch.setattr(ingest, "RAGTool", FakeTool)

    manager = CorpusManager(materials_dir=materials, cache_dir=cache)
    monkeypatch.setattr(CorpusManager, "instance", classmethod(lambda cls: manager))

    # ``-q`` and the default both set the shared logger's level, which outlives
    # the test.
    level = logging.getLogger("chillrag").level
    FakeTool.calls = []
    yield materials
    logging.getLogger("chillrag").setLevel(level)


def folder(materials, name: str, *files: str):
    directory = materials / name
    directory.mkdir(parents=True, exist_ok=True)
    for file in files:
        (directory / file).write_text(MD, encoding="utf-8")
    return directory


def run(*argv: str) -> int:
    return ingest.main(list(argv))


# ------------------------------------------------------------- the happy path --

def test_reports_the_corpus_and_registers_it(workspace, capsys):
    folder(workspace, "bank-a", "a.pdf", "b.docx")
    assert run("bank-a") == 0
    out = capsys.readouterr().out

    assert "bank-a" in out
    assert "2 documents" in out and "2 chunks" in out
    assert "a.pdf" in out and "b.docx" in out
    # The two facts the caller cannot get anywhere else.
    assert str(source_id_for("bank-a")) in out
    assert str(workspace / "bank-a") in out  # materials
    assert "/cache/bank-a" in out  # derived artifacts, where the service reads them


def test_registers_the_source_and_every_document(workspace):
    folder(workspace, "bank-a", "a.pdf", "b.docx")
    assert run("bank-a") == 0
    source_id = source_id_for("bank-a")

    assert registry.folder_of(source_id) == "bank-a"
    assert registry.sources() == [(source_id, "bank-a")]
    for name in ("a.pdf", "b.docx"):
        assert registry.file_of(file_id_for(name)) == {"source": source_id, "name": name}


def test_a_registered_folder_is_what_the_service_lists(workspace):
    """The point of registering: the corpus is reachable by source id afterwards."""
    folder(workspace, "bank-a", "a.pdf")
    run("bank-a")
    assert source_id_for("bank-a") in [sid for sid, _ in registry.sources()]


def test_second_run_is_incremental_and_reports_the_same_corpus(workspace, capsys):
    folder(workspace, "bank-a", "a.pdf")
    run("bank-a")
    assert run("bank-a") == 0
    assert "1 document, 1 chunks" in capsys.readouterr().out


def test_rebuild_is_forwarded_to_the_tool(workspace):
    folder(workspace, "bank-a", "a.pdf")
    run("bank-a", "--rebuild")
    assert FakeTool.calls == [("bank-a", True)]
    run("bank-a")
    assert FakeTool.calls[-1] == ("bank-a", False)


def test_several_folders_are_summarised_in_one_line(workspace, capsys):
    folder(workspace, "a", "x.pdf")
    folder(workspace, "b", "y.pdf", "z.pdf")
    assert run("a", "b") == 0
    out = capsys.readouterr().out
    assert "ingested 2 folders, 3 documents, 3 chunks" in out


# ----------------------------------------------------------------- refusals --

def test_an_empty_folder_is_one_error_line_and_no_registration(workspace, capsys):
    folder(workspace, "empty")
    assert run("empty") == 1
    captured = capsys.readouterr()
    assert captured.out == ""  # nothing to report
    assert "error: empty:" in captured.err
    assert registry.sources() == []


def test_a_missing_folder_is_an_error(workspace, capsys):
    assert run("nope") == 1
    assert "error: nope:" in capsys.readouterr().err
    assert registry.sources() == []


def test_a_path_shaped_name_is_refused(workspace, capsys):
    assert run("../secret") == 1
    assert "foldername must be a folder name" in capsys.readouterr().err
    assert registry.sources() == []


def test_a_registration_is_only_written_after_a_successful_ingest(workspace):
    folder(workspace, "good", "a.pdf")
    folder(workspace, "bad")
    assert run("good", "bad") == 1
    assert [name for _, name in registry.sources()] == ["good"]


# -------------------------------------------------------------------- --all --

def test_all_walks_every_folder_and_survives_one_failure(workspace, capsys):
    folder(workspace, "a", "x.pdf")
    folder(workspace, "b")  # empty: must not abandon the batch
    folder(workspace, "c", "y.pdf")
    (workspace / ".hidden").mkdir()
    (workspace / "loose.txt").write_text(MD, encoding="utf-8")  # not a folder

    assert run("--all") == 1
    captured = capsys.readouterr()
    assert sorted(name for _, name in registry.sources()) == ["a", "c"]
    assert "error: b:" in captured.err
    assert "ingested 2 folders, 2 documents" in captured.out
    assert "hidden" not in captured.out and "loose" not in captured.out


def test_all_without_a_materials_tree_is_an_error(workspace, monkeypatch, capsys):
    monkeypatch.setattr(ingest, "MATERIALS_DIR", workspace / "gone")
    assert run("--all") == 1
    assert "no materials directory" in capsys.readouterr().err


def test_all_and_a_name_together_is_refused(workspace, capsys):
    folder(workspace, "a", "x.pdf")
    assert run("a", "--all") == 1
    assert "not both" in capsys.readouterr().err


def test_no_arguments_says_what_to_do(workspace, capsys):
    assert run() == 1
    assert "name at least one folder, or pass --all" in capsys.readouterr().err


# -------------------------------------------------------------------- quiet --

def test_quiet_raises_the_log_level_so_the_summary_stands_alone(workspace):
    folder(workspace, "a", "x.pdf")
    assert run("a", "-q") == 0
    assert logging.getLogger("chillrag").level == logging.WARNING


def test_the_ingest_log_is_visible_by_default(workspace):
    folder(workspace, "a", "x.pdf")
    assert run("a") == 0
    assert logging.getLogger("chillrag").level <= logging.INFO


# ------------------------------------------------------------------ warning --

def test_a_name_shared_with_another_folder_is_reported(workspace, capsys):
    """A file id hashes the name alone, so two folders cannot both own one."""
    folder(workspace, "a", "报告.pdf")
    folder(workspace, "b", "报告.pdf")
    run("a")
    capsys.readouterr()
    assert run("b") == 0
    assert "shares a file id" in capsys.readouterr().err
    # The later registration wins, which is what the warning is about.
    assert registry.file_of(file_id_for("报告.pdf"))["source"] == source_id_for("b")
