"""Source-document dispatch — which front end parses which file.

A folder is not guaranteed to hold one kind of document, and the two parse
front ends are not interchangeable: MinerU rejects an unknown suffix outright,
so a markdown file reaching it takes the whole batch down rather than being
skipped. The dispatch is therefore exercised directly, with MinerU stubbed, plus
one markdown-only ingest end to end.
"""

from __future__ import annotations

import pytest

from chillrag import corpus
from chillrag.corpus import CorpusManager
from chillrag.tools.parser import parse_sources
from chillrag.tools.parser.sources import MARKDOWN_SUFFIXES, SOURCE_SUFFIXES

MD = "# 年报\n\n营业收入为 1234 万元。\n\n## 风险\n\n| 项目 | 2024 |\n| --- | --- |\n| 收入 | 5678 |\n"


def test_every_document_a_folder_may_hold_is_scanable():
    """A suffix the scan accepts but the dispatch cannot parse is a dead end."""
    assert MARKDOWN_SUFFIXES <= SOURCE_SUFFIXES
    assert {".pdf", ".docx", ".xlsx", ".pptx", ".png"} <= SOURCE_SUFFIXES


def test_markdown_never_reaches_mineru(tmp_path, monkeypatch):
    """MinerU sees only what it can parse; the batch is not failed by a ``.md``."""
    sent: list[list] = []
    monkeypatch.setattr(
        "chillrag.tools.parser.sources.parse_docs",
        lambda paths, **kw: sent.append([p.name for p in paths]) or {},
    )
    (tmp_path / "a.md").write_text(MD, encoding="utf-8")
    (tmp_path / "b.docx").write_bytes(b"x")

    parsed = parse_sources([tmp_path / "a.md", tmp_path / "b.docx"], out_dir=tmp_path)
    assert sent == [["b.docx"]]
    # The markdown still lands in the result, under its filename, like MinerU's.
    assert set(parsed) == {"a.md"}
    assert parsed["a.md"][0].type == "title"


def test_a_markdown_only_folder_ingests(tmp_path, monkeypatch):
    """The claim in the README, end to end: a folder of ``.md`` produces a corpus."""
    manager = CorpusManager(materials_dir=tmp_path / "materials", cache_dir=tmp_path / "cache")
    folder = manager.source_dir("样例")
    folder.mkdir(parents=True)
    (folder / "年报.md").write_text(MD, encoding="utf-8")

    monkeypatch.setattr(corpus, "get_embedder", lambda: object())
    monkeypatch.setattr(corpus, "embed_store", lambda *a, **k: None)
    monkeypatch.setattr(corpus, "BM25Searcher", lambda store, **kw: None)

    bundle = manager.prepare("样例")
    assert [d.name for d in manager.source_docs("样例")] == ["年报.md"]
    assert sorted(bundle.store.docs) == ["年报.md"]
    assert bundle.store.all_chunks()


def test_a_batch_of_foreign_suffixes_still_names_every_one(tmp_path):
    """The error lists what a folder may hold, so a stray file is actionable."""
    (tmp_path / "笔记.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError) as err:
        parse_sources([tmp_path / "笔记.txt"], out_dir=tmp_path)
    assert ".txt" in str(err.value)
