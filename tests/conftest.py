"""Shared fixtures: an in-memory corpus and a faked dense searcher.

Every test in this suite runs without documents, a model or an index — the
corpus is built in memory and the one backend that would need an embedding
model is replaced. What is under test is the interface, not retrieval quality.
"""

from __future__ import annotations

import pytest

from chillrag.corpus import CorpusBundle, CorpusManager
from chillrag.rag_tool import RAGTool
from chillrag.tools.searcher import BaseSearcher, Chunk, DocumentStore, Hit


class FakeFaiss(BaseSearcher):
    """A dense searcher returning preset hits, recording the query and k."""

    name = "faiss"
    description = "fake dense retrieval"

    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, k: int) -> list[Hit]:
        self.calls.append((query, k))
        return self.hits[:k]


def _chunk(doc_id: str, order: int, text: str, types: list[str]) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}#{order}",
        doc_id=doc_id,
        order=order,
        text=text,
        meta={"doc": doc_id, "page": 1, "section": None, "content_types": types},
    )


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """A two-document corpus whose dense hits are fixed, wired into one tool.

    Returns ``(tool, chunks, searcher)``; the searcher's hit order is the
    score order, so tests can control ranking without an embedding model.
    """
    chunks = [
        _chunk("报告A.docx", 0, "营业收入为 1234 万元。", ["text"]),
        _chunk("报告A.docx", 1, "| 指标 | 2024 |\n| 收入 | 5678 |", ["excel"]),
        _chunk("报告B.docx", 0, "风险提示：市场波动可能影响收益。", ["text"]),
    ]
    store = DocumentStore(name="样例")
    for c in chunks:
        store.add_chunk(c)

    manager = CorpusManager(materials_dir=tmp_path, cache_dir=tmp_path)
    monkeypatch.setattr(
        CorpusManager, "instance", classmethod(lambda cls: manager)
    )
    searcher = FakeFaiss([Hit(score=s, chunk_id=c.chunk_id, text=c.text, meta=c.meta)
                          for s, c in zip((0.9, 0.7, 0.5), chunks)])
    bundle = CorpusBundle(
        folder="样例", store=store, signature=manager.signature("样例"),
        cache_dir=tmp_path,
    )
    manager._bundles["样例"] = bundle
    monkeypatch.setattr(manager, "searcher", lambda folder, kind=None: searcher)

    # The corpus is already loaded, so this must not touch the source folder.
    tool = RAGTool()
    tool.upload_file_complete("样例")
    assert tool.file_info
    return tool, chunks, searcher
