"""``CorpusManager`` tests — the cache-invalidation decision.

Everything downstream of ``ensure`` is exercised in ``test_contract.py`` against
an in-memory corpus. What is tested here is the one thing that cannot be faked
there: whether a corpus that is already loaded still gets re-checked against its
source folder. Getting that wrong is silent — the caller sees an old corpus and
no error.

Ingestion itself is stubbed, so these run without documents, a model or MinerU.
"""

from __future__ import annotations

import pytest

from chillrag import corpus
from chillrag.corpus import CorpusBundle, CorpusManager
from chillrag.tools.searcher import DocumentStore


@pytest.fixture
def manager(tmp_path):
    """A manager over empty directories whose ingestion is stubbed out."""
    return CorpusManager(materials_dir=tmp_path, cache_dir=tmp_path)


def _load(manager, folder: str, signature: str) -> CorpusBundle:
    """Register a bundle the way a completed ``prepare`` would."""
    bundle = CorpusBundle(
        folder=folder,
        store=DocumentStore(name=folder),
        signature=signature,
        cache_dir=manager.cache_dir_for(folder),
    )
    manager._bundles[folder] = bundle
    return bundle


def test_ensure_reuses_a_loaded_bundle_whose_folder_is_unchanged(manager, monkeypatch):
    """The fast path: same fingerprint means the answer is already known."""
    bundle = _load(manager, "demo", "same")
    monkeypatch.setattr(CorpusManager, "signature", lambda self, folder: "same")

    def explode(self, folder, *, rebuild=False):  # pragma: no cover - must not run
        raise AssertionError("an unchanged folder must not be re-prepared")

    monkeypatch.setattr(CorpusManager, "prepare", explode)
    assert manager.ensure("demo") is bundle


def test_ensure_prepares_when_a_document_was_added(manager, monkeypatch):
    """A changed folder must reach ``prepare``, which is what ingests new documents.

    Serving the loaded bundle here would be invisible to the caller: it reports
    the documents it already had, and the new one never appears.
    """
    _load(manager, "demo", "before")
    monkeypatch.setattr(CorpusManager, "signature", lambda self, folder: "after")
    prepared: list[str] = []

    def fake_prepare(self, folder, *, rebuild=False):
        prepared.append(folder)
        return _load(self, folder, "after")

    monkeypatch.setattr(CorpusManager, "prepare", fake_prepare)
    assert manager.ensure("demo").signature == "after"
    assert prepared == ["demo"]


def test_ensure_prepares_an_unloaded_folder(manager, monkeypatch):
    """Nothing loaded and nothing to compare against: prepare decides."""
    monkeypatch.setattr(CorpusManager, "signature", lambda self, folder: "after")
    prepared: list[str] = []
    monkeypatch.setattr(
        CorpusManager, "prepare",
        lambda self, folder, *, rebuild=False: prepared.append(folder)
        or _load(self, folder, "after"),
    )
    manager.ensure("demo")
    assert prepared == ["demo"]


def test_index_builds_bm25_whatever_the_agent_binds(manager, monkeypatch):
    """Ingest builds both indexes; ``SEARCHER_TYPE`` only picks the agent's tools.

    Gating the lexical index on the agent's tool set leaves a corpus that looks
    prepared but has none — invisible until a query needs it, by which point
    re-ingesting the whole folder is the only fix. The cost is symmetric: a
    deployment that binds ``faiss`` alone still pays for a BM25 build it does not
    currently query.
    """
    built: list[str] = []
    monkeypatch.setattr(corpus, "get_embedder", lambda: object())
    monkeypatch.setattr(corpus, "embed_store", lambda *a, **k: None)
    monkeypatch.setattr(corpus, "BM25Searcher", lambda store, **kw: built.append(store.name))
    monkeypatch.setattr(corpus, "SEARCHER_TYPE", "faiss")

    manager._index(
        DocumentStore(name="demo"), manager.cache_dir_for("demo"), [], rebuild=True
    )
    assert built == ["demo"]


def test_ensure_indexes_builds_what_an_older_cache_lacks(manager, monkeypatch):
    """A store written before an index existed must acquire it.

    ``prepare`` returns early when no document was added, which is exactly the
    state such a cache is in — so without this the corpus would report itself up
    to date forever while having no lexical index to query.
    """
    built: list[str] = []
    monkeypatch.setattr(corpus, "get_embedder", lambda: object())
    monkeypatch.setattr(corpus, "embed_store", lambda *a, **k: None)
    monkeypatch.setattr(corpus, "BM25Searcher", lambda store, **kw: built.append(store.name))

    manager._ensure_indexes(DocumentStore(name="demo"), manager.cache_dir_for("demo"))
    assert built == ["demo"]


def test_ensure_indexes_leaves_a_complete_corpus_alone(manager, monkeypatch):
    """The check runs on every ``prepare``, so it must not rebuild a finished corpus."""
    cache = manager.cache_dir_for("demo")
    (cache / "faiss").mkdir(parents=True)
    # Both files: an index whose id list is missing counts as absent, because a
    # crash between the two writes leaves exactly that state.
    (cache / "faiss" / "index.faiss").write_bytes(b"")
    (cache / "faiss" / "vector_ids.json").write_text("[]", encoding="utf-8")
    # A committed Lucene index, not the empty shell a failed build leaves:
    # counting that shell as present is what made a failed build un-retryable.
    (cache / "bm25").mkdir()
    (cache / "bm25" / "segments_0").write_bytes(b"")
    rebuilt: list[str] = []
    monkeypatch.setattr(corpus, "get_embedder", lambda: object())
    monkeypatch.setattr(corpus, "embed_store", lambda *a, **k: rebuilt.append("faiss"))
    monkeypatch.setattr(corpus, "BM25Searcher", lambda store, **kw: rebuilt.append("bm25"))

    manager._ensure_indexes(DocumentStore(name="demo"), cache)
    assert rebuilt == []


def test_ensure_indexes_retries_a_failed_bm25_build(manager, monkeypatch):
    """An empty ``bm25/`` left by a failed build must not count as an index.

    ``BM25Searcher`` creates the target dir before building; when the build
    fails, that empty dir stayed behind, and treating it as "done" made the
    failure permanent until someone deleted the dir by hand.
    """
    built: list[str] = []
    monkeypatch.setattr(corpus, "get_embedder", lambda: object())
    monkeypatch.setattr(corpus, "embed_store", lambda *a, **k: None)
    monkeypatch.setattr(corpus, "BM25Searcher", lambda store, **kw: built.append(store.name))

    cache = manager.cache_dir_for("demo")
    (cache / "faiss").mkdir(parents=True)
    (cache / "faiss" / "index.faiss").write_bytes(b"")
    (cache / "faiss" / "vector_ids.json").write_text("[]", encoding="utf-8")
    (cache / "bm25").mkdir()  # exactly what a crashed build leaves behind

    manager._ensure_indexes(DocumentStore(name="demo"), cache)
    assert built == ["demo"]


def test_ensure_passes_rebuild_through_untouched(manager, monkeypatch):
    """``rebuild=True`` is the escape hatch for a same-second, same-size edit."""
    _load(manager, "demo", "same")
    monkeypatch.setattr(CorpusManager, "signature", lambda self, folder: "same")
    seen: list[bool] = []

    def fake_prepare(self, folder, *, rebuild=False):
        seen.append(rebuild)
        return _load(self, folder, "after")

    monkeypatch.setattr(CorpusManager, "prepare", fake_prepare)
    manager.ensure("demo", rebuild=True)
    assert seen == [True], "a rebuild must not be short-circuited by a matching fingerprint"
