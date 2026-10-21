"""BM25 tokenizer: word-level Chinese + case-insensitive English.

Unit tests cover ``tokenize`` directly; the functional tests build a real
Lucene index over a small zh/en corpus and check that both languages actually
hit (the failure mode they guard against is a query/index analyzer mismatch,
which silently returns zero hits).
"""

from __future__ import annotations

import pytest

from chillrag.tools.searcher import Chunk, DocumentStore
from chillrag.tools.searcher.tokenizer import ANALYZER_VERSION, tokenize, to_pretokenized

# jieba is a hard dependency of the tokenizer; pyserini boots a JVM and is
# only exercised by the functional tests at the bottom.
jieba = pytest.importorskip("jieba")
jieba.setLogLevel(60)


# ── tokenize ─────────────────────────────────────────────────────────────


def test_chinese_tokenized_into_words_not_characters():
    tokens = tokenize("应当建立数据安全管理制度")
    assert len(tokens) < len("应当建立数据安全管理制度")
    assert "数据安全" in tokens or "管理制度" in tokens


def test_english_lowercased_and_word_split():
    assert tokenize("Data Security Management") == ["data", "security", "management"]


def test_mixed_zh_en():
    tokens = tokenize("遵守ISO 27001标准")
    assert "iso" in tokens
    assert "27001" in tokens


def test_empty_and_whitespace():
    assert tokenize("") == []
    assert tokenize("   \n\t ") == []


def test_pretokenized_is_space_joined():
    assert to_pretokenized("Data Security") == "data security"
    assert " " in to_pretokenized("应当建立数据安全管理制度")


# ── end-to-end over a real Lucene index ──────────────────────────────────


@pytest.fixture
def bilingual_store(tmp_path):
    lucene = pytest.importorskip("pyserini.search.lucene")
    lucene.LuceneSearcher  # noqa: B018 — fail here if the JVM can't boot
    from chillrag.tools.searcher.bm25_search import BM25Searcher

    chunks = [
        ("zh", 0, "公司应当建立数据安全管理制度，明确数据分类分级。"),
        ("zh", 1, "定期开展网络安全漏洞扫描与渗透测试。"),
        ("en", 0, "The company shall establish a Data Security Management policy."),
        ("en", 1, "Annual business continuity plan review is required."),
    ]
    store = DocumentStore(name="bilingual")
    for doc_id, order, text in chunks:
        store.add_chunk(Chunk(
            chunk_id=f"{doc_id}#{order}", doc_id=doc_id, order=order,
            text=text, meta={"doc": doc_id, "page": 1, "section": None,
                             "content_types": ["text"]},
        ))
    return store, BM25Searcher(store, index_dir=tmp_path / "bm25")


def test_analyzer_version_recorded_and_bumped(bilingual_store, tmp_path):
    _store, searcher = bilingual_store
    marker = tmp_path / "bm25" / "analyzer_version.txt"
    assert marker.read_text(encoding="utf-8").strip() == ANALYZER_VERSION


def test_chinese_query_hits_chinese_chunk_word_level(bilingual_store):
    _store, searcher = bilingual_store
    hits = searcher.search("数据安全管理制度", k=4)
    assert hits and hits[0].chunk_id == "zh#0"


def test_english_query_case_insensitive(bilingual_store):
    _store, searcher = bilingual_store
    # Query casing differs from the chunk's ("Data Security Management").
    hits = searcher.search("data security management policy", k=4)
    assert hits and hits[0].chunk_id == "en#0"


def test_english_index_text_case_insensitive(bilingual_store):
    _store, searcher = bilingual_store
    # Even ALL-CAPS query terms find the mixed-case indexed chunk.
    hits = searcher.search("BUSINESS CONTINUITY", k=4)
    assert hits and hits[0].chunk_id == "en#1"


def test_cross_language_query_returns_nothing_spurious(bilingual_store):
    _store, searcher = bilingual_store
    ids = {h.chunk_id for h in searcher.search("数据安全管理制度", k=4)}
    assert "en#0" not in ids and "en#1" not in ids
