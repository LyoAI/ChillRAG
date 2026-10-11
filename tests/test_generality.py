"""Generality tests — the layer must not assume a subject matter or a caller.

This package is a general document retriever. A quantitative pipeline is one
caller of it; a qualitative one is planned. What keeps that possible is
negative: nothing domain-specific, and no output contract, may be baked in.

Two kinds of test:

- **Source scan** — the cheap, exhaustive one. Vocabulary is copied by habit,
  and a stray example in a tool description is a real behavioural change: the
  model reads it. So the scan covers prose and docstrings, not just code.
- **Behaviour** — a corpus of plain, non-financial documents must go through
  ingest → retrieve → answer with nothing financial anywhere in the request.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import chillrag

PACKAGE = Path(chillrag.__file__).parent

#: Vocabulary that belongs to a caller or to a specific deployment, never here.
#: Each entry is a regex matched case-insensitively with word boundaries where
#: the term is a word.
FORBIDDEN = {
    # Benchmark / evaluation concepts — a separate concern from retrieval.
    r"\bfinsearch\b": "benchmark package name",
    r"\bbenchmark\b": "evaluation concept",
    r"\bqid\b": "benchmark question id",
    r"\bgold[_ ]": "benchmark annotation",
    r"\brun_eval\b": "benchmark scoring entry point",
    r"<answer>|<num>|<unit>": "benchmark answer contract",
    r"\bBrowseComp\b": "the benchmark this stack was written for",
    r"\bOUTLINE\b": "benchmark design document",
    # Financial subject matter.
    r"不良贷款|营业收入|净利润|年报|财报|财务": "financial subject matter",
    r"\bindicator\b|\bmetric\b": "financial-indicator framing",
    r"\bcompany\b|\bcompanies\b": "vendor's word for a corpus is folder",
    # Caller policy that must be passed in per call, not defaulted here.
    r"表格为主|table[- ]preference": "quantitative retrieval policy",
}

#: Files allowed to break the rule, with the reason. Keep this empty if at all
#: possible — an exception here is a leak the scan can no longer catch.
ALLOWED: dict[str, str] = {}


def _sources() -> list[Path]:
    return sorted(p for p in PACKAGE.rglob("*.py") if "__pycache__" not in p.parts)


def test_scan_finds_the_package():
    """Guard the guard: an empty or misplaced scan would pass vacuously."""
    files = _sources()
    assert len(files) > 10, f"scan found only {len(files)} modules under {PACKAGE}"
    assert any(p.name == "rag_tool.py" for p in files)


@pytest.mark.parametrize("pattern", sorted(FORBIDDEN))
def test_no_foreign_vocabulary(pattern: str):
    """No module may mention concepts from another layer."""
    rx = re.compile(pattern, re.IGNORECASE)
    offenders = []
    for path in _sources():
        rel = str(path.relative_to(PACKAGE))
        if rel in ALLOWED:
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if rx.search(line):
                offenders.append(f"{rel}:{n}: {line.strip()[:100]}")
    assert not offenders, (
        f"{FORBIDDEN[pattern]!r} leaked into the general layer "
        f"(/{pattern}/):\n" + "\n".join(offenders)
    )


def test_public_api_is_generic():
    """The names a caller sees describe documents, not a subject matter."""
    assert set(chillrag.__all__) == {"RAGTool", "CorpusManager"}
    for name in chillrag.__all__:
        assert re.fullmatch(r"[A-Za-z]+", name)


# ------------------------------------------------------------- behaviour ----

@pytest.fixture
def plain_corpus(tmp_path, monkeypatch):
    """A small corpus of ordinary prose — a manual, not a financial report."""
    from chillrag.corpus import CorpusBundle, CorpusManager
    from chillrag.rag_tool import RAGTool
    from chillrag.tools.searcher import BaseSearcher, Chunk, DocumentStore, Hit

    class _FakeFaiss(BaseSearcher):
        name = "faiss"
        description = "fake"

        def search(self, query, k):
            return [
                Hit(score=0.9 - i / 10, chunk_id=c.chunk_id, text=c.text, meta=c.meta)
                for i, c in enumerate(store.all_chunks())
            ]

    store = DocumentStore(name="手册")
    store.add_chunks([
        Chunk(
            chunk_id="手册.md#0", doc_id="手册.md", order=0,
            text="## 安装\n\n将设备放置在通风处，接通电源后按下启动键。",
            meta={"doc": "手册.md", "content_types": ["text"], "native_types": ["paragraph"]},
        ),
        Chunk(
            chunk_id="手册.md#1", doc_id="手册.md", order=1,
            text="| 部件 | 数量 |\n| 螺丝 | 4 |",
            meta={"doc": "手册.md", "content_types": ["excel"], "native_types": ["table"]},
        ),
    ])
    manager = CorpusManager(materials_dir=tmp_path, cache_dir=tmp_path)
    monkeypatch.setattr(CorpusManager, "instance", classmethod(lambda cls: manager))
    manager._bundles["手册"] = CorpusBundle(
        folder="手册", store=store, signature=manager.signature("手册"), cache_dir=tmp_path,
    )
    monkeypatch.setattr(manager, "searcher", lambda folder, kind=None: _FakeFaiss())
    tool = RAGTool()
    tool.upload_file_complete("手册")
    return tool


def test_non_financial_corpus_retrieves(plain_corpus):
    """A manual is a corpus like any other."""
    results = plain_corpus.search_by_question("怎么开机")
    assert results and results[0]["segment"].startswith("## 安装")
    assert results[0]["metadata"] == "文件名称:手册"


def test_non_financial_corpus_filters_by_content_type(plain_corpus):
    """Type filtering is a document-shape concern, valid in any domain."""
    results = plain_corpus.search_by_question("部件数量", recall_content_types=["excel"])
    assert [r["content_type"] for r in results] == ["excel"]


def test_answer_defaults_to_the_callers_own_format(plain_corpus, monkeypatch):
    """A caller asking for JSON gets JSON-shaped instructions, not a default."""
    captured: dict = {}

    class _Agent:
        def __init__(self, llm, registry, **kwargs):
            captured.update(kwargs)

        async def arun(self, question):
            class _R:
                num_tool_calls, terminated, answer = 0, "answer", "ok"
            return _R()

    monkeypatch.setattr("chillrag.agent.ReActAgent", _Agent)
    monkeypatch.setattr("chillrag.llm.get_llm", lambda model, provider="openai": object())
    monkeypatch.setattr("chillrag.rag_tool._model_name", lambda: "fake")

    contract = '只输出 JSON：{"sections": ["..."]}'
    assert plain_corpus.answer_by_question("有哪些章节", output_instruction=contract) == "ok"
    # Reaches the model verbatim, and the general layer adds no format of its own.
    assert captured["user_prompt"]("有哪些章节").endswith(contract)
    assert captured["instructions"] is None


def test_system_prompt_carries_caller_instructions(monkeypatch):
    """`instructions` is the only channel for caller-specific requirements."""
    from chillrag.agent.prompts import get_system_prompt

    base = get_system_prompt("en")
    extended = get_system_prompt("en", instructions="Always cite the page number.")
    assert "Always cite the page number." in extended
    assert len(extended) > len(base)
