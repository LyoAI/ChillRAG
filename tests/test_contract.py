"""Consumer-contract tests for ``RAGTool``.

``RAGTool`` exists to be called by an existing pipeline without that pipeline
changing, so what matters is not what it does internally but that the five
places reading its output keep working. Each test below is one of those reads,
written as the consumer writes it — a change that breaks a consumer should fail
here first.

Corpora are built in memory and the dense searcher is faked, so these run
without documents, a model or an index.
"""

from __future__ import annotations

import pytest

from chillrag.corpus import CorpusManager
from chillrag.rag_tool import RAGTool, file_id_for, source_id_for
from chillrag.tools.searcher import Chunk, Hit

# ``corpus`` / ``FakeFaiss`` / ``_chunk`` live in conftest.py — shared with
# test_dropin.py, which checks the same surface from the caller's side.


# ---------------------------------------------------------------- identity --

def test_file_ids_are_ints_and_stable():
    """Consumers pass ids back as JSON integers, so they must be plain ints."""
    assert isinstance(file_id_for("报告A.docx"), int)
    assert file_id_for("报告A.docx") == file_id_for("报告A.docx")
    assert file_id_for("报告A.docx") != file_id_for("报告B.docx")
    assert isinstance(source_id_for("样例"), int)


def test_file_info_maps_every_document(corpus):
    """``file_info`` is the only enumeration source for the whole-document readback."""
    tool, _, _ = corpus
    assert tool.file_info == {
        "报告A.docx": file_id_for("报告A.docx"),
        "报告B.docx": file_id_for("报告B.docx"),
    }
    assert tool.token  # truthy: consumers pass it straight back in


# ----------------------------------------------------------- result shape ---

def test_result_has_exactly_the_vendor_fields(corpus):
    """The seven fields the vendor returns — no more.

    Extra convenience keys would silently change what consumer #2 renders, so
    the shape is pinned rather than treated as a minimum.
    """
    tool, _, _ = corpus
    results = tool.search_by_question("营业收入")
    assert results, "expected hits"
    assert set(results[0]) == {
        "confidence", "content_type", "metadata", "rank", "score", "segment", "source",
    }


def test_metadata_carries_the_label_prefix(corpus):
    """Consumers #1 and #4 split on ``文件名称:`` to recover the filename."""
    tool, _, _ = corpus
    meta = tool.search_by_question("营业收入")[0]["metadata"]
    assert "文件名称:" in meta
    assert meta.split("文件名称:")[-1] == "报告A"


def test_score_formats_as_fixed_point(corpus):
    """Two consumers render the score with ``f"{score:.4f}"``."""
    tool, _, _ = corpus
    assert f"{tool.search_by_question('营业收入')[0]['score']:.4f}"


def test_results_are_ranked_by_score_and_numbered(corpus):
    tool, _, _ = corpus
    results = tool.search_by_question("营业收入")
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    assert [r["rank"] for r in results] == [1, 2, 3]


def test_source_identifies_the_passage(corpus):
    """``source`` is ``<file_id>#<chunk index>``, matching ``get_segments``' index."""
    tool, _, _ = corpus
    doc_id, order = "报告A.docx", 1
    by_source = {r["source"]: r for r in tool.search_by_question("收入")}
    assert f"{file_id_for(doc_id)}#{order}" in by_source


# ------------------------------------------------------------- retrieval ----

def test_content_type_filter_is_a_union_in_priority_order(corpus):
    """Every requested type is returned, in the caller's priority order.

    The vendor honours only the first list element, which silently drops the
    rest; this is the documented departure.
    """
    tool, _, _ = corpus
    results = tool.search_by_question("收入", recall_content_types=["excel", "text"])
    assert [r["content_type"] for r in results] == ["excel", "text", "text"]


def test_content_type_filter_reports_the_true_type(corpus):
    """``content_type`` carries the passage's real type, so a filter is verifiable."""
    tool, _, _ = corpus
    only_excel = tool.search_by_question("收入", recall_content_types=["excel"])
    assert [r["content_type"] for r in only_excel] == ["excel"]


def test_a_filter_labels_hits_with_the_type_that_matched(corpus):
    """A passage with several forms is labelled by whichever the filter matched.

    The passage's own type list cannot answer "did my filter work?" — a caller
    asking for text would otherwise get hits labelled excel and have no way to
    tell that from a corpus carrying no type information at all.
    """
    tool, _, searcher = corpus
    mixed = Chunk(
        chunk_id="纪要C.docx#0", doc_id="纪要C.docx", order=0,
        text="表头\n| 项目 | 值 |", meta={"doc": "纪要C.docx", "content_types": ["excel", "text"]},
    )
    tool._manager.get("样例").store.add_chunk(mixed)
    searcher.hits = [
        Hit(score=0.9, chunk_id=mixed.chunk_id, text=mixed.text, meta=mixed.meta)
    ]
    assert [r["content_type"] for r in tool.search_by_question("表", recall_content_types=["text"])] == ["text"]
    assert [r["content_type"] for r in tool.search_by_question("表", recall_content_types=["excel"])] == ["excel"]
    # Unfiltered there is nothing to match, so the dominant form is reported.
    assert [r["content_type"] for r in tool.search_by_question("表")] == ["excel"]


def test_no_filter_returns_everything(corpus):
    tool, _, _ = corpus
    assert len(tool.search_by_question("收入")) == 3


def test_origin_ids_restrict_to_those_documents(corpus):
    """``origin_ids`` are ``file_info`` values and filter to those documents."""
    tool, _, _ = corpus
    results = tool.search_by_question(
        "收入", origin_ids=[file_id_for("报告B.docx")]
    )
    assert [r["content_type"] for r in results] == ["text"]
    assert all(r["metadata"].endswith("报告B") for r in results)


def test_unknown_origin_id_yields_no_hits(corpus):
    tool, _, _ = corpus
    assert tool.search_by_question("收入", origin_ids=[123456]) == []


def test_source_id_list_selects_a_known_corpus(corpus):
    tool, _, _ = corpus
    assert tool.search_by_question("收入", source_id_list=[source_id_for("样例")])


def test_unknown_source_id_falls_back_to_the_attached_corpus(corpus):
    """A stale id from a previous process degrades, rather than failing."""
    tool, _, _ = corpus
    assert tool.search_by_question("收入", source_id_list=[999999])


def test_search_without_a_corpus_is_an_error(tmp_path, monkeypatch):
    """No corpus anywhere means the caller has a bug, not an empty result."""
    manager = CorpusManager(materials_dir=tmp_path, cache_dir=tmp_path)
    monkeypatch.setattr(CorpusManager, "instance", classmethod(lambda cls: manager))
    with pytest.raises(ValueError, match="no corpus selected"):
        RAGTool().search_by_question("收入")


def test_result_count_is_capped(corpus, monkeypatch):
    """The cap is the interface (the vendor returns at most 10)."""
    tool, chunks, _ = corpus
    monkeypatch.setattr("chillrag.rag_tool.RECALL_TOP_K", 2)
    assert len(tool.search_by_question("收入")) == 2


# ------------------------------------------------------- whole-doc readback -

def test_get_segments_returns_the_whole_document_in_order(corpus):
    """The verifier rebuilds a document's full text from this."""
    tool, chunks, _ = corpus
    segments = tool.get_segments(file_id_for("报告A.docx"), tool.token)
    assert [s["segment"]["index"] for s in segments] == [0, 1]
    assert "".join(s["segment"]["content"] for s in segments) == (
        chunks[0].text + chunks[1].text
    )


def test_get_segments_accepts_the_token_positionally(corpus):
    """Called as ``get_segments(fid, rag_tool.token)``."""
    tool, _, _ = corpus
    assert tool.get_segments(file_id_for("报告A.docx"), tool.token)


def test_get_segments_of_an_unknown_file_is_empty(corpus):
    tool, _, _ = corpus
    assert tool.get_segments(123456, tool.token) == []


# ---------------------------------------------------------------- headers ---

def test_headers_shape(corpus):
    tool, _, _ = corpus
    headers = tool.get_file_headers()
    assert {h["name"] for h in headers} == {"报告A.docx", "报告B.docx"}
    assert all(isinstance(h["fileid"], int) for h in headers)
    assert all(h["content"] for h in headers)


def test_headers_before_upload_returns_none(tmp_path, monkeypatch):
    """The vendor returns ``None`` here and its caller checks for it."""
    manager = CorpusManager(materials_dir=tmp_path, cache_dir=tmp_path)
    monkeypatch.setattr(CorpusManager, "instance", classmethod(lambda cls: manager))
    assert RAGTool().get_file_headers() is None


# ---------------------------------------------------------------- answers ---

def test_answer_cache_distinguishes_instructions(corpus, monkeypatch):
    """A cached answer must never be served for a differently-specified ask."""
    tool, _, _ = corpus
    seen: list[str] = []

    def fake_author(self, bundle, question, instructions, output_instruction, lang):
        seen.append(instructions or "")
        return f"answer for {instructions!r}"

    monkeypatch.setattr(RAGTool, "_answer_with_agent", fake_author)
    first = tool.answer_by_question("问题", instructions="A")
    assert tool.answer_by_question("问题", instructions="A") == first
    assert len(seen) == 1, "identical ask should hit the cache"
    second = tool.answer_by_question("问题", instructions="B")
    assert second != first
    assert len(seen) == 2


def test_oneshot_sends_retrieved_passages_to_the_model(corpus, monkeypatch):
    """The bulk path: one completion over the retrieved passages, no tools."""
    tool, _, _ = corpus
    sent: list[list[dict]] = []

    class _Resp:
        finish_reason = "stop"
        content = "  依据材料的回答  "

    class _LLM:
        async def ainvoke(self, messages, **kwargs):
            sent.append(messages)
            return _Resp()

    monkeypatch.setattr("chillrag.llm.get_llm", lambda model, provider="openai": _LLM())
    monkeypatch.setattr("chillrag.rag_tool._model_name", lambda: "fake")
    answer = tool.answer_by_question("收入是多少", mode="oneshot")
    assert answer == "依据材料的回答"
    assert "营业收入为 1234 万元。" in sent[0][1]["content"]


def test_oneshot_with_no_hits_answers_nothing(corpus, monkeypatch):
    """Nothing retrieved: the caller gets an empty answer, not a fabrication."""
    tool, _, _ = corpus
    monkeypatch.setattr(RAGTool, "normal_rag_recall", lambda *a, **k: [])
    assert tool.answer_by_question("无关问题", mode="oneshot") == ""


def test_agent_path_passes_the_output_instruction_through(corpus, monkeypatch):
    """A caller's output contract reaches the user message."""
    tool, _, _ = corpus
    captured: dict = {}

    class _Agent:
        def __init__(self, llm, registry, **kwargs):
            captured.update(kwargs)

        async def arun(self, question):
            class _R:
                num_tool_calls = 0
                terminated = "answer"
                raw_answer = "{}"
                answer = "{}"
            return _R()

    monkeypatch.setattr("chillrag.agent.ReActAgent", _Agent)
    monkeypatch.setattr("chillrag.llm.get_llm", lambda model, provider="openai": object())
    monkeypatch.setattr("chillrag.rag_tool._model_name", lambda: "fake")
    assert tool.answer_by_question("问题", output_instruction="只输出 JSON") == "{}"
    assert captured["user_prompt"]("问题") == "问题\n\n只输出 JSON"
    assert captured["instructions"] is None


def test_the_configured_provider_reaches_the_model_constructor(corpus, monkeypatch):
    """``CHILLRAG_LLM_PROVIDER`` selects the YAML section that answers.

    Answering used to take ``get_llm``'s own default, so a config holding only a
    ``pingan:`` section raised ``ValueError: no 'openai' section`` the moment a
    question was asked — the deployment's gateway was unreachable by
    construction.
    """
    from chillrag import config

    tool, _, _ = corpus
    seen: list[tuple[str, str]] = []

    class _LLM:
        async def ainvoke(self, messages, **kwargs):
            class _Resp:
                finish_reason = "stop"
                content = "ok"

            return _Resp()

    def _record(model, provider):
        seen.append((model, provider))
        return _LLM()

    monkeypatch.setattr("chillrag.llm.get_llm", _record)
    monkeypatch.setattr("chillrag.rag_tool._model_name", lambda: "fake")
    monkeypatch.setattr(config, "LLM_PROVIDER", "pingan")

    assert tool.answer_by_question("收入是多少", mode="oneshot") == "ok"
    assert seen == [("fake", "pingan")]


def test_prompt_never_carries_a_benchmark_answer_contract():
    """The general layer must not impose an output format of its own."""
    from chillrag.agent.prompts import get_system_prompt

    for lang in ("zh", "en"):
        prompt = get_system_prompt(lang)
        assert "<answer>" not in prompt
        assert "<num>" not in prompt


# ------------------------------------------------------------- folder name --

@pytest.mark.parametrize("bad", ["", "  ", ".", "..", "a/b", "/abs/path", "C:\\x"])
def test_a_path_is_not_a_folder_name(tmp_path, monkeypatch, bad):
    """A path would place the cache inside the materials tree; refuse it."""
    manager = CorpusManager(materials_dir=tmp_path, cache_dir=tmp_path)
    monkeypatch.setattr(CorpusManager, "instance", classmethod(lambda cls: manager))
    with pytest.raises(ValueError, match="folder name"):
        RAGTool().upload_file_complete(bad)


def test_untagged_corpus_warns_when_a_type_filter_is_requested(corpus, caplog):
    """An untagged corpus can never match a filter; say so, don't just return [].

    A corpus prepared before content types were recorded looks identical to
    "no passage matches" from the caller's side, which is the wrong thing to
    conclude and impossible to notice without the warning.
    """
    tool, _, searcher = corpus
    searcher.hits = [
        Hit(score=0.9, chunk_id="报告A.docx#9", text="no tags here",
            meta={"doc": "报告A.docx"})
    ]
    with caplog.at_level("WARNING"):
        assert tool.search_by_question("收入", recall_content_types=["excel"]) == []
    assert "no content-type tags" in caplog.text
