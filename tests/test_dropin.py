"""Can an existing caller switch to this without editing a line?

"Zero-cost migration" is a claim about *the caller's* code, so it is checked the
way the caller experiences it: which members it binds to, which call shapes it
writes, and which expressions it applies to the result. A rename, a dropped
member, a new required argument or an added result key all break a caller
somewhere, and each of those is asserted below.

The vendor surface transcribed here comes from measurements recorded in
``scripts/PROBE_FINDINGS.md`` — what the service actually returned — rather than
from its client source, because the measurements are what callers were built
against.

Signatures are checked by *binding the vendor's call shape*, not by equality:
extra optional parameters (``instructions``, ``mode``, ...) are additive and
leave every existing call site valid.
"""

from __future__ import annotations

import inspect

import pytest

from chillrag.rag_tool import RAGTool

#: Every member the vendor exposes that a caller anywhere binds to, with the
#: argument shapes callers actually write. Transcribed from the call sites in
#: the consuming pipeline.
VENDOR_CALLS: dict[str, tuple[tuple, dict]] = {
    "upload_file_complete": (("annual_reports",), {}),
    "get_file_headers": ((), {}),
    "add_missing_file": ((), {}),
    "search_by_question": (("营业收入是多少",), {}),
    "normal_rag_recall": (("营业收入是多少",), {}),
    "answer_by_question": (("营业收入是多少",), {}),
    "get_segments": ((123, "token"), {}),
}

#: The same members, called with every optional argument by keyword — the shape
#: a caller that uses filtering or scoping writes.
VENDOR_KEYWORD_CALLS: dict[str, tuple[tuple, dict]] = {
    "search_by_question": (
        ("营业收入是多少",),
        {"source_id_list": [1], "origin_ids": [2], "recall_content_types": ["excel", "text"]},
    ),
    "normal_rag_recall": (
        ("营业收入是多少",),
        {
            "source_id_list": [1],
            "origin_ids": [2],
            "context_trace_ids": "trace",
            "recall_content_types": ["excel", "text"],
        },
    ),
    "answer_by_question": (("营业收入是多少",), {"source_id_list": [1]}),
}

#: Attributes read directly off a handle. Callers treat these as part of the
#: interface, not as implementation detail.
VENDOR_ATTRIBUTES = ("file_info", "token", "foldername", "source_id")

#: The vendor's result shape, measured rather than documented. Anything missing
#: breaks a caller; anything added changes what a caller renders.
VENDOR_RESULT_FIELDS = frozenset(
    {"confidence", "content_type", "metadata", "rank", "score", "segment", "source"}
)


def test_constructor_takes_no_arguments():
    """Callers write ``RAGTool()`` — one per request, with no configuration."""
    inspect.signature(RAGTool).bind()


@pytest.mark.parametrize("member", sorted(VENDOR_CALLS))
def test_member_exists_and_accepts_the_vendor_call_shape(member):
    """A caller's line must keep working verbatim."""
    assert hasattr(RAGTool, member), f"{member} is part of the interface"
    args, kwargs = VENDOR_CALLS[member]
    inspect.signature(getattr(RAGTool, member)).bind(object(), *args, **kwargs)


@pytest.mark.parametrize("member", sorted(VENDOR_KEYWORD_CALLS))
def test_member_accepts_the_vendor_keyword_shape(member):
    """Optional arguments must keep their vendor names — callers pass them by keyword."""
    args, kwargs = VENDOR_KEYWORD_CALLS[member]
    inspect.signature(getattr(RAGTool, member)).bind(object(), *args, **kwargs)


def test_handle_has_the_attributes_callers_read():
    """``file_info`` and ``token`` are read straight off the handle."""
    handle = RAGTool()
    for name in VENDOR_ATTRIBUTES:
        assert hasattr(handle, name), f"{name} is read by a caller"


def test_result_carries_exactly_the_vendor_fields(corpus):
    """Pinned as a set, not a minimum: an extra key changes what a caller renders."""
    tool, _, _ = corpus
    assert set(tool.search_by_question("营业收入")[0]) == VENDOR_RESULT_FIELDS


# ------------------------------------------------------- consumer reads ----
#
# One entry per place in the consuming pipeline that reads a result, written as
# that place writes it. These are the expressions that must not have to change.


def _quant_tool_registry(seg: dict) -> tuple[str, str, str]:
    """``quant_tool_registry``: segment/metadata with fallbacks, score formatted."""
    return (
        seg.get("segment", seg.get("content", "")),
        seg.get("metadata", seg.get("file_name", "unknown")),
        f"{seg.get('score', 0.0):.4f}",
    )


def _rag_extractor(seg: dict) -> tuple[str, str, str]:
    """``rag_extractor``: reads ``file_name`` with no ``metadata`` fallback."""
    return (
        seg.get("segment", seg.get("content", "")),
        seg.get("file_name", "unknown"),
        f"{seg.get('score', 0.0):.4f}",
    )


def _core_retry(seg: dict) -> str:
    """``pipeline/core.py``: recovers the filename by splitting the label prefix."""
    meta = seg["metadata"]
    assert "文件名称:" in meta
    return meta.split("文件名称:")[-1]


CONSUMER_READS = {
    "quant_tool_registry": _quant_tool_registry,
    "rag_extractor": _rag_extractor,
    "core_retry": _core_retry,
}


@pytest.mark.parametrize("reader", sorted(CONSUMER_READS))
def test_consumer_read_works_on_a_real_result(corpus, reader):
    """Every consuming read pattern still produces a usable value."""
    tool, _, _ = corpus
    seg = tool.search_by_question("营业收入")[0]
    text, name, score = CONSUMER_READS[reader](seg)
    assert text, "a caller renders the passage text"
    assert name, "a caller renders the source name"
    assert score, "a caller renders the score"


def test_metadata_label_matches_the_measured_vendor_format(corpus):
    """``文件名称:<stem>``, half-width colon, extension stripped.

    Measured: ``文件名称:万科A：2025年年度报告``. Three retry paths split on this
    exact prefix.
    """
    tool, _, _ = corpus
    assert tool.search_by_question("营业收入")[0]["metadata"] == "文件名称:报告A"


def test_source_is_file_id_hash_index(corpus):
    """``<file_id>#<index>`` — the only passage identity the vendor returns."""
    tool, _, _ = corpus
    for seg in tool.search_by_question("营业收入"):
        file_id, _, index = seg["source"].partition("#")
        assert file_id.isdigit() and index.isdigit()
        assert int(file_id) in (tool.file_info or {}).values()


def test_get_segments_shape_matches_the_whole_document_readback(corpus):
    """``[{"segment": {"index", "content"}}]`` — the verifier's reconstruction."""
    tool, _, _ = corpus
    file_id = (tool.file_info or {})["报告A.docx"]
    segments = tool.get_segments(file_id, tool.token)
    assert segments
    for item in segments:
        assert set(item) == {"segment"}
        assert set(item["segment"]) == {"index", "content"}
        assert isinstance(item["segment"]["index"], int)
        assert isinstance(item["segment"]["content"], str)


def test_file_info_values_are_plain_ints(corpus):
    """Callers send these back as JSON integers and their prompts render them as numbers."""
    tool, _, _ = corpus
    for name, file_id in (tool.file_info or {}).items():
        assert isinstance(file_id, int) and not isinstance(file_id, bool), name


def test_constructing_a_handle_is_free_of_side_effects(tmp_path, monkeypatch):
    """The consumer builds one per request; construction must not touch the disk."""
    from chillrag.corpus import CorpusManager

    manager = CorpusManager(materials_dir=tmp_path, cache_dir=tmp_path)
    monkeypatch.setattr(CorpusManager, "instance", classmethod(lambda cls: manager))
    monkeypatch.setattr(
        CorpusManager, "get", lambda self, folder, **kw: pytest.fail("construction read the corpus")
    )
    handle = RAGTool()
    assert handle.foldername is None and handle.file_info is None
