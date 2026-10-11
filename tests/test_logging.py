"""Logging contract: every failure, degradation and costly step leaves a trace.

Some callers swallow exceptions (one calls ``upload_file_complete`` with no
try/except at all), so the log is sometimes the only evidence that something
went wrong. Each test below pins one representative of a level: ``error``
before a raise, ``warning`` on a degraded-but-usable path, ``info`` when an
expensive step completes.

The package logger is namespaced and does not propagate, so ``caplog`` never
sees its records; a handler is attached to it directly instead.
"""

from __future__ import annotations

import logging

import pytest

from chillrag.corpus import CorpusManager
from chillrag.rag_tool import RAGTool
from chillrag.tools import ToolRegistry
from chillrag.tools.searcher import DocumentStore
from chillrag.utils import get_logger


@pytest.fixture
def records():
    """Captured ``(level, message)`` pairs from the package logger."""
    captured: list[tuple[int, str]] = []

    class _Sink(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append((record.levelno, record.getMessage()))

    handler = _Sink()
    log = get_logger()
    log.addHandler(handler)
    try:
        yield captured
    finally:
        log.removeHandler(handler)


def _logged(records, level: int, fragment: str = "") -> bool:
    return any(
        got >= level and fragment in message for got, message in records
    )


# ------------------------------------------------------------------ error --

def test_a_path_shaped_foldername_is_logged_before_it_raises(records):
    """The raise is the caller's problem; the reason is the operator's."""
    with pytest.raises(ValueError):
        RAGTool().upload_file_complete("../secret")
    assert _logged(records, logging.ERROR, "not a folder name")


def test_an_empty_folder_is_logged_before_it_raises(records, tmp_path):
    manager = CorpusManager(materials_dir=tmp_path, cache_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        manager.prepare("demo")
    assert _logged(records, logging.ERROR, "no ingestable documents")


def test_an_unknown_tool_is_logged_before_it_raises(records):
    registry = ToolRegistry({}, DocumentStore(name="demo"))
    with pytest.raises(KeyError):
        registry.execute("nope", {})
    assert _logged(records, logging.ERROR, "unknown tool")


# ---------------------------------------------------------------- warning --

def test_a_read_without_a_corpus_is_logged_as_a_warning(records):
    """``None`` is a legal answer here, so nothing else marks the misuse."""
    assert RAGTool().get_file_headers() is None
    assert _logged(records, logging.WARNING, "no corpus attached")


def test_an_unknown_document_id_is_logged_as_a_warning(records, corpus):
    tool, _, _ = corpus
    records.clear()
    assert tool.get_segments(999) == []
    assert _logged(records, logging.WARNING, "not a document")


# ------------------------------------------------------------------- info --

def test_an_upload_is_logged_as_info(records, corpus):
    tool, _, _ = corpus
    records.clear()
    tool.upload_file_complete("样例")
    assert _logged(records, logging.INFO, "ready")


def test_a_recall_logs_its_filters(records, corpus):
    """The filter is a call argument, so the log has to carry it: a result set
    that came back short is otherwise indistinguishable from a filter that
    silently matched nothing."""
    tool, _, _ = corpus
    records.clear()
    tool.search_by_question("营业收入", recall_content_types=["excel"])
    assert _logged(records, logging.INFO, "types=['excel']")


# -------------------------------------------------------------- secrets ----

def test_provider_credentials_never_reach_the_log(records):
    """The gateway provider used to log app_key and app_secret verbatim."""
    from chillrag.llm.provider.pingan import PingAnLLMProvider

    PingAnLLMProvider(
        api_key="token",
        api_base="http://localhost:8000/v1",
        default_model="demo",
        extra_headers={
            "appKey": "AK-DEADBEEF",
            "appSecret": "AS-DEADBEEF",
            "rsaPrivateKey": "DEADBEEF",
            "openApiCredential": "CRED-DEADBEEF",
            "sceneId": "7",
            "requestId": "req-1",
            "model_name": "demo",
        },
    )
    assert records, "the provider should log something"
    assert not any("DEADBEEF" in message for _, message in records)
