"""The upload page's endpoints: a customer's ingest path.

Same bargain as ``test_service.py`` — the fake stands in for parsing and
embedding, everything else is real: routing, auth, the multipart write, the
registry on disk, the ingest job and its status. A corpus is a real directory
tree here, because the endpoint's whole job is to produce one.
"""

import threading
import time

import pytest
from fastapi.testclient import TestClient

from app import auth, jobs, registry
from app.main import create_app
from app.routes import asset_file, ui
from chillrag.corpus import CorpusBundle, CorpusManager
from chillrag.rag_tool import file_id_for, source_id_for
from chillrag.tools.parser import SOURCE_SUFFIXES
from chillrag.tools.searcher import Chunk, DocumentStore

PASSWORD = "correct horse"
STATIC = ui._STATIC


class IngestTool:
    """Stands in for parse → chunk → embed: one chunk per document, instantly."""

    def __init__(self) -> None:
        self._manager = CorpusManager.instance()

    def upload_file_complete(self, folder, *, rebuild=False):
        store = DocumentStore(name=folder)
        for path in self._manager.source_docs(folder):
            store.add_chunk(Chunk(
                chunk_id=f"{path.name}#0", doc_id=path.name, order=0, text="x",
                meta={"doc": path.name, "content_types": ["text"]},
            ))
        self._manager._bundles[folder] = CorpusBundle(
            folder=folder, store=store, signature=self._manager.signature(folder),
            cache_dir=self._manager.cache_dir_for(folder),
        )
        return None


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    materials = tmp_path / "materials"
    cache = tmp_path / "cache"
    materials.mkdir()
    cache.mkdir()
    monkeypatch.setattr(registry, "_PATH", cache / "service_registry.json")
    monkeypatch.setattr(ui, "MATERIALS_DIR", materials)
    monkeypatch.setattr(asset_file, "MATERIALS_DIR", materials)
    monkeypatch.setattr(jobs, "RAGTool", IngestTool)
    monkeypatch.setattr(jobs, "_STATUS", {})
    monkeypatch.setattr(jobs, "_DIRTY", set())

    manager = CorpusManager(materials_dir=materials, cache_dir=cache)
    monkeypatch.setattr(CorpusManager, "instance", classmethod(lambda cls: manager))
    return materials


@pytest.fixture
def client(workspace):
    store = auth.AuthStore(
        users={
            "svc": auth.User("svc", PASSWORD, "admin"),
            "viewer": auth.User("viewer", "v", "user"),
        },
        secret="test-secret",
    )
    with TestClient(create_app(store)) as test_client:
        test_client.post("/space/api/passport/login", json={"username": "svc", "passwd": PASSWORD})
        yield test_client


def upload(client, *files: tuple[str, bytes], **params):
    """POST ``files`` as ``(name, content)`` pairs to the UI upload endpoint."""
    return client.post(
        "/api/upload",
        params=params,
        files=[("files", (name, content, "application/octet-stream")) for name, content in files],
    )


def wait_for_ingest(folder: str) -> None:
    for _ in range(200):
        if jobs.status(folder) != "processing":
            return
        time.sleep(0.01)
    raise AssertionError("ingest did not finish")


# ------------------------------------------------------------ new corpus --

def test_upload_creates_the_corpus_and_registers_every_file(client, workspace):
    response = upload(client, ("a.pdf", b"one"), ("b.docx", b"two!"), folder="手册")
    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 0
    data = body["data"]
    assert data["sourceId"] == source_id_for("手册") and data["folder"] == "手册"
    assert [f["name"] for f in data["files"]] == ["a.pdf", "b.docx"]
    assert [f["size"] for f in data["files"]] == [3, 4]
    assert [f["fileId"] for f in data["files"]] == [file_id_for("a.pdf"), file_id_for("b.docx")]

    # The files are on disk under the materials root, where the ingest reads them.
    assert (workspace / "手册" / "a.pdf").read_bytes() == b"one"
    # And the corpus is registered, which is what puts it in the dropdown.
    assert registry.folder_of(source_id_for("手册")) == "手册"
    assert registry.file_of(file_id_for("a.pdf")) == {"source": source_id_for("手册"), "name": "a.pdf"}


def test_upload_queues_an_ingest(client):
    upload(client, ("a.pdf", b"one"), folder="手册")
    wait_for_ingest("手册")
    assert jobs.status("手册") == "finish"


def test_a_second_upload_lands_in_the_same_corpus(client, workspace):
    first = upload(client, ("a.pdf", b"one"), folder="手册").json()["data"]
    second = upload(client, ("b.pdf", b"two"), sourceId=first["sourceId"]).json()["data"]

    assert second["sourceId"] == first["sourceId"] and second["folder"] == "手册"
    assert sorted(p.name for p in (workspace / "手册").iterdir()) == ["a.pdf", "b.pdf"]
    assert registry.sources() == [(first["sourceId"], "手册")]


def test_a_plain_user_may_create_a_corpus(client):
    """The decision this endpoint exists for: no admin step to start a corpus."""
    client.post("/api/auth/logout")
    client.post("/space/api/passport/login", json={"username": "viewer", "passwd": "v"})
    assert upload(client, ("a.pdf", b"one"), folder="viewer-corpus").status_code == 200
    # ...while the vendor's source creation stays admin-only.
    assert client.put("/space/api/service/asset-file/resource",
                      json={"resourceName": "nope"}).status_code == 403


def test_upload_without_a_login_is_refused(client, workspace):
    client.post("/api/auth/logout")
    assert upload(client, ("a.pdf", b"one"), folder="手册").status_code == 401
    assert not (workspace / "手册").exists()


# -------------------------------------------------------------- refusals --

def test_the_whole_upload_is_refused_when_one_file_is_unsupported(client, workspace):
    """.txt is not ingestable, so accepting it would silently produce nothing."""
    response = upload(client, ("a.pdf", b"one"), ("notes.txt", b"two"), folder="手册")
    assert response.status_code == 400
    assert "notes.txt" in response.json()["msg"]
    assert not (workspace / "手册").exists() and registry.sources() == []


def test_no_files_is_400(client):
    assert client.post("/api/upload", params={"folder": "手册"}).status_code == 400


def test_a_path_shaped_corpus_name_is_400(client, workspace):
    for name in ("../escape", "a/b", ".."):
        response = upload(client, ("a.pdf", b"one"), folder=name)
        assert response.status_code == 400, name
    assert list(workspace.iterdir()) == []


def test_a_folder_and_a_source_id_together_is_400(client):
    upload(client, ("a.pdf", b"one"), folder="手册")
    response = upload(client, ("b.pdf", b"two"), folder="手册", sourceId=source_id_for("手册"))
    assert response.status_code == 400


def test_neither_a_folder_nor_a_source_id_is_400(client):
    assert upload(client, ("a.pdf", b"one")).status_code == 400


def test_an_unknown_source_id_is_404(client):
    response = upload(client, ("a.pdf", b"one"), sourceId=12345)
    assert response.status_code == 404
    assert response.json()["msg"] == "unknown source 12345"


def test_a_nameless_upload_is_400(client):
    """``..`` has a directory part but no file name to write."""
    response = client.post("/api/upload", params={"folder": "手册"},
                           files=[("files", ("..", b"one", "application/octet-stream"))])
    assert response.status_code == 400


# ---------------------------------------------------------------- status --

def test_status_reports_the_finished_corpus_per_file(client):
    source_id = upload(client, ("a.pdf", b"one"), ("b.docx", b"two"), folder="手册").json()["data"]["sourceId"]
    wait_for_ingest("手册")
    data = client.get("/api/upload/status", params={"sourceId": source_id}).json()["data"]

    assert data["state"] == "finish"
    assert data["documents"] == 2 and data["chunks"] == 2
    assert [(f["name"], f["chunks"]) for f in data["files"]] == [("a.pdf", 1), ("b.docx", 1)]


def test_status_reports_a_file_that_produced_no_chunks(client, workspace):
    """A document the parser got nothing out of is visible rather than counted."""
    source_id = upload(client, ("a.pdf", b"one"), folder="手册").json()["data"]["sourceId"]
    wait_for_ingest("手册")
    CorpusManager.instance()._bundles["手册"].store.docs.clear()

    data = client.get("/api/upload/status", params={"sourceId": source_id}).json()["data"]
    assert data["documents"] == 0 and data["chunks"] == 0
    assert data["files"] == [{"name": "a.pdf", "fileId": file_id_for("a.pdf"), "chunks": 0}]


def test_status_of_a_failed_ingest_is_failed(client, workspace, monkeypatch):
    class Broken(IngestTool):
        def upload_file_complete(self, folder, *, rebuild=False):
            raise RuntimeError("mineru died")

    source_id = upload(client, ("a.pdf", b"one"), folder="手册").json()["data"]["sourceId"]
    wait_for_ingest("手册")  # let the good job drain before breaking the next one
    monkeypatch.setattr(jobs, "RAGTool", Broken)
    upload(client, ("b.pdf", b"two"), sourceId=source_id)
    wait_for_ingest("手册")

    assert client.get("/api/upload/status", params={"sourceId": source_id}).json()["data"]["state"] == "failed"


def test_status_of_an_unknown_source_is_404(client):
    assert client.get("/api/upload/status", params={"sourceId": 1}).status_code == 404


def test_status_reports_while_ingesting_instead_of_waiting(client, workspace, monkeypatch):
    """A poll taken during an ingest must answer, not block on the parse.

    ``prepare`` holds the manager lock for the whole parse and embed, so any
    manager read during a job stalls for as long as that job runs — minutes, on
    a real annual report — instead of reporting ``processing``.
    """
    source_id = upload(client, ("a.pdf", b"one"), folder="手册").json()["data"]["sourceId"]
    wait_for_ingest("手册")

    manager = CorpusManager.instance()
    jobs._STATUS["手册"] = "processing"

    def would_block(folder, **kwargs):
        raise AssertionError("status read consulted the manager while a job was running")

    monkeypatch.setattr(manager, "try_get", would_block)

    data = client.get("/api/upload/status", params={"sourceId": source_id}).json()["data"]
    assert data["state"] == "processing"
    assert [f["name"] for f in data["files"]] == ["a.pdf"]


def test_status_answers_while_another_folder_holds_the_manager_lock(client, workspace):
    """The manager lock is global, so one folder's parse stalled every other poll.

    ``CorpusManager._lock`` is on the singleton, not per folder: while any
    folder is ingesting, a status read for a *different*, already-finished
    folder waited for that parse to finish. Measured against a real corpus, that
    was a 60s client timeout, which the client reads as "the service has no
    files" and answers by re-uploading the lot.

    The lock is held for the whole poll rather than released on a timer, so a
    regression hangs here instead of quietly taking the timeout.
    """
    source_id = upload(client, ("a.pdf", b"one"), folder="手册").json()["data"]["sourceId"]
    wait_for_ingest("手册")

    manager = CorpusManager.instance()
    held, release = threading.Event(), threading.Event()

    def hold_the_lock():
        with manager._lock:
            held.set()
            release.wait(30)

    holder = threading.Thread(target=hold_the_lock)
    holder.start()
    assert held.wait(5), "holder thread never took the lock"

    answer: dict = {}

    def poll():
        answer.update(client.get("/api/upload/status", params={"sourceId": source_id}).json()["data"])

    poller = threading.Thread(target=poll)
    poller.start()
    poller.join(3)
    answered = not poller.is_alive()
    release.set()
    holder.join()
    poller.join(5)

    assert answered, "the status poll blocked on the manager lock"
    # The chunk counts are what a locked-out read has to give up; the answer is not.
    assert answer["state"] == "finish"
    assert [f["name"] for f in answer["files"]] == ["a.pdf"]


# ---------------------------------------------------------------- ensure --

def test_ensure_queues_an_ingest_for_a_folder_that_was_never_built(client, workspace):
    """The state the client cannot fix on its own: documents present, no job queued.

    ``upload`` queues a job, so drain it and drop what it built — which is what
    a folder whose documents were copied onto the server rather than uploaded
    looks like. The client uploads only what is missing, so it never reaches
    ``web-upload``, which is the only other place a job is queued.
    """
    source_id = upload(client, ("a.pdf", b"one"), folder="手册").json()["data"]["sourceId"]
    wait_for_ingest("手册")

    manager = CorpusManager.instance()
    manager._bundles.clear()
    jobs._STATUS.clear()
    assert manager.find("手册") is None

    client.post("/api/upload/ensure", params={"sourceId": source_id})
    wait_for_ingest("手册")
    assert manager.find("手册") is not None


def test_ensure_leaves_a_prepared_corpus_alone(client, workspace, monkeypatch):
    """Asking about a ready corpus must not queue a job behind another ingest.

    The job pool has one worker, so a redundant job would sit behind whatever is
    parsing now and leave this folder reporting ``processing`` until it drained.
    """
    source_id = upload(client, ("a.pdf", b"one"), folder="手册").json()["data"]["sourceId"]
    wait_for_ingest("手册")

    monkeypatch.setattr(CorpusManager.instance(), "is_prepared", lambda folder: True)
    submitted: list[str] = []
    monkeypatch.setattr(jobs, "submit", submitted.append)

    data = client.post("/api/upload/ensure", params={"sourceId": source_id}).json()["data"]
    assert submitted == []
    assert data["prepared"] is True


def test_ensure_requeues_a_failed_ingest(client, workspace, monkeypatch):
    """A failed job leaves no corpus and no running job, so it is queued again."""
    source_id = upload(client, ("a.pdf", b"one"), folder="手册").json()["data"]["sourceId"]
    wait_for_ingest("手册")

    jobs._STATUS["手册"] = "failed"
    submitted: list[str] = []
    monkeypatch.setattr(jobs, "submit", submitted.append)

    client.post("/api/upload/ensure", params={"sourceId": source_id})
    assert submitted == ["手册"]


def test_ensure_of_an_unknown_source_is_404(client):
    assert client.post("/api/upload/ensure", params={"sourceId": 1}).status_code == 404


# ---------------------------------------------------------------- config --

def test_ui_config_serves_the_ingestable_suffixes(client):
    data = client.get("/api/ui/config").json()["data"]
    assert data["suffixes"] == sorted(SOURCE_SUFFIXES)


def test_the_upload_page_does_not_hardcode_what_it_accepts():
    """The accept list comes from ``/api/ui/config``, so it cannot drift."""
    html = (STATIC / "upload.html").read_text(encoding="utf-8")
    assert "/api/ui/config" in html
    assert ".pdf" not in html and ".docx" not in html
