"""HTTP contract of the service, exercised end to end with a fake RAGTool.

The fake stands in for parsing, embedding and the agent, which need models and
a JVM. Everything else is real: routing, auth cookies, query-string parsing,
the registry on disk, background ingest, and the SSE encoding.
"""

import json
import time

import pytest
from fastapi.testclient import TestClient

from app import auth, jobs, registry
from app.main import create_app
from chillrag.rag_tool import source_id_for
from app.routes import asset_file, answer as answer_route

PASSWORD = "correct horse"


class FakeTool:
    """Records what the service asked for; answers are canned."""

    calls: list = []

    def __init__(self) -> None:
        pass

    def upload_file_complete(self, folder):
        FakeTool.calls.append(("upload", folder))
        return None

    def normal_rag_recall(self, question, source_id_list=(), origin_ids=(), context_trace_ids="", recall_content_types=None):
        FakeTool.calls.append(("recall", question, list(source_id_list), list(origin_ids), recall_content_types))
        return [{"confidence": 0.6, "content_type": "text", "metadata": "文件名称:a",
                 "rank": 1, "score": 0.5, "segment": "hi", "source": "1#0"}]

    def get_segments(self, file_id, token=None):
        return [{"segment": {"index": 0, "content": "line one\nline two"}}]

    def answer_by_question(self, question, source_id_list=(), **kwargs):
        return "The answer is 42.\nSecond line."


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "_PATH", tmp_path / "registry.json")
    monkeypatch.setattr(asset_file, "MATERIALS_DIR", tmp_path / "materials")
    monkeypatch.setattr("app.corpus.RAGTool", FakeTool)
    monkeypatch.setattr(jobs, "RAGTool", FakeTool)
    monkeypatch.setattr(jobs, "_STATUS", {})
    FakeTool.calls = []
    store = auth.AuthStore(
        users={
            "svc": auth.User("svc", PASSWORD, "admin"),
            "viewer": auth.User("viewer", "v", "user"),
        },
        secret="test-secret",
    )
    with TestClient(create_app(store)) as test_client:
        yield test_client


def login(client: TestClient) -> None:
    response = client.post("/space/api/passport/login", json={"username": "svc", "passwd": PASSWORD})
    assert response.status_code == 200


def wait_for_ingest(folder: str) -> None:
    for _ in range(200):
        if jobs.status(folder) != "processing":
            return
        time.sleep(0.01)
    raise AssertionError("ingest did not finish")


# ----------------------------------------------------------------- auth --

def test_login_issues_space_auth_cookie_and_envelope(client):
    response = client.post("/space/api/passport/login", json={"username": "svc", "passwd": PASSWORD})
    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 0 and body["data"]["userId"] == 1
    assert "space-auth=" in response.headers["set-cookie"]


def test_wrong_password_is_401_with_envelope(client):
    response = client.post("/space/api/passport/login", json={"username": "svc", "passwd": "nope"})
    assert response.status_code == 401
    assert response.json() == {"code": 401, "msg": "invalid username or password", "data": None}


def test_protected_route_without_cookie_is_401_envelope(client):
    response = client.post("/space/api/service/llmapi/ht/normal_rag_recall_without_doc_group",
                           params={"question": "q", "source_ids": "1"})
    assert response.status_code == 401
    assert response.json()["msg"] == "No Auth"


def test_tampered_or_expired_token_is_rejected():
    store = auth.AuthStore(users={"svc": auth.User("svc", "pw", "admin")}, secret="s")
    user = store.users["svc"]
    assert store.verify_token(store.issue_token(user, now=0)) is None  # expired
    fresh = store.issue_token(user)
    name, expiry, sig = fresh.split(".")
    assert store.verify_token(f"{name}.{expiry}.{'0' * len(sig)}") is None  # forged


def test_roles_gate_source_creation(client):
    client.post("/space/api/passport/login", json={"username": "viewer", "passwd": "v"})
    assert client.put("/space/api/service/asset-file/resource",
                      json={"resourceName": "nope"}).status_code == 403
    assert client.get("/api/auth/me").json()["data"] == {"username": "viewer", "role": "user"}


def test_users_file_is_loaded(tmp_path, monkeypatch):
    users = tmp_path / "users.yaml"
    users.write_text("users:\n  bob:\n    password: pw\n    role: user\n", encoding="utf-8")
    monkeypatch.setenv("CHILLRAG_USERS_FILE", str(users))
    monkeypatch.delenv("CHILLRAG_USER", raising=False)
    monkeypatch.setenv("CHILLRAG_AUTH_SECRET", "s")
    store = auth.AuthStore.from_env()
    assert store.verify("bob", "pw").role == "user"
    assert store.verify("bob", "wrong") is None


# ------------------------------------------------------------ sources --

def test_source_upload_ingest_and_status(client):
    login(client)
    source = client.put("/space/api/service/asset-file/resource", json={"resourceName": "bank-a"}).json()["data"]
    upload = client.post(f"/space/api/service/asset-file/web-upload?sourceId={source}",
                         files=[("files", ("report.txt", b"hello", "text/plain"))]).json()
    file_id = upload["data"][0]
    wait_for_ingest("bank-a")
    status = client.post("/space/api/service/asset-file", json=[file_id]).json()["data"]
    assert status == [{"id": file_id, "fileName": "report.txt", "status": [{"code": "finish"}]}]
    assert ("upload", "bank-a") in FakeTool.calls


def test_upload_rejects_path_traversal_in_filename(client):
    login(client)
    source = client.put("/space/api/service/asset-file/resource", json={"resourceName": "bank-b"}).json()["data"]
    response = client.post(f"/space/api/service/asset-file/web-upload?sourceId={source}",
                           files=[("files", ("../../escape.txt", b"x", "text/plain"))])
    # The name is reduced to its basename, so it lands inside the folder.
    assert response.status_code == 200
    assert not (asset_file.MATERIALS_DIR.parent / "escape.txt").exists()


# ------------------------------------------------------------- recall --

def test_recall_parses_vendor_query_string(client):
    login(client)
    registry.register_source(5, "bank-x")
    registry.register_source(6, "bank-y")
    response = client.post(
        "/space/api/service/llmapi/ht/normal_rag_recall_without_doc_group",
        params=[("question", "revenue?"), ("source_ids", "5,6"), ("origin_ids", "[7, 8]"),
                ("context_trace_ids", ""), ("recall_content_types", "excel"), ("recall_content_types", "text")],
    )
    assert response.status_code == 200
    recall_call = [c for c in FakeTool.calls if c[0] == "recall"][0]
    assert recall_call == ("recall", "revenue?", [5, 6], [7, 8], ["excel", "text"])
    assert response.json()["data"][0]["source"] == "1#0"


def test_recall_without_source_is_400(client):
    login(client)
    response = client.post("/space/api/service/llmapi/ht/normal_rag_recall_without_doc_group",
                           params={"question": "q", "source_ids": ""})
    assert response.status_code == 400


# ----------------------------------------------------------- segments --

def test_segments_wrap_content_as_nested_json_string(client):
    login(client)
    source = client.put("/space/api/service/asset-file/resource", json={"resourceName": "bank-c"}).json()["data"]
    file_id = client.post(f"/space/api/service/asset-file/web-upload?sourceId={source}",
                          files=[("files", ("doc.txt", b"x", "text/plain"))]).json()["data"][0]
    segments = client.get(f"/space/api/service/asset-file/segments/{file_id}").json()["data"]
    content = segments[0]["segment"]["content"]
    assert json.loads(content) == {"content": "line one\nline two"}


# ------------------------------------------------------------- answer --

def decode_like_vendor_client(stream_text: str) -> str:
    """The vendor client's SSE reader, reproduced for the contract check."""
    out = ""
    for line in stream_text.splitlines():
        if not line:
            continue
        content = line[5:].strip() if line.startswith("data:") else line
        out += content.replace("[new_line]", "\n").replace("[whitespace]", " ")
    parts = out.split("</strong>")
    return "".join(parts[1:]).strip() if len(parts) > 1 else out


def test_post_answer_streams_text_the_vendor_client_decodes(client):
    login(client)
    source = client.put("/space/api/service/asset-file/resource", json={"resourceName": "bank-d"}).json()["data"]
    response = client.post(
        "/space/api/service/sse/llmapi/ht/text/post-answer",
        json={"question": "What is it?", "source_category_filter": json.dumps({str(source): []}), "mode": 0},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert decode_like_vendor_client(response.text) == "The answer is 42.\nSecond line."


def test_answer_encoding_keeps_newlines_and_spaces():
    encoded = answer_route.encode("a b\nc")
    assert encoded == "a[whitespace]b[new_line]c"


def test_post_answer_rejects_more_than_one_source(client):
    login(client)
    registry.register_source(1, "bank-p")
    registry.register_source(2, "bank-q")
    filter_json = json.dumps({"1": [], "2": []})
    response = client.post("/space/api/service/sse/llmapi/ht/text/post-answer",
                           json={"question": "q", "source_category_filter": filter_json})
    assert response.status_code == 400
    assert response.json()["msg"] == "post-answer takes exactly one source"


def test_post_answer_reports_a_failed_model_as_503(client, monkeypatch):
    """The agent returns nothing when the model call fails; 200 with an empty
    body would read as a blank answer rather than a broken service."""
    login(client)
    source = client.put("/space/api/service/asset-file/resource",
                        json={"resourceName": "bank-r"}).json()["data"]
    monkeypatch.setattr(FakeTool, "answer_by_question", lambda self, question, **kwargs: "")
    response = client.post("/space/api/service/sse/llmapi/ht/text/post-answer",
                           json={"question": "q", "source_category_filter": json.dumps({str(source): []})})
    assert response.status_code == 503
    assert "returned nothing" in response.json()["msg"]


def test_upload_during_ingest_reruns_the_folder():
    """An upload arriving mid-ingest must re-queue: the running pass snapshots
    an incomplete file set, so its store would silently miss the new files."""
    import threading
    from app import jobs as jobs_mod

    calls = []
    entered = threading.Event()
    release = threading.Event()

    class BlockingTool:
        def upload_file_complete(self, folder):
            calls.append(folder)
            entered.set()
            release.wait(timeout=5)
            return None

    monkey = pytest.MonkeyPatch()
    monkey.setattr(jobs_mod, "RAGTool", BlockingTool)
    monkey.setattr(jobs_mod, "_STATUS", {})
    monkey.setattr(jobs_mod, "_DIRTY", set())
    try:
        jobs_mod.submit("f")            # first pass starts, blocks inside
        assert entered.wait(timeout=5)
        jobs_mod.submit("f")            # upload while processing -> dirty
        release.set()                   # let the first pass finish
        for _ in range(100):            # wait for the re-run
            if len(calls) >= 2:
                break
            time.sleep(0.02)
        assert calls == ["f", "f"]
        assert jobs_mod.status("f") == "finish"
    finally:
        release.set()
        monkey.undo()


def test_ui_page_and_source_list(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "/api/chat/stream" in page.text and "ChillRAG" in page.text
    login(client)
    client.put("/space/api/service/asset-file/resource", json={"resourceName": "bank-ui"})
    listing = client.get("/space/api/service/asset-file/resource").json()["data"]
    assert {"sourceId": source_id_for("bank-ui"), "resourceName": "bank-ui"} in listing


def test_chat_stream_emits_json_events(client, monkeypatch):
    from chillrag.agent.react_agent import AgentEvent

    async def fake_stream(question, **kwargs):
        yield AgentEvent(kind="tool", tool_name="search", tool_args={"query": "q"})
        yield AgentEvent(kind="delta", text="the answer is ")
        yield AgentEvent(kind="delta", text="42")
        yield AgentEvent(kind="done", answer="the answer is 42", num_tool_calls=1, terminated="answer")

    class FakeTool:
        def answer_stream(self, question, **kwargs):
            return fake_stream(question, **kwargs)

    monkeypatch.setattr("app.routes.ui.attached_tool", lambda ids: FakeTool())
    login(client)
    with client.stream("POST", "/api/chat/stream", json={"question": "q", "source_id": 1}) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())
    assert '"type": "tool"' in body and '"tool": "search"' in body
    assert '"type": "delta"' in body and '"text": "42"' in body
    assert '"type": "done"' in body


def test_chat_stream_forwards_history(client, monkeypatch):
    from chillrag.agent.react_agent import AgentEvent
    captured = {}

    class FakeTool:
        def answer_stream(self, question, **kwargs):
            captured.update(kwargs)
            async def gen():
                yield AgentEvent(kind="done", answer="ok")
            return gen()

    monkeypatch.setattr("app.routes.ui.attached_tool", lambda ids: FakeTool())
    login(client)
    history = [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"}]
    with client.stream("POST", "/api/chat/stream", json={"question": "Q2", "source_id": 1, "history": history}) as resp:
        assert resp.status_code == 200
        "".join(resp.iter_text())
    assert captured["history"] == history


def test_logout_expires_the_session(client):
    login(client)
    assert client.get("/space/api/service/asset-file/resource").status_code == 200
    assert client.post("/api/auth/logout").status_code == 200
    assert client.get("/space/api/service/asset-file/resource").status_code == 401
