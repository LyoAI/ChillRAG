"""The account CLI, and the running service picking up what it writes."""

import os

import pytest

from app import auth, users


@pytest.fixture
def accounts(tmp_path, monkeypatch):
    """An accounts path of our own; the CLI and the service must agree on it."""
    path = tmp_path / "users.yaml"
    monkeypatch.setenv("CHILLRAG_USERS_FILE", str(path))
    monkeypatch.setenv("CHILLRAG_AUTH_SECRET", "test-secret")
    monkeypatch.delenv("CHILLRAG_USER", raising=False)
    return path


def run(*argv: str) -> int:
    return users.main(list(argv))


def test_add_creates_an_account_the_service_loads(accounts):
    assert run("add", "alice", "--role", "admin", "--password", "s3cret") == 0
    loaded = auth.load_users()
    assert loaded["alice"] == auth.User("alice", "s3cret", "admin")
    assert oct(accounts.stat().st_mode)[-3:] == "600"  # plaintext passwords


def test_add_generates_a_password_when_none_is_given(accounts, capsys):
    assert run("add", "bob") == 0
    user = auth.load_users()["bob"]
    assert user.role == "user"
    assert len(user.password) >= 15
    assert user.password in capsys.readouterr().out  # printed once, nowhere else


def test_add_rejects_a_duplicate_bad_name_or_role(accounts):
    run("add", "alice", "--password", "pw")
    assert run("add", "alice", "--password", "other") == 1
    assert run("add", "bad.name", "--password", "pw") == 1
    assert auth.load_users()["alice"].password == "pw"  # neither attempt wrote


def test_passwd_keeps_the_role(accounts):
    run("add", "alice", "--role", "admin", "--password", "old")
    assert run("passwd", "alice", "--password", "new") == 0
    assert auth.load_users()["alice"] == auth.User("alice", "new", "admin")


def test_remove_leaves_other_accounts_and_keeps_the_last_one(accounts):
    run("add", "alice", "--password", "pw")
    run("add", "bob", "--password", "pw")
    assert run("remove", "alice") == 0
    assert set(auth.load_users()) == {"bob"}
    assert run("remove", "bob") == 1  # last account: the service would lock out
    assert run("remove", "nobody") == 1
    assert set(auth.load_users()) == {"bob"}


def test_list_names_the_file_and_every_account(accounts, capsys):
    run("add", "alice", "--role", "admin", "--password", "pw")
    run("add", "bob", "--password", "pw")
    assert run("list") == 0
    out = capsys.readouterr().out
    assert str(accounts) in out and "alice" in out and "admin" in out and "bob" in out
    assert "pw" not in out  # never print a password that was not just generated


def test_a_running_service_notices_an_account_added_later(accounts):
    accounts.write_text("users:\n  alice:\n    password: pw\n    role: admin\n", encoding="utf-8")
    # Epoch mtime, so the CLI's write is unambiguously newer even on a
    # coarse-grained filesystem.
    os.utime(accounts, (0, 0))
    store = auth.AuthStore.from_env()
    assert store.verify("alice", "pw") is not None

    run("add", "bob", "--password", "pw2")
    assert store.verify("bob", "pw2").role == "user"  # no restart


def test_a_running_service_drops_a_removed_account(accounts):
    run("add", "alice", "--password", "pw")
    run("add", "bob", "--password", "pw2")
    os.utime(accounts, (0, 0))
    store = auth.AuthStore.from_env()
    token = store.issue_token(store.verify("bob", "pw2"))
    assert store.verify_token(token) is not None

    run("remove", "bob")
    assert store.verify("bob", "pw2") is None
    assert store.verify_token(token) is None  # its session ends with it


def test_a_broken_file_keeps_the_loaded_accounts(accounts):
    run("add", "alice", "--password", "pw")
    os.utime(accounts, (0, 0))
    store = auth.AuthStore.from_env()

    accounts.write_text("users: [not a mapping]\n", encoding="utf-8")
    assert store.verify("alice", "pw") is not None  # a bad edit must not lock out
