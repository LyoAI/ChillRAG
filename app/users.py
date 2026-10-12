"""Account administration from the command line.

    uv run python -m app.users list
    uv run python -m app.users add alice --role admin
    uv run python -m app.users passwd alice
    uv run python -m app.users remove alice

Writes the same accounts file the service reads (``users.yaml`` at the repo root,
or ``CHILLRAG_USERS_FILE``). A running service notices the change at the next
login, so no restart is needed. A password left off the command line is generated
and printed — preferring that keeps it out of your shell history.
"""

from __future__ import annotations

import argparse
import secrets
import sys

import yaml

from app import auth

_HEADER = """\
# Accounts for the web UI and HTTP API. Gitignored; plaintext intranet policy.
# role: admin (manage sources) | user (ask and retrieve)
# Managed with: uv run python -m app.users --help
"""


def _save(users: dict[str, auth.User]) -> None:
    """Rewrite the whole file — the account list is tiny, and a full rewrite
    keeps it canonical. Temp file plus rename, so a reader never sees a partial
    file."""
    path = auth.users_file()
    document = {
        "users": {name: {"password": user.password, "role": user.role} for name, user in sorted(users.items())}
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(_HEADER + yaml.safe_dump(document, sort_keys=False, allow_unicode=True), encoding="utf-8")
    tmp.replace(path)
    path.chmod(0o600)  # plaintext passwords


def _password(given: str | None) -> tuple[str, bool]:
    """``(password, generated)``."""
    return (given, False) if given else (secrets.token_urlsafe(12), True)


def _run(args: argparse.Namespace) -> int:
    path = auth.users_file()
    users = auth.load_users(path)

    if args.command == "list":
        print(f"{path} ({len(users)} account{'' if len(users) == 1 else 's'})")
        for name in sorted(users):
            print(f"  {name:<20} {users[name].role}")
        return 0

    if args.command == "add":
        if not auth.valid_username(args.username):
            raise RuntimeError(f"invalid username {args.username!r}: letters, digits, _ or - only")
        if args.username in users:
            raise RuntimeError(f"{args.username} already exists; use passwd to change its password")
        password, generated = _password(args.password)
        users[args.username] = auth.User(args.username, password, args.role)
        _save(users)
        print(f"created {args.username} ({args.role}) in {path}")
        if generated:
            print(f"password: {password}")
        return 0

    if args.command == "passwd":
        if args.username not in users:
            raise RuntimeError(f"no such account {args.username!r}")
        password, generated = _password(args.password)
        users[args.username] = auth.User(args.username, password, users[args.username].role)
        _save(users)
        print(f"changed the password of {args.username}")
        if generated:
            print(f"password: {password}")
        return 0

    # remove
    if args.username not in users:
        raise RuntimeError(f"no such account {args.username!r}")
    if len(users) == 1:
        raise RuntimeError("refusing to remove the last account: the service would have no way in")
    del users[args.username]
    _save(users)
    print(f"removed {args.username}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.users", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="show the accounts and the file they come from")

    add = sub.add_parser("add", help="create an account")
    add.add_argument("username")
    add.add_argument("--role", choices=auth.ROLES, default="user")
    add.add_argument("--password", help="default: generated and printed")

    passwd = sub.add_parser("passwd", help="set an account's password")
    passwd.add_argument("username")
    passwd.add_argument("--password", help="default: generated and printed")

    remove = sub.add_parser("remove", help="delete an account")
    remove.add_argument("username")

    args = parser.parse_args(argv)
    try:
        return _run(args)
    except (OSError, RuntimeError) as exc:  # unreadable file, bad name, unknown account
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
