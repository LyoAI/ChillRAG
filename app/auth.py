"""Username + password auth with an account list and roles.

Accounts live in a gitignored ``users.yaml`` (several users, each with a role),
or — for the one-account case — the ``CHILLRAG_USER`` / ``CHILLRAG_PASSWORD``
pair. Manage the file with ``python -m app.users`` (see that module) or edit it
directly. Tokens are HMAC-signed with a server secret: ``CHILLRAG_AUTH_SECRET``,
or one auto-generated and cached under the cache dir, so restarts keep
sessions without any configuration. No session table: verification needs only
the account list and the secret.
"""

import hmac
import logging
import os
import re
import secrets
import time
from dataclasses import dataclass
from pathlib import Path

import yaml
from fastapi import HTTPException, Request

logger = logging.getLogger("chillrag.auth")

COOKIE = "space-auth"
TOKEN_TTL_S = 12 * 3600
ROLES = ("admin", "user")
# No dot: the token is ``<username>.<expiry>.<signature>``, split on dots.
_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class User:
    username: str
    password: str
    role: str = "user"


def users_file() -> Path:
    """The accounts file: ``CHILLRAG_USERS_FILE``, or ``users.yaml`` at the repo
    root. Anchored at the repo so the service and the CLI agree from any cwd."""
    from chillrag.config import REPO_ROOT, resolve_path

    return resolve_path(os.environ.get("CHILLRAG_USERS_FILE", ""), REPO_ROOT / "users.yaml")


def valid_username(name: str) -> bool:
    """Letters, digits, ``_`` and ``-``: safe in YAML, in a cookie and in the
    dotted token."""
    return bool(_NAME.match(name))


def load_users(path: Path | None = None) -> dict[str, User]:
    """Accounts from the accounts file; a missing file yields none.

    A bad username or role raises, so a typo in the file fails loudly rather
    than silently dropping the account.
    """
    path = path or users_file()
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    users: dict[str, User] = {}
    for name, entry in (data.get("users") or {}).items():
        if not valid_username(name):
            raise RuntimeError(f"invalid username {name!r}: use letters, digits, _ or -")
        role = entry.get("role", "user")
        if role not in ROLES:
            raise RuntimeError(f"user {name!r}: role must be one of {ROLES}")
        users[name] = User(name, str(entry.get("password", "")), role)
    return users


def _mtime(path: Path | None) -> float | None:
    try:
        return path.stat().st_mtime if path else None
    except OSError:
        return None


class AuthStore:
    """The account list plus token issuing/verification.

    An accounts file that exists is watched: an edit while the service runs (the
    CLI) applies from the next login on, so adding a user needs no restart.
    """

    def __init__(self, users: dict[str, User], secret: str, users_path: Path | None = None) -> None:
        self.users = users
        self.secret = secret
        self._users_path = users_path
        self._users_mtime = _mtime(users_path)

    # ------------------------------------------------------------ loading --

    @classmethod
    def from_env(cls) -> "AuthStore":
        path = users_file()
        users = load_users(path) or cls._from_env_pair()
        if not users:
            raise RuntimeError(
                "auth is not configured: create users.yaml (see README) "
                "or set CHILLRAG_USER and CHILLRAG_PASSWORD"
            )
        return cls(users, cls._load_secret(), path)

    def _reload_if_changed(self) -> None:
        """Re-read the accounts file when it changes on disk.

        A file that is malformed or holds no accounts keeps the loaded ones: a
        half-finished edit must not lock everyone out mid-write.
        """
        if self._users_path is None:
            return
        stamp = _mtime(self._users_path)
        if stamp is None or stamp == self._users_mtime:
            return
        self._users_mtime = stamp
        try:
            users = load_users(self._users_path)
        except Exception:
            logger.exception("accounts file %s is invalid, keeping the loaded accounts", self._users_path)
            return
        if not users:
            logger.warning("accounts file %s lists no accounts, keeping the loaded ones", self._users_path)
            return
        self.users = users

    @staticmethod
    def _from_env_pair() -> dict[str, User]:
        name = os.environ.get("CHILLRAG_USER", "")
        if not name:
            return {}
        return {name: User(name, os.environ.get("CHILLRAG_PASSWORD", ""), "admin")}

    @staticmethod
    def _load_secret() -> str:
        env = os.environ.get("CHILLRAG_AUTH_SECRET")
        if env:
            return env
        # Auto-generate once and cache, so restarts keep issued tokens valid.
        from chillrag.config import CACHE_DIR

        path = CACHE_DIR / "auth_secret"
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
        value = secrets.token_hex(32)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
        path.chmod(0o600)
        return value

    # ----------------------------------------------------------- accounts --

    def verify(self, username: str, password: str) -> User | None:
        self._reload_if_changed()
        user = self.users.get(username)
        if user and hmac.compare_digest(password.encode(), user.password.encode()):
            return user
        return None

    def user_id(self, username: str) -> int:
        """A stable small id per username (sorted order), for API compat."""
        return sorted(self.users).index(username) + 1

    # ------------------------------------------------------------- tokens --

    def issue_token(self, user: User, now: float | None = None) -> str:
        expiry = int((time.time() if now is None else now) + TOKEN_TTL_S)
        body = f"{user.username}.{user.role}.{expiry}"
        return f"{user.username}.{expiry}.{self._sign(body)}"

    def verify_token(self, token: str, now: float | None = None) -> User | None:
        self._reload_if_changed()
        try:
            username, expiry, signature = token.split(".")
        except ValueError:
            return None
        user = self.users.get(username)
        if user is None:
            return None
        body = f"{username}.{user.role}.{expiry}"
        if not hmac.compare_digest(self._sign(body), signature):
            return None
        return user if int(expiry) > (time.time() if now is None else now) else None

    def _sign(self, body: str) -> str:
        return hmac.new(self.secret.encode(), body.encode(), "sha256").hexdigest()


def require_user(request: Request) -> User:
    """FastAPI dependency: 401 unless the request carries a valid ``space-auth``."""
    store: AuthStore = request.app.state.auth
    token = request.cookies.get(COOKIE, "")
    user = store.verify_token(token) if token else None
    if user is None:
        raise HTTPException(status_code=401, detail="No Auth")
    request.state.user = user
    return user


def require_admin(request: Request) -> None:
    """FastAPI dependency: 403 unless the signed-in user is an admin."""
    require_user(request)
    if request.state.user.role != "admin":
        raise HTTPException(status_code=403, detail="admin only")
