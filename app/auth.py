"""Username + password auth: login issues ``space-auth``, every other route checks it.

The token is ``<userId>.<expiry>.<hmac>``, signed with a key derived from the
password — so the password is the only credential and there is no separate
secret to configure. No session table: verification needs only the password,
and issued tokens stay valid across restarts until they expire.
"""

import hashlib
import hmac
import os
import time
from dataclasses import dataclass

from fastapi import HTTPException, Request

COOKIE = "space-auth"
TOKEN_TTL_S = 12 * 3600


@dataclass(frozen=True)
class AuthConfig:
    username: str
    password: str
    user_id: int = 1

    @property
    def _key(self) -> bytes:
        """The token-signing key, derived from the password."""
        return hashlib.sha256(self.password.encode()).digest()

    @classmethod
    def from_env(cls) -> "AuthConfig":
        try:
            return cls(
                username=os.environ["CHILLRAG_USER"],
                password=os.environ["CHILLRAG_PASSWORD"],
            )
        except KeyError as missing:
            raise RuntimeError(f"auth is not configured: set {missing.args[0]}") from None

    def check_password(self, password: str) -> bool:
        return hmac.compare_digest(password.encode(), self.password.encode())

    def issue_token(self, now: float | None = None) -> str:
        expiry = int(_now(now) + TOKEN_TTL_S)
        body = f"{self.user_id}.{expiry}"
        return f"{body}.{self._sign(body)}"

    def verify_token(self, token: str, now: float | None = None) -> bool:
        try:
            user_id, expiry, signature = token.split(".")
        except ValueError:
            return False
        body = f"{user_id}.{expiry}"
        if not hmac.compare_digest(self._sign(body), signature):
            return False
        return int(expiry) > _now(now)

    def _sign(self, body: str) -> str:
        return hmac.new(self._key, body.encode(), hashlib.sha256).hexdigest()


def _now(now: float | None) -> float:
    # ``now=0`` is a real time (tests use it), so compare against None, not falsiness.
    return time.time() if now is None else now


def require_user(request: Request) -> None:
    """FastAPI dependency: 401 unless the request carries a valid ``space-auth``."""
    config: AuthConfig = request.app.state.auth
    token = request.cookies.get(COOKIE, "")
    if not token or not config.verify_token(token):
        raise HTTPException(status_code=401, detail="No Auth")
