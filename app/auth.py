"""Signed-token auth: login issues ``space-auth``, every other route checks it.

The token is ``<userId>.<expiry>.<hmac>``. No session table: verification needs
only the secret, so a restart keeps issued tokens valid until they expire.
"""

import hashlib
import hmac
import os
import secrets
import time
from dataclasses import dataclass

from fastapi import HTTPException, Request

COOKIE = "space-auth"
TOKEN_TTL_S = 12 * 3600
_ITERATIONS = 600_000


@dataclass(frozen=True)
class AuthConfig:
    username: str
    password_hash: str  # "pbkdf2_sha256$<iterations>$<salt hex>$<digest hex>"
    secret: str
    user_id: int = 1

    @classmethod
    def from_env(cls) -> "AuthConfig":
        try:
            return cls(
                username=os.environ["CHILLRAG_USER"],
                password_hash=os.environ["CHILLRAG_PASSWORD_HASH"],
                secret=os.environ["CHILLRAG_AUTH_SECRET"],
            )
        except KeyError as missing:
            raise RuntimeError(f"auth is not configured: set {missing.args[0]}") from None


def make_password_hash(password: str) -> str:
    """The value to store in ``CHILLRAG_PASSWORD_HASH`` for ``password``."""
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), _ITERATIONS)
    return f"pbkdf2_sha256${_ITERATIONS}${salt}${digest.hex()}"


def check_password(password: str, stored: str) -> bool:
    _, iterations, salt, digest = stored.split("$")
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), int(iterations))
    return hmac.compare_digest(candidate.hex(), digest)


def _sign(secret: str, body: str) -> str:
    return hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()


def _now(now: float | None) -> float:
    # ``now=0`` is a real time (tests use it), so compare against None, not falsiness.
    return time.time() if now is None else now


def issue_token(config: AuthConfig, now: float | None = None) -> str:
    expiry = int(_now(now) + TOKEN_TTL_S)
    body = f"{config.user_id}.{expiry}"
    return f"{body}.{_sign(config.secret, body)}"


def verify_token(config: AuthConfig, token: str, now: float | None = None) -> bool:
    try:
        user_id, expiry, signature = token.split(".")
    except ValueError:
        return False
    body = f"{user_id}.{expiry}"
    if not hmac.compare_digest(_sign(config.secret, body), signature):
        return False
    return int(expiry) > _now(now)


def require_user(request: Request) -> None:
    """FastAPI dependency: 401 unless the request carries a valid ``space-auth``."""
    config: AuthConfig = request.app.state.auth
    token = request.cookies.get(COOKIE, "")
    if not token or not verify_token(config, token):
        raise HTTPException(status_code=401, detail="No Auth")
