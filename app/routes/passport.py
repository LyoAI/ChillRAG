"""``POST /space/api/passport/login``: check credentials, issue ``space-auth``."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app import auth
from app.responses import fail, ok

router = APIRouter(prefix="/space/api/passport")


class LoginBody(BaseModel):
    username: str
    passwd: str


@router.post("/login")
def login(body: LoginBody, request: Request):
    config: auth.AuthConfig = request.app.state.auth
    if body.username != config.username or not config.check_password(body.passwd):
        return fail(401, "invalid username or password")
    token = config.issue_token()
    response = JSONResponse(ok({"token": token, "userId": config.user_id}))
    response.set_cookie(auth.COOKIE, token, httponly=True, max_age=auth.TOKEN_TTL_S)
    return response
