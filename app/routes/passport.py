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
    store: auth.AuthStore = request.app.state.auth
    user = store.verify(body.username, body.passwd)
    if user is None:
        return fail(401, "invalid username or password")
    token = store.issue_token(user)
    response = JSONResponse(ok({
        "token": token,
        "userId": store.user_id(user.username),  # the vendor client reads this
        "username": user.username,
        "role": user.role,
    }))
    response.set_cookie(auth.COOKIE, token, httponly=True, max_age=auth.TOKEN_TTL_S)
    return response
