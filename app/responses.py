"""Response envelope shared by every endpoint: ``{"code", "msg", "data"}``."""

from typing import Any

from fastapi.responses import JSONResponse


def ok(data: Any) -> dict:
    return {"code": 0, "msg": "success", "data": data}


def fail(status: int, msg: str) -> JSONResponse:
    return JSONResponse({"code": status, "msg": msg, "data": None}, status_code=status)
