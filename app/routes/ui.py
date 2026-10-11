"""Web UI: a self-contained question-answering page served by the service.

Independent of the vendor-protocol routers — it adds only a page route and one
list-sources endpoint for its dropdown, and nothing else depends on them.
"""

from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from app import registry
from app.auth import require_user
from app.responses import ok

router = APIRouter()

_PAGE = Path(__file__).resolve().parent.parent / "static" / "index.html"


@router.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(_PAGE)


@router.get("/space/api/service/asset-file/resource", dependencies=[Depends(require_user)])
def list_sources():
    """UI-support extension (the vendor protocol has no list endpoint)."""
    return ok([{"sourceId": sid, "resourceName": folder} for sid, folder in registry.sources()])
