"""FastAPI application: vendor-compatible RAG endpoints behind ``space-auth``.

Run with a single worker (``uvicorn app.main:create_app --factory --workers 1``): the registry
and ingest status are per process, and the embedder is loaded once per process.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app import auth
from app.responses import fail
from app.routes import answer, asset_file, passport, recall, ui
from chillrag.corpus import CorpusManager


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Build the process-wide manager now, so the first request does not pay for
    # it. The embedder itself still loads lazily on first use (it is ~GB-scale).
    CorpusManager.instance()
    yield


def create_app(auth_config: auth.AuthConfig | None = None) -> FastAPI:
    app = FastAPI(title="ChillRAG", version="0.1.0", lifespan=lifespan)
    app.state.auth = auth_config or auth.AuthConfig.from_env()

    @app.exception_handler(HTTPException)
    async def _http_error(request: Request, exc: HTTPException) -> JSONResponse:
        # Same envelope as success responses, so clients parse one shape.
        return fail(exc.status_code, str(exc.detail))

    for router in (passport.router, asset_file.router, recall.router, answer.router, ui.router):
        app.include_router(router)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    return app
