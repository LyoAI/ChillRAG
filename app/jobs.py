"""Background ingest: one worker, one job per folder, status polled by clients.

Ingest is parse → chunk → embed → index, which takes minutes, so an upload only
writes the file and queues the folder. An upload arriving while the folder is
already ingesting marks it dirty: that pass ends with an incomplete file set, so
the job re-runs until it sees a stable folder.
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from chillrag.rag_tool import RAGTool

logger = logging.getLogger("chillrag.jobs")

# One worker: ingest is GPU/CPU-bound and the embedder is a single shared model.
_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")
_LOCK = threading.Lock()
_STATUS: dict[str, str] = {}  # folder -> "processing" | "finish" | "failed"
_DIRTY: set[str] = set()


def submit(folder: str) -> None:
    with _LOCK:
        if _STATUS.get(folder) == "processing":
            _DIRTY.add(folder)
            return
        _STATUS[folder] = "processing"
    _POOL.submit(_run, folder)


def _run(folder: str) -> None:
    try:
        while True:
            with _LOCK:
                _DIRTY.discard(folder)
            RAGTool().upload_file_complete(folder)
            with _LOCK:
                if folder not in _DIRTY:
                    break
        outcome = "finish"
    except Exception:
        logger.exception("ingest failed for folder %r", folder)
        outcome = "failed"
    with _LOCK:
        _STATUS[folder] = outcome


def status(folder: str) -> str:
    """Unknown folders report ``finish``: a restart forgets jobs, and a folder
    that was never queued in this process has nothing pending."""
    with _LOCK:
        return _STATUS.get(folder, "finish")
