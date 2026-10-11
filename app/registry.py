"""Maps the vendor's source and file ids back to folders and file names.

``source_id_for`` and ``file_id_for`` are one-way hashes, so the service keeps
the reverse mapping in ``<cache>/service_registry.json``. Writes go through a
temp file and ``os.replace`` so a crash never leaves a half-written registry.
"""

import json
import threading
from pathlib import Path

from chillrag.config import CACHE_DIR

_LOCK = threading.Lock()
_PATH = CACHE_DIR / "service_registry.json"


def _load() -> dict:
    if not _PATH.exists():
        return {"sources": {}, "files": {}}
    return json.loads(_PATH.read_text(encoding="utf-8"))


def _save(state: dict) -> None:
    _PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(_PATH)


def register_source(source_id: int, folder: str) -> None:
    with _LOCK:
        state = _load()
        state["sources"][str(source_id)] = folder
        _save(state)


def sources() -> list[tuple[int, str]]:
    """All registered ``(source_id, folder)`` pairs, ordered by folder name."""
    return sorted(
        ((int(sid), folder) for sid, folder in _load()["sources"].items()),
        key=lambda pair: pair[1],
    )


def folder_of(source_id: int) -> str | None:
    return _load()["sources"].get(str(source_id))


def register_file(file_id: int, source_id: int, name: str) -> None:
    with _LOCK:
        state = _load()
        state["files"][str(file_id)] = {"source": source_id, "name": name}
        _save(state)


def file_of(file_id: int) -> dict | None:
    """``{"source": int, "name": str}`` for a registered file, else None."""
    return _load()["files"].get(str(file_id))


def folder_of_file(file_id: int) -> str | None:
    record = file_of(file_id)
    return folder_of(record["source"]) if record else None
