"""Resolve vendor source ids to an attached ``RAGTool``, refusing folders mid-ingest."""

from fastapi import HTTPException

from app import jobs, registry
from chillrag.rag_tool import RAGTool


def folders_of(source_ids: list[int]) -> list[str]:
    folders = []
    for source_id in source_ids:
        folder = registry.folder_of(source_id)
        if folder is None:
            raise HTTPException(status_code=404, detail=f"unknown source {source_id}")
        if folder not in folders:
            folders.append(folder)
    return folders


def attached_tool(source_ids: list[int]) -> RAGTool:
    """A ``RAGTool`` with every folder of ``source_ids`` attached.

    409 while a folder is still ingesting: attaching it would wait on the ingest
    lock, and the caller is better served by retrying after the status is finish.
    """
    tool = RAGTool()
    for folder in folders_of(source_ids):
        if jobs.status(folder) == "processing":
            raise HTTPException(status_code=409, detail=f"folder {folder!r} is still ingesting")
        tool.upload_file_complete(folder)
    return tool
