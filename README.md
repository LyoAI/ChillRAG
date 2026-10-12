# ChillRAG

A document RAG service: ingest a folder of documents, then answer questions
over it. Retrieval combines dense (faiss), lexical (BM25) and literal (grep)
search. Answers come from an agent that can navigate document structure.

The HTTP interface mirrors the hosted RAG vendor it replaces, so existing
clients only change their base URL.

## Run

```bash
uv sync
# accounts: create users.yaml (gitignored) — see Accounts below
uv run uvicorn app.main:create_app --factory --workers 1 --host 0.0.0.0 --port 9022
```

Keep `--workers 1`. The source registry, ingest status and embedder are per
process.

## Accounts

Login uses a gitignored `users.yaml` at the repo root:

```yaml
users:
  alice:
    password: "change-me"     # single-quote if it has # $ @ etc.
    role: admin               # admin manages sources; user asks and retrieves
  bob:
    password: "..."
    role: user
```

Edits take effect on restart. Without the file, a single admin account falls
back to `CHILLRAG_USER` / `CHILLRAG_PASSWORD`. Tokens are signed with
`CHILLRAG_AUTH_SECRET`, or a secret auto-generated under the cache dir, so
restarts keep sessions.

## Configure

| Variable | Purpose |
|---|---|
| `CHILLRAG_USERS_FILE` | Accounts file (default `users.yaml`) |
| `CHILLRAG_USER`, `CHILLRAG_PASSWORD` | Fallback single account when no users file |
| `CHILLRAG_AUTH_SECRET` | Token signing key (auto-generated if unset) |
| `CHILLRAG_MATERIALS_DIR` | Where uploaded folders are written |
| `CHILLRAG_CACHE_DIR` | Parsed documents, indexes and the service registry |
| `CHILLRAG_LLM_PROVIDER`, `CHILLRAG_MODEL`, `CHILLRAG_LLM_CONFIG_DIR` | Answer model |
| `EMBED_MODEL`, `EMBED_DEVICE` | Embedding model and device |

Other knobs are documented in `chillrag/config.py`.

## Endpoints

All paths except `/healthz` and login need the `space-auth` cookie.

| Method and path | Purpose |
|---|---|
| `POST /space/api/passport/login` | Issue `space-auth` |
| `PUT /space/api/service/asset-file/resource` | Create a source (folder) |
| `POST /space/api/service/asset-file/web-upload?sourceId=` | Upload files; ingest runs in the background |
| `POST /space/api/service/asset-file` | Ingest status for a list of file ids |
| `POST /space/api/service/llmapi/ht/normal_rag_recall_without_doc_group` | Retrieval; parameters in the query string |
| `GET /space/api/service/asset-file/segments/{file_id}` | Whole-document readback |
| `POST /space/api/service/sse/llmapi/ht/text/post-answer` | Answer, streamed as SSE |

Responses use the envelope `{"code", "msg", "data"}`.

## Behaviour that differs from the vendor

- `recall_content_types` is a union in the order given, so `["excel", "text"]`
  returns both types with tables ranked first. The vendor honours only the
  first type.
- `normal_rag_recall_without_doc_group` requires `source_ids`. The vendor falls
  back to a default source; this service has none.
- `post-answer` ignores `mode`.

## Test

```bash
uv run pytest
```
