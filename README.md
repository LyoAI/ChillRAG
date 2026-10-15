# ChillRAG

> A document RAG service — drop in a folder, ask questions across it.

Retrieval fuses three independent searchers — dense (**faiss**), lexical
(**BM25**) and literal (**grep**) — and answers come from an agent that can walk
a document's own structure before it commits to one.

The HTTP interface mirrors the hosted RAG vendor this service replaces, so an
existing client changes its base URL and nothing else.

```mermaid
flowchart LR
  C([client]) -->|"space-auth cookie"| S[FastAPI]
  S -->|upload| I["parse → chunk → embed → index"]
  S -->|question| A[agent]
  A --> D[faiss]
  A --> B[BM25]
  A --> G[grep]
```

## Quickstart

```bash
uv sync                                            # install; or use your own venv
source .venv/bin/activate                          # so `python` below is that venv

python -m app.users add alice --role admin         # prints a password
python -m uvicorn app.main:create_app --factory --workers 1 --host 0.0.0.0 --port 9022
```

Keep `--workers 1`: the source registry, ingest status and the embedder are all
per process.

Run every command below **from the repository root, with the virtualenv active**.
`chillrag` is installed in editable mode; `app` is not installed at all — it is
found because the working directory is on `sys.path`. From any other directory
uvicorn stops with `Could not import module "app.main"`. Without activating
anything, the equivalent is:

```bash
.venv/bin/python -m uvicorn app.main:create_app --factory --workers 1 --host 0.0.0.0 --port 9022
```

## Getting documents in

Two roles, one destination. Source files live under `CHILLRAG_MATERIALS_DIR`
(`materials/<folder>/`), everything derived from them under `CHILLRAG_CACHE_DIR`
(`.cache/<folder>/`). Both routes register the folder as a *source*, which is what
the web UI's corpus selector reads — never the materials directory itself.

Ingest is incremental: a document whose name, size and mtime are unchanged since
the last run is not parsed again.

**Developer — shell and a full checkout.** Drop the files in
`materials/<folder>/` and run the CLI:

```bash
python -m app.ingest <folder>            # one or more folders
python -m app.ingest --all               # every folder under materials/
python -m app.ingest <folder> --rebuild  # discard the derived cache, redo
```

Each run prints a per-folder summary — document and chunk counts, elapsed time,
the derived `sourceId` and the two directories — and the folder shows up in the
web UI immediately. Prefer stopping the service first: the command loads its own
copy of the embedding model, and on a single GPU that is memory the running
service is also using.

**User — a browser and nothing else.** Visit `/upload`, name a corpus (or pick an
existing one) and drop files in. Any signed-in account may name a corpus and have
it created on the spot; the vendor's own `PUT /resource` stays admin-only. The
page uploads, then polls until parsing finishes and reports the passages each
file produced.

Limits worth knowing:

- File types are limited to the parser's supported set, which the upload page
  reads from the service rather than hardcoding. `.txt` is **not** among them
  and is refused with a 400.
- A file's id is derived from its **name**, so two documents with the same name
  in different folders collide: the later one overwrites the earlier one's
  registry record. Rename before uploading.
- Don't ingest the same folder from the shell *and* the web page at once. The
  per-folder lock is in-process, and those are two processes writing the same
  cache directory.

## Accounts

Accounts are a gitignored `users.yaml` at the repo root, managed with the CLI:

```bash
python -m app.users add alice --role admin    # or --password to choose one
python -m app.users list
python -m app.users passwd alice
python -m app.users remove alice
```

`admin` manages sources; `user` asks and retrieves. The file is plain text, so
editing it by hand works too — either way the running service picks the change up
at the next login, with no restart:

```yaml
users:
  alice:
    password: "change-me"     # quote it if it contains # $ @ etc.
    role: admin
  bob:
    password: "..."
    role: user
```

Without the file, a single admin account falls back to `CHILLRAG_USER` /
`CHILLRAG_PASSWORD`. Tokens are signed with `CHILLRAG_AUTH_SECRET` (or a secret
generated once under the cache dir), so sessions survive a restart. A new
password does not end sessions already issued — they run 12 h; removing the
account, or changing the secret, does.

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

The remaining knobs are documented in `chillrag/config.py`.

## Endpoints

Everything except `/healthz` and login needs the `space-auth` cookie.

| Method and path | Purpose |
|---|---|
| `POST /space/api/passport/login` | Issue `space-auth` |
| `PUT /space/api/service/asset-file/resource` | Create a source (folder) |
| `POST /space/api/service/asset-file/web-upload?sourceId=` | Upload files; ingest runs in the background |
| `POST /space/api/service/asset-file` | Ingest status for a list of file ids |
| `POST /space/api/service/llmapi/ht/normal_rag_recall_without_doc_group` | Retrieval; parameters travel in the query string |
| `GET /space/api/service/asset-file/segments/{file_id}` | Whole-document readback |
| `POST /space/api/service/sse/llmapi/ht/text/post-answer` | Answer, streamed as SSE |

Responses use the envelope `{"code", "msg", "data"}`.

Three pages sit outside the protocol: `/` asks questions over a corpus, `/upload`
adds documents to one, and `/api-test` fires each endpoint above from a card —
the quickest way to see what a request and its response actually look like.

The pages are served by the same service and need the same cookie, but their
endpoints are not part of the vendor protocol — they exist for this UI, which is
why they live under `/api/` and not under `/space/api/`:

| Method and path | Purpose |
|---|---|
| `GET /upload` | The upload page |
| `POST /api/upload?folder=` or `?sourceId=` | Multipart upload; `folder` creates a corpus, `sourceId` appends to one |
| `GET /api/upload/status?sourceId=` | Per-file passage counts while ingest runs |
| `GET /api/ui/config` | The parser's supported suffixes, served so the page cannot drift from the backend |
| `GET /space/api/service/asset-file/resource` | Lists sources; added for this UI, also fills the corpus selector |

## Where it differs from the vendor

- `recall_content_types` is a union in the order given, so `["excel", "text"]`
  returns both, tables first. The vendor honours only the first type.
- `normal_rag_recall_without_doc_group` requires `source_ids`. The vendor falls
  back to a default source; this service has none.
- `post-answer` ignores `mode`.

## Layout

```
app/          HTTP layer — vendor-shaped routes, auth, background ingest
app/static/   the three pages, their shared CSS/JS and the vendored markdown libs
app/ingest.py the developer CLI that ingests a folder from the shell
chillrag/     the core — parser, chunker, store, searchers, agent, LLM clients
tests/        contract tests, one per endpoint
```

## Test

```bash
python -m pytest
```
