"""Local single-user API for the MemBukkit GUI (`membukkit ui`).

Serves the local stores under ``~/.membukkit/stores`` plus the built React
frontend. This is deliberately separate from ``service.app`` (the multi-tenant
Turbopuffer service): here everything is on-disk, single-user, and optimized
for the explainability views — upload, facts browser, provenance drill-down,
ask-with-trace.
"""

import json
import logging
import os
import queue
import subprocess
import sys
import tempfile
import threading
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


def _ndjson_progress_stream(work: Callable) -> "object":
    """Run ``work(on_progress) -> result_dict`` on a worker thread; stream NDJSON.

    Emits ``{"type":"progress", ...}`` lines as ProgressEvents arrive, then a
    final ``{"type":"result", ...}`` (or ``{"type":"error", "detail":...}``).
    """
    from fastapi.responses import StreamingResponse

    from membukkit.progress import ProgressEvent

    q: queue.Queue = queue.Queue()

    def on_progress(ev: ProgressEvent) -> None:
        q.put(("progress", ev.to_dict()))

    def worker() -> None:
        try:
            result = work(on_progress)
            q.put(("result", result if isinstance(result, dict) else {"value": result}))
        except Exception as e:
            logger.exception("streamed work failed")
            q.put(("error", str(e)))
        finally:
            q.put(None)

    def gen() -> Iterator[str]:
        threading.Thread(target=worker, daemon=True).start()
        while True:
            item = q.get()
            if item is None:
                break
            kind, payload = item
            if kind == "progress":
                yield json.dumps({"type": "progress", **payload}) + "\n"
            elif kind == "result":
                yield json.dumps({"type": "result", **payload}) + "\n"
            else:
                yield json.dumps({"type": "error", "detail": payload}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")

# --------------------------------------------------------------------- bench
# Benchmark repro runs launched from the GUI. The registry is in-memory only:
# a server restart forgets every run started this session (the subprocesses
# themselves keep going, and their run.log + summary artifacts stay on disk
# under results/bench/<id>) — acceptable for a local single-user tool.
_BENCH_RUNS: Dict[str, Dict] = {}

_BENCH_LOG_TAIL_LINES = 50


def _bench_root() -> Path:
    """Working directory for bench subprocesses.

    Recipes write to paths relative to the invocation directory
    (``results/bench/<id>``, ``runs/distill_cache_*.json``), matching the CLI
    convention of running from the repo checkout root. Fall back to the
    current working directory for non-checkout installs.
    """
    root = Path(__file__).resolve().parents[3]
    return root if (root / "pyproject.toml").exists() else Path.cwd()


def _bench_command(recipe_id: str, lite: bool) -> List[str]:
    cmd = [sys.executable, "-m", "membukkit.cli", "bench", "--repro", recipe_id, "--yes"]
    if lite:
        cmd.append("--lite")
    return cmd


class _StoreHub:
    """Lazily opened MemorySystems for each local store, models shared."""

    def __init__(self, llm: str):
        self._llm = llm
        self._systems: Dict[str, object] = {}

    def open(self, name: str, create: bool = False):
        from membukkit.cli.common import _autoscale_budget, build_system
        from membukkit.config import PromptConfig
        from membukkit.storage.localstore import LocalStore

        store = LocalStore(name, create=create)
        mem = self._systems.get(name)
        if mem is None:
            spec = store.meta().get("encoder")
            if not spec:
                from membukkit.cli.common import DEFAULT_ENCODER

                spec = DEFAULT_ENCODER
                store.update_meta(encoder=spec)
            prompts = None
            if store.meta().get("prompts"):
                prompts = PromptConfig.from_dict(store.meta().get("prompts"))
            mem = build_system(llm=self._llm, encoder_spec=spec, prompts=prompts)
            store.load_backend(mem.backend)
            self._systems[name] = mem
        _autoscale_budget(mem)
        return mem, store

    def drop(self, name: str) -> None:
        self._systems.pop(name, None)


def create_local_app(ui_dist: Optional[Path] = None, llm: str = "openai:gpt-4o-mini"):
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field

    from membukkit.credentials import bootstrap_credentials, key_status, set_keys
    from membukkit.storage.localstore import LocalStore, list_stores

    bootstrap_credentials()

    hub = _StoreHub(llm)
    app = FastAPI(title="MemBukkit Local", version="1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _open(name: str, create: bool = False):
        try:
            return hub.open(name, create=create)
        except FileNotFoundError:
            raise HTTPException(404, f"store {name!r} not found")
        except ValueError as e:
            raise HTTPException(400, str(e))

    # ------------------------------------------------------------------ stores
    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    # ----------------------------------------------------------- credentials
    class KeysBody(BaseModel):
        openai_api_key: Optional[str] = Field(default=None)
        anthropic_api_key: Optional[str] = Field(default=None)
        gemini_api_key: Optional[str] = Field(default=None)
        ollama_host: Optional[str] = Field(default=None)
        persist: bool = True

    @app.get("/api/settings/keys")
    def get_keys():
        return key_status(llm)

    @app.put("/api/settings/keys")
    def put_keys(body: KeysBody):
        applied, path = set_keys(
            openai_api_key=body.openai_api_key,
            anthropic_api_key=body.anthropic_api_key,
            gemini_api_key=body.gemini_api_key,
            ollama_host=body.ollama_host,
            persist=body.persist,
        )
        status = key_status(llm)
        status["applied"] = applied
        status["persisted_to"] = path
        return status

    @app.get("/api/stores")
    def stores():
        return {"stores": list_stores()}

    @app.post("/api/stores/{name}")
    def create_store(name: str):
        try:
            LocalStore(name, create=True)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"name": name, "created": True}

    @app.delete("/api/stores/{name}")
    def delete_store(name: str):
        try:
            LocalStore(name, create=False).delete()
        except FileNotFoundError:
            raise HTTPException(404, f"store {name!r} not found")
        hub.drop(name)
        return {"deleted": name}

    @app.get("/api/stores/{name}/overview")
    def overview(name: str):
        mem, store = _open(name)
        latest_fn = getattr(mem.backend, "latest_fact_date", None)
        latest_fact_date = latest_fn() if callable(latest_fn) else None
        meta = store.meta()
        totals = meta.get("usage_totals") if isinstance(meta.get("usage_totals"), dict) else {}
        suggested = meta.get("suggested_questions") or []
        if not isinstance(suggested, list):
            suggested = []
        suggested = [str(q) for q in suggested if isinstance(q, str) and q.strip()]
        return {
            "name": name,
            "n_facts": mem.backend.count(),
            "n_verbatim": mem.backend.count_kind("verbatim"),
            "n_atomic": mem.backend.count_kind("atomic"),
            "latest_fact_date": latest_fact_date,
            "documents": store.documents(),
            "meta": meta,
            "usage_totals": totals or None,
            "est_lifetime_cost_usd": totals.get("est_cost_usd") if totals else None,
            "suggested_questions": suggested,
        }

    # ------------------------------------------------------------------- demos
    @app.get("/api/demos")
    def list_demos():
        from membukkit.cli.demo import available_demos

        demos = available_demos()
        return {
            "demos": [
                {
                    "id": name,
                    "title": m.get("title", name),
                    "description": m.get("description", ""),
                    "proves": m.get("proves") or "",
                    "question_date": m.get("question_date") or None,
                    "prompt_pack": m.get("prompt_pack") or None,
                    "questions": list(m.get("questions") or []),
                    "ask_callouts": list(m.get("ask_callouts") or []),
                    "prove_beats": list(m.get("prove_beats") or []),
                }
                for name, m in demos.items()
            ]
        }

    @app.post("/api/demos/{name}")
    def load_demo(name: str, stream: bool = False):
        from membukkit.cli.demo import available_demos, ensure_demo_store

        demos = available_demos()
        if name not in demos:
            raise HTTPException(404, f"unknown demo {name!r}")
        m = demos[name]

        def _finish(store_name: str) -> dict:
            hub.drop(store_name)
            return {
                "store": store_name,
                "id": name,
                "title": m.get("title", name),
                "questions": list(m.get("questions") or []),
                "proves": m.get("proves") or "",
                "question_date": m.get("question_date") or None,
                "ask_callouts": list(m.get("ask_callouts") or []),
                "prove_beats": list(m.get("prove_beats") or []),
            }

        if stream:
            def work(on_progress):
                try:
                    store_name = ensure_demo_store(name, llm=llm, on_progress=on_progress)
                except ValueError as e:
                    raise RuntimeError(str(e)) from e
                return _finish(store_name)

            return _ndjson_progress_stream(work)

        try:
            store_name = ensure_demo_store(name, llm=llm)
        except ValueError as e:
            raise HTTPException(404, str(e))
        except Exception as e:
            logger.exception("demo load failed for %s", name)
            raise HTTPException(500, f"demo load failed: {e}") from e
        return _finish(store_name)

    # ------------------------------------------------------------------ upload
    @app.post("/api/stores/{name}/upload")
    async def upload(name: str, files: list[UploadFile] = File(...), stream: bool = False):
        from membukkit.ingest import parse_file

        # Read uploads to temp files first so the response can stream sync work.
        prepared = []
        for f in files:
            suffix = Path(f.filename or "upload.txt").suffix or ".txt"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(await f.read())
                prepared.append((f.filename or "upload.txt", Path(tmp.name)))

        def _run(on_progress=None):
            from membukkit.cli.common import record_store_usage
            from membukkit.usage import TokenUsage, estimate_cost_usd, format_cost

            mem, store = _open(name, create=True)
            results = []
            usage_acc = TokenUsage()
            n_superseded = 0
            try:
                for filename, tmp_path in prepared:
                    try:
                        doc = parse_file(tmp_path)
                        doc.name = filename or doc.name
                        doc_id = store.add_document(
                            doc.name, doc.sessions, doc.dates, doc_type=doc.doc_type
                        )
                        report = mem.ingest(
                            doc.sessions,
                            dates=doc.dates,
                            doc_id=doc_id,
                            doc_name=doc.name,
                            doc_type=doc.doc_type,
                            on_progress=on_progress,
                        )
                        n_new = int(report)
                        n_superseded += len(report.superseded)
                        usage_acc.merge(TokenUsage.from_dict(report.usage))
                        record_store_usage(store, report, llm)
                        entry = {
                            "file": doc.name,
                            "doc_id": doc_id,
                            "doc_type": doc.doc_type,
                            "sessions": len(doc.sessions),
                            "new_facts": n_new,
                            "superseded": len(report.superseded),
                            "usage": report.usage,
                            "est_cost_usd": report.est_cost_usd,
                            "write": report.to_dict(),
                        }
                        if report.status == "empty_extract":
                            entry["warning"] = "empty_extract — nothing stored from this file"
                        results.append(entry)
                    except (ValueError, ImportError) as e:
                        results.append({"file": filename, "error": str(e)})
            finally:
                for _, tmp_path in prepared:
                    tmp_path.unlink(missing_ok=True)
            store.save_backend(mem.backend)
            if any(r.get("new_facts") for r in results):
                store.update_meta(bucket_labels={})
            suggested: list = []
            if sum(int(r.get("new_facts") or 0) for r in results) > 0:
                from membukkit.suggestions import refresh_store_suggestions

                suggested = refresh_store_suggestions(store, mem, llm_spec=llm)
            batch_cost = estimate_cost_usd(usage_acc, llm)
            return {
                "results": results,
                "n_facts": mem.backend.count(),
                "suggested_questions": suggested,
                "receipt": {
                    "files": len(results),
                    "new_facts": sum(int(r.get("new_facts") or 0) for r in results),
                    "superseded": n_superseded,
                    "usage": usage_acc.to_dict() if usage_acc.total_tokens else None,
                    "est_cost_usd": batch_cost,
                    "est_cost_label": format_cost(batch_cost) if batch_cost is not None else None,
                    "note": "one-time index cost",
                },
            }

        if stream:
            return _ndjson_progress_stream(_run)
        return _run()

    # --------------------------------------------------------------------- ask
    class AskRequest(BaseModel):
        question: str
        top_k: Optional[int] = None
        question_date: Optional[str] = None
        include_history: bool = True

    @app.post("/api/stores/{name}/ask")
    def ask(name: str, req: AskRequest):
        mem, store = _open(name)
        if mem.backend.count() == 0:
            raise HTTPException(400, "store is empty — upload some files first")
        q_date = (req.question_date or "").strip() or date.today().isoformat()
        # Evidence may include superseded facts (badges); the reader stays active-as-of.
        search = mem.search(
            req.question,
            top_k=req.top_k,
            question_date=q_date,
            include_history=req.include_history,
        )
        result = mem.answer(req.question, question_date=q_date, include_history=False)
        t = result.trace
        from membukkit.cli.common import record_store_usage

        record_store_usage(store, t, llm)
        # Attach cached topic labels so the GUI can narrate opened buckets.
        meta = store.meta()
        bucket_labels = {
            str(k): v for k, v in (meta.get("bucket_labels") or {}).items()
        }
        return {
            "answer": result.answer,
            "question_date": q_date,
            "trace": {
                "scan_fraction": t.scan_fraction,
                "n_facts": t.n_facts,
                "n_scanned": t.n_scanned,
                "est_reader_tokens": t.est_reader_tokens,
                "reader_type": t.reader_type,
                "lanes": t.lanes,
                "opened_buckets": t.opened_buckets,
                "bucket_labels": bucket_labels,
                "bucket_labels_lane": meta.get("bucket_labels_lane"),
                "usage": getattr(t, "usage", None),
                "est_cost_usd": getattr(t, "est_cost_usd", None),
                "window_fraction": getattr(t, "window_fraction", 0) or 0,
                "model": getattr(t, "model", "") or "",
            },
            "evidence": [
                {
                    "ref": h.ref,
                    "fact": h.fact,
                    "text": h.text,
                    "timestamp": h.timestamp,
                    "fact_id": h.source_id,
                    "doc_id": h.doc_id,
                    "doc_name": h.doc_name,
                    "source_ref": h.source_ref,
                    "kind": h.kind or "",
                    "status": h.status or "current",
                    "superseded_by": h.superseded_by or "",
                }
                for h in search.hits
            ],
        }

    # ------------------------------------------------------------------- facts
    @app.get("/api/stores/{name}/facts")
    def facts(
        name: str,
        offset: int = 0,
        limit: int = 50,
        kind: Optional[str] = None,
        bucket: Optional[int] = None,
    ):
        mem, _ = _open(name)
        return mem.backend.facts_page(
            offset=offset, limit=min(limit, 200), kind=kind, bucket=bucket
        )

    @app.get("/api/stores/{name}/facts/{fact_id}/source")
    def fact_source(name: str, fact_id: str):
        mem, store = _open(name)
        fact = mem.backend.get_fact(fact_id)
        if fact is None:
            raise HTTPException(404, "fact not found")
        source = None
        if fact.get("doc_id"):
            # fact_text drives the lexical fallback: legacy atomic facts carry
            # only "session:N" refs, so the best-matching turn is picked at
            # resolve time from the fact's own text.
            source = store.resolve_source(
                fact["doc_id"], fact.get("source_ref", ""), fact_text=fact.get("text", "")
            )
        return {"fact": fact, "source": source}

    @app.delete("/api/stores/{name}/facts/{fact_id}")
    def delete_fact(name: str, fact_id: str, purge_source: bool = False):
        """Erase one memory and persist.

        Goes through `MemorySystem.delete_facts`, so the verbatim turn behind
        the fact goes too (unless another fact still needs it) and anything the
        fact had superseded becomes current again. Invalidates cached bucket
        labels — the topic partition is rebuilt lazily on the next map view.
        """
        mem, store = _open(name)
        report = mem.delete_facts([fact_id], purge_source=purge_source)
        if report["deleted"] == 0:
            raise HTTPException(404, "fact not found")
        store.save_backend(mem.backend)
        store.update_meta(bucket_labels={}, bucket_labels_lane=None)
        return {
            "deleted": fact_id,
            "rows_removed": report["deleted"],
            "revived": report["revived"],
            "n_facts": mem.backend.count(),
            "n_verbatim": mem.backend.count_kind("verbatim"),
            "n_atomic": mem.backend.count_kind("atomic"),
        }

    @app.delete("/api/stores/{name}/documents/{doc_id}")
    def delete_document(name: str, doc_id: str):
        """Delete an uploaded document: registry row, raw source, and every
        fact ingested from it. Persists the backend and clears label caches."""
        mem, store = _open(name)
        removed = store.remove_document(doc_id)
        n_facts_removed = mem.backend.delete_doc_facts(doc_id)
        if not removed and n_facts_removed == 0:
            raise HTTPException(404, "document not found")
        store.save_backend(mem.backend)
        store.update_meta(bucket_labels={}, bucket_labels_lane=None)
        return {
            "deleted": doc_id,
            "facts_removed": n_facts_removed,
            "n_facts": mem.backend.count(),
        }

    # ----------------------------------------------------------------- distill
    @app.post("/api/stores/{name}/distill")
    def distill(name: str, stream: bool = False):
        """Force fact extraction over the store's preserved raw documents.

        Rescues verbatim-only stores (ingested with --no-distill or by an old
        build). Idempotent — existing facts dedupe, only missing distilled
        facts are added. Pass ``stream=1`` for NDJSON progress events.
        """
        from membukkit.cli.common import distill_store

        mem, store = _open(name)
        if not store.documents():
            raise HTTPException(400, "store has no preserved source documents to extract from")

        def _run(on_progress=None):
            try:
                n_new = distill_store(mem, store, on_progress=on_progress)
            except Exception as e:
                logger.exception("re-distillation failed for store %r", name)
                raise RuntimeError(f"fact extraction failed: {e}") from e
            if n_new:
                store.update_meta(bucket_labels={}, bucket_labels_lane=None)
            return {
                "new_facts": n_new,
                "n_facts": mem.backend.count(),
                "n_atomic": mem.backend.count_kind("atomic"),
            }

        if stream:
            return _ndjson_progress_stream(_run)
        try:
            return _run()
        except RuntimeError as e:
            raise HTTPException(502, str(e)) from e

    # --------------------------------------------------------------- partition
    @app.get("/api/stores/{name}/partition")
    def partition(
        name: str, label: bool = False, refresh: bool = False, stream: bool = False
    ):
        from collections import Counter

        mem, store = _open(name)
        # The map is over the atomic (distilled) lane — those dated facts are
        # the user-facing product. Verbatim is the fallback for stores without
        # distillation. Bucket ids in the response are lane-local, matching
        # `facts?bucket=&kind=` filtering.
        lane = "atomic" if mem.backend.count_kind("atomic") > 0 else "verbatim"
        view = mem.backend.lane_view(lane) or {}
        k_eff = int(view.get("k_eff", 0))
        sizes = Counter(view.get("labels", []))

        meta = store.meta()
        labels: Dict[int, str] = {}
        # Cached labels are only valid for the lane they were generated on
        # (legacy caches from the old mixed-lane partition have no lane and
        # are ignored, so relabeling is never blocked by stale entries).
        if not refresh and meta.get("bucket_labels_lane") == lane:
            labels = {int(k): v for k, v in (meta.get("bucket_labels") or {}).items()}

        def _view(lbls: Dict[int, str]) -> dict:
            return {
                "k_eff": k_eff,
                "lane": lane,
                "n_facts": mem.backend.count_kind(lane),
                "buckets": [
                    {
                        "bucket": b,
                        "size": sizes.get(b, 0),
                        "label": lbls.get(b, ""),
                        "exemplars": mem.backend.topic_exemplars(b, n=3, kind=lane),
                    }
                    for b in range(k_eff)
                ],
            }

        need_label = bool(label and k_eff and (refresh or not labels))
        if need_label and stream:

            def work(on_progress):
                try:
                    lbls = mem.label_buckets(kind=lane, on_progress=on_progress)
                except Exception as e:
                    logger.exception("bucket labeling failed for store %r", name)
                    raise RuntimeError(f"bucket labeling failed: {e}") from e
                if lbls:
                    store.update_meta(
                        bucket_labels={str(k): v for k, v in lbls.items()},
                        bucket_labels_lane=lane,
                    )
                return _view(lbls)

            return _ndjson_progress_stream(work)

        if need_label:
            try:
                labels = mem.label_buckets(kind=lane)
            except Exception as e:
                logger.exception("bucket labeling failed for store %r", name)
                raise HTTPException(502, f"bucket labeling failed: {e}")
            if labels:
                store.update_meta(
                    bucket_labels={str(k): v for k, v in labels.items()},
                    bucket_labels_lane=lane,
                )
        return _view(labels)

    @app.get("/api/stores/{name}/documents/{doc_id}")
    def document(name: str, doc_id: str):
        _, store = _open(name)
        content = store.document_content(doc_id)
        if content is None:
            raise HTTPException(404, "document not found")
        return content

    # ----------------------------------------------------------------- prompts
    from membukkit.config import PromptConfig
    from membukkit.prompts.packs import list_prompt_packs, load_prompt_pack
    from membukkit.prompts.resolve import PLACEHOLDERS, validate_prompt_config

    class PromptUpdate(BaseModel):
        extraction: Optional[str] = None
        extraction_named: Optional[str] = None
        extraction_document: Optional[str] = None
        dated_reader: Optional[str] = None
        recommendation_reader: Optional[str] = None
        reasoning_reader: Optional[str] = None
        abstain_gate: Optional[str] = None
        extraction_instructions: Optional[str] = None
        reader_instructions: Optional[str] = None

    class PackRequest(BaseModel):
        pack_id: str

    def _prompts_view(store) -> dict:
        meta = store.meta()
        cfg = PromptConfig.from_dict(meta.get("prompts"))
        return {
            "prompts": cfg.to_dict(),
            "pack_id": meta.get("prompt_pack") or None,
            "packs": list_prompt_packs(),
            "placeholders": {k: list(v) for k, v in PLACEHOLDERS.items()},
            "is_default": cfg.is_default(),
        }

    def _store_only(name: str, create: bool = False):
        try:
            return LocalStore(name, create=create)
        except FileNotFoundError:
            raise HTTPException(404, f"store {name!r} not found")
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.get("/api/stores/{name}/prompts")
    def get_prompts(name: str):
        return _prompts_view(_store_only(name))

    @app.put("/api/stores/{name}/prompts")
    def put_prompts(name: str, body: PromptUpdate):
        store = _store_only(name)
        cfg = PromptConfig.from_dict(body.model_dump())
        try:
            validate_prompt_config(cfg)
        except ValueError as e:
            raise HTTPException(400, str(e))
        store.update_meta(prompts=cfg.to_dict(), prompt_pack=None)
        hub.drop(name)
        return _prompts_view(store)

    @app.post("/api/stores/{name}/prompts/pack")
    def apply_prompt_pack(name: str, body: PackRequest):
        store = _store_only(name)
        try:
            cfg = load_prompt_pack(body.pack_id)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        store.update_meta(prompts=cfg.to_dict(), prompt_pack=body.pack_id)
        hub.drop(name)
        return _prompts_view(store)

    @app.post("/api/stores/{name}/prompts/reset")
    def reset_prompts(name: str):
        store = _store_only(name)
        store.update_meta(prompts=None, prompt_pack=None)
        hub.drop(name)
        return _prompts_view(store)

    # ------------------------------------------------------------------- bench
    from membukkit.bench.recipes import RECIPES, check_recipe_output
    from membukkit.cli.bench import _estimate

    class BenchRunRequest(BaseModel):
        recipe_id: str
        lite: bool = False

    @app.get("/api/bench/recipes")
    def bench_recipes():
        return {
            "recipes": [
                {
                    "id": r.id,
                    "title": r.title,
                    "dataset": r.dataset,
                    "description": r.description,
                    "reader": r.reader,
                    "distiller": r.distiller,
                    "judge": r.judge,
                    "encoder": r.encoder,
                    "expected": r.expected,
                    "metric": r.metric,
                    "tolerance": r.tolerance,
                    "env": [
                        {"name": var, "set": bool(os.environ.get(var))}
                        for var in r.required_env
                    ],
                    "cost_estimate": _estimate(r.estimate_key, r.distiller, 1.0),
                    "cli_command": f"membukkit bench --repro {r.id} --yes",
                }
                for r in RECIPES.values()
            ]
        }

    def _bench_progress(recipe_id: str) -> Optional[Dict]:
        recipe = RECIPES.get(recipe_id)
        if recipe is None:
            return None
        path = _bench_root() / recipe.output_subdir / "progress.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def _bench_run_view(run_id: str, run: Dict) -> Dict:
        rc = run["proc"].poll()
        status = "running" if rc is None else ("done" if rc == 0 else "failed")
        view = {
            "run_id": run_id,
            "recipe_id": run["recipe_id"],
            "lite": run["lite"],
            "status": status,
            "started_at": run["started_at"],
        }
        prog = _bench_progress(run["recipe_id"])
        if prog:
            view["progress"] = prog
        return view

    @app.post("/api/bench/runs")
    def bench_start(req: BenchRunRequest):
        recipe = RECIPES.get(req.recipe_id)
        if recipe is None:
            raise HTTPException(404, f"unknown recipe {req.recipe_id!r}")
        missing = [v for v in recipe.required_env if not os.environ.get(v)]
        if missing:
            raise HTTPException(
                400,
                "missing required environment variables: " + ", ".join(missing),
            )
        for rid, run in _BENCH_RUNS.items():
            if run["recipe_id"] == req.recipe_id and run["proc"].poll() is None:
                raise HTTPException(
                    409, f"recipe {req.recipe_id!r} already has a live run ({rid})"
                )

        root = _bench_root()
        out_dir = root / recipe.output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "run.log"

        env = dict(os.environ)
        # Recipe env defaults (e.g. the BEAM distiller turn cap) apply only
        # when the variable isn't already set — same rule as the CLI.
        for var, val in recipe.env_defaults.items():
            if not env.get(var):
                env[var] = val

        log_fh = open(log_path, "a", buffering=1)  # line-buffered append
        log_fh.write(
            f"=== bench run {datetime.now().isoformat(timespec='seconds')} "
            f"({'lite' if req.lite else 'full'}) ===\n"
        )
        try:
            proc = subprocess.Popen(
                _bench_command(req.recipe_id, req.lite),
                cwd=str(root),
                env=env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
        finally:
            log_fh.close()  # the child holds its own copy of the fd

        run_id = uuid.uuid4().hex[:12]
        _BENCH_RUNS[run_id] = {
            "recipe_id": req.recipe_id,
            "lite": req.lite,
            "pid": proc.pid,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "log_path": str(log_path),
            "proc": proc,
        }
        return {"run_id": run_id}

    @app.get("/api/bench/runs")
    def bench_runs():
        return {"runs": [_bench_run_view(rid, run) for rid, run in _BENCH_RUNS.items()]}

    @app.get("/api/bench/runs/{run_id}")
    def bench_run(run_id: str):
        run = _BENCH_RUNS.get(run_id)
        if run is None:
            raise HTTPException(
                404, "unknown run (the run registry is in-memory; restarts forget it)"
            )
        view = _bench_run_view(run_id, run)
        log_path = Path(run["log_path"])
        view["log_tail"] = (
            log_path.read_text(errors="replace").splitlines()[-_BENCH_LOG_TAIL_LINES:]
            if log_path.exists()
            else []
        )
        if view["status"] == "done":
            recipe = RECIPES[run["recipe_id"]]
            try:
                passed, measured, _ = check_recipe_output(
                    recipe, str(_bench_root() / recipe.output_subdir)
                )
            except (FileNotFoundError, KeyError, ValueError):
                passed, measured = None, None
            view["result"] = {
                "measured": measured,
                "expected": recipe.expected,
                "tolerance": recipe.tolerance,
                # Lite subsets are not comparable to the frozen full-run
                # numbers — a lite run is a smoke test, never PASS/FAIL.
                "passed": None if run["lite"] else passed,
                "smoke": run["lite"],
            }
        return view

    # -------------------------------------------------------- agent v1 API
    # Local agent surface over the same stores as the GUI; no cloud. `subject`
    # attributes written facts to a person; retrieval is not scoped by it, so
    # keep separate people in separate stores.
    class V1AddRequest(BaseModel):
        content: object  # str | message | list
        subject: str = ""
        date: Optional[str] = None

    class V1SearchRequest(BaseModel):
        query: str
        top_k: Optional[int] = None
        as_of: Optional[str] = None
        include_history: bool = False

    class V1AskRequest(BaseModel):
        query: str
        as_of: Optional[str] = None
        include_history: bool = False
        top_k: Optional[int] = None

    @app.post("/api/v1/{name}/add")
    def v1_add(name: str, req: V1AddRequest):
        from membukkit.cli.common import record_store_usage
        from membukkit.memory_api import Memory

        mem, store = _open(name, create=True)
        report = Memory.wrap(mem).add(
            req.content,  # type: ignore[arg-type]
            subject=req.subject,
            date=req.date,
        )
        store.save_backend(mem.backend)
        if report.n_stored:
            store.update_meta(bucket_labels={})
        record_store_usage(store, report, llm)
        body = report.to_dict()
        body["n_facts"] = mem.backend.count()
        if report.n_stored:
            from membukkit.suggestions import refresh_store_suggestions

            body["suggested_questions"] = refresh_store_suggestions(
                store, mem, llm_spec=llm
            )
        if report.status == "empty_extract":
            raise HTTPException(422, detail=body)
        return body

    @app.post("/api/v1/{name}/search")
    def v1_search(name: str, req: V1SearchRequest):
        from membukkit.memory_api import Memory

        mem, _ = _open(name)
        if mem.backend.count() == 0:
            raise HTTPException(400, "store is empty")
        res = Memory.wrap(mem).search(
            req.query,
            top_k=req.top_k,
            as_of=req.as_of,
            include_history=req.include_history,
        )
        return {
            "query": res.query,
            "hits": [
                {
                    "ref": h.ref,
                    "fact": h.fact,
                    "text": h.text,
                    "timestamp": h.timestamp,
                    "fact_id": h.source_id,
                    "kind": h.kind,
                    "status": h.status,
                    "superseded_by": h.superseded_by,
                    "source_ref": h.source_ref,
                    "doc_name": h.doc_name,
                }
                for h in res.hits
            ],
            "trace": {
                "scan_fraction": res.trace.scan_fraction,
                "n_facts": res.trace.n_facts,
                "n_scanned": res.trace.n_scanned,
                "est_reader_tokens": res.trace.est_reader_tokens,
                "reader_type": res.trace.reader_type,
                "lanes": res.trace.lanes,
            },
        }

    @app.post("/api/v1/{name}/ask")
    def v1_ask(name: str, req: V1AskRequest):
        from membukkit.memory_api import Memory

        mem, store = _open(name)
        if mem.backend.count() == 0:
            raise HTTPException(400, "store is empty")
        receipt = Memory.wrap(mem).ask(
            req.query,
            as_of=req.as_of,
            include_history=req.include_history,
            top_k=req.top_k,
        )
        meta = store.meta()
        out = receipt.to_dict()
        out["trace"]["bucket_labels"] = {
            str(k): v for k, v in (meta.get("bucket_labels") or {}).items()
        }
        return out

    # ---------------------------------------------------------------- frontend
    if ui_dist is not None:
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=str(ui_dist), html=True), name="ui")

    return app
