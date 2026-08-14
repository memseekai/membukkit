"""FastAPI app exposing the multi-tenant memory service.

    uvicorn membukkit.service.app:app          # uses env-configured defaults
    membukkit serve --host 0.0.0.0 --port 8080

Each route is scoped to an owner; `{owner}` maps 1:1 to a Turbopuffer namespace
(`<prefix><owner>`, prefix `mem_` by default — configurable via
`ServiceConfig.namespace_prefix`). FastAPI/pydantic are imported lazily so
importing this module's helpers does not require them until `create_app()` is called.
"""

import logging
from typing import Any, Dict, List, Optional

from membukkit import telemetry
from membukkit.service.manager import MemoryService, ServiceConfig

logger = logging.getLogger(__name__)


def _trace_dict(trace) -> Dict[str, Any]:
    return {
        "opened_buckets": trace.opened_buckets,
        "scan_fraction": trace.scan_fraction,
        "n_facts": trace.n_facts,
        "n_scanned": trace.n_scanned,
        "est_reader_tokens": getattr(trace, "est_reader_tokens", 0),
        "k_total": trace.k_total,
        "reader_type": trace.reader_type,
        "ranked_fact_times": trace.ranked_fact_times,
        "backend": trace.backend,
        "perf": trace.perf,
    }


def create_app(service: Optional[MemoryService] = None, config: Optional[ServiceConfig] = None):
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field, field_validator
    from membukkit.time_utils import parse_datetime

    svc = service or MemoryService(config)
    app = FastAPI(title="MEMBUKKIT Memory Service", version="1.0")

    # Observability: configure once, then auto-instrument FastAPI + LLM clients +
    # httpx (the latter covers the Turbopuffer HTTP client). Safe no-op without logfire.
    cfg = getattr(svc, "config", None)
    if cfg is None or getattr(cfg, "telemetry", True):
        telemetry.configure(
            service_name="membukkit",
            environment=getattr(cfg, "environment", None),
            capture_content=getattr(cfg, "capture_content", False),
        )
        telemetry.instrument_fastapi(app)
        telemetry.instrument_llm()
        telemetry.instrument_httpx()
        if hasattr(svc, "register_metrics"):
            svc.register_metrics()

    class Turn(BaseModel):
        role: str = "user"
        content: str = ""

    class IngestRequest(BaseModel):
        sessions: List[List[Turn]]
        dates: Optional[List[Any]] = None
        subject: Optional[str] = None

        @field_validator("dates")
        @classmethod
        def _parse_dates(cls, value):
            if value is None:
                return None
            return [parse_datetime(v) for v in value]

    class AnswerRequest(BaseModel):
        question: str
        question_date: Optional[Any] = None
        identity: str = ""
        answer: bool = True  # generate the LLM answer (default on)
        trace: bool = False  # include the retrieval trace in the response (default off)

        @field_validator("question_date")
        @classmethod
        def _parse_question_date(cls, value):
            return parse_datetime(value)

    class AnswerResponse(BaseModel):
        answer: Optional[str] = None
        facts: List[str] = Field(default_factory=list)
        trace: Optional[Dict[str, Any]] = None

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/v1/{owner}/ingest")
    def ingest(owner: str, req: IngestRequest):
        try:
            mem = svc.get(owner)
            sessions = [[t.model_dump() for t in s] for s in req.sessions]
            mem.ingest(sessions=sessions, dates=req.dates, subject=req.subject)
            return {"ok": True, "n_facts": mem._backend.count()}
        except Exception as e:
            telemetry.counter("membukkit.errors").add(1, {"stage": "ingest"})
            logger.exception("ingest failed")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/{owner}/answer", response_model=AnswerResponse, response_model_exclude_none=True)
    def answer(owner: str, req: AnswerRequest):
        try:
            mem = svc.get(owner)
            res = mem.answer(
                req.question,
                question_date=req.question_date,
                identity=req.identity,
                generate_answer=req.answer,
            )
            return AnswerResponse(
                answer=res.answer,
                facts=res.facts,
                trace=_trace_dict(res.trace) if req.trace else None,
            )
        except Exception as e:
            telemetry.counter("membukkit.errors").add(1, {"stage": "answer"})
            logger.exception("answer failed")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/v1/{owner}/partition")
    def partition(owner: str):
        mem = svc.get(owner)
        part = mem.partition()
        return {
            "k_eff": part.get("k_eff", 0),
            "version": part.get("version", 0),
            "sizes": part.get("sizes", {}),
        }

    @app.post("/v1/{owner}/label_buckets")
    def label_buckets(owner: str):
        return svc.get(owner).label_buckets()

    @app.post("/v1/{owner}/warm")
    def warm(owner: str):
        svc.warm(owner)
        return {"ok": True}

    @app.post("/v1/{owner}/recluster")
    def recluster(owner: str):
        ran = svc.recluster(owner)
        return {"ok": True, "reclustered": ran}

    @app.delete("/v1/{owner}")
    def delete(owner: str):
        svc.delete(owner)
        return {"ok": True}

    return app


# Module-level app for `uvicorn membukkit.service.app:app` (env-configured).
def __getattr__(name: str):
    if name == "app":
        return create_app()
    raise AttributeError(name)
