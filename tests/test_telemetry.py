"""Tests for the telemetry facade: no-op contract (in-process) + enabled path
(subprocess-isolated, because logfire.configure sets global OTel state)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap

import pytest

from membukkit import telemetry


# ---------------------------------------------------------- no-op contract
def test_noop_primitives_are_safe():
    sp = telemetry._NoopSpan()
    with sp:
        sp.set_attribute("a", 1)
        sp.set_attributes({"b": 2})
        sp.record_exception(ValueError("x"))
    inst = telemetry._Instrument()
    inst.add(1, {"k": "v"})
    inst.record(1.0)
    inst.set(3)


def test_facade_calls_are_safe_and_decorator_is_identity():
    # Whether or not telemetry is configured, these must never raise.
    with telemetry.span("x", a=1) as s:
        telemetry.set_attributes(s, b=2)
    telemetry.histogram("m.h").record(1.0, {"k": "v"})
    telemetry.counter("m.c").add(1)
    with telemetry.timed("y", telemetry.histogram("m.h2"), kind="facts", count=3):
        pass

    @telemetry.instrument()
    def f(secret):
        return "ok"

    assert f("sensitive") == "ok"


def test_metric_attrs_drops_numeric_labels():
    attrs = telemetry._metric_attrs({"mode": "gated", "ok": True, "count": 5, "ratio": 0.3})
    assert attrs == {"mode": "gated", "ok": True}  # numeric kept off metric labels


# ------------------------------------------------ enabled path (subprocess)
_ENABLED_SCRIPT = textwrap.dedent("""
    import json, sys
    import numpy as np
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from membukkit import telemetry
    from membukkit.config import RetrievalConfig, PromptConfig
    from membukkit.pipeline import MemorySystem

    CAPTURE = sys.argv[1] == "True"
    exp = InMemorySpanExporter()
    telemetry.configure(service_name="test", send_to_logfire=False, capture_content=CAPTURE,
                        additional_span_processors=[SimpleSpanProcessor(exp)])

    class Enc:
        def encode(self, texts, normalize=True, show_progress=False):
            single = isinstance(texts, str); items = [texts] if single else list(texts)
            out = []
            for t in items:
                rng = np.random.default_rng(abs(hash(t)) % (2**32))
                v = rng.standard_normal(16).astype(np.float32); v /= np.linalg.norm(v)+1e-8
                out.append(v)
            a = np.vstack(out).astype(np.float32); return a[0] if single else a

    class RR:
        def score(self, q, texts, batch_size=64):
            return np.asarray([len(set(q.split()) & set(t.split())) for t in texts], dtype=np.float32)

    mem = MemorySystem(encoder=Enc(), reranker=RR(), llm_fn=lambda p: "ans",
                       retrieval=RetrievalConfig(num_buckets=4, scan_budget=0.7, top_k=3),
                       prompts=PromptConfig.default(), distiller=None)
    mem.ingest(sessions=[[{"role":"user","content":f"fact about thing {i}"}] for i in range(10)],
               dates=["2024/06/01"]*10)
    mem.answer("which thing 3 is it")

    spans = exp.get_finished_spans()
    names = sorted(set(s.name for s in spans))
    answer = [s for s in spans if s.name == "memory.answer"][0]
    has_q = "question" in dict(answer.attributes)
    print(json.dumps({"names": names, "has_question": has_q}))
""")


def _run_enabled(capture: bool) -> dict:
    import json

    out = subprocess.run(
        [sys.executable, "-c", _ENABLED_SCRIPT, str(capture)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    # last stdout line is our JSON (logfire may print a banner)
    line = [ln for ln in out.stdout.strip().splitlines() if ln.startswith("{")][-1]
    return json.loads(line)


@pytest.mark.skipif(importlib.util.find_spec("logfire") is None, reason="logfire not installed")
def test_enabled_emits_expected_spans():
    res = _run_enabled(capture=False)
    for expected in [
        "memory.answer",
        "memory.retrieve",
        "memory.rerank",
        "memory.upsert",
        "memory.embed",
        "memory.ingest",
    ]:
        assert expected in res["names"], f"missing span {expected}: {res['names']}"


@pytest.mark.skipif(importlib.util.find_spec("logfire") is None, reason="logfire not installed")
def test_pii_gate_controls_content_capture():
    assert _run_enabled(capture=False)["has_question"] is False
    assert _run_enabled(capture=True)["has_question"] is True
