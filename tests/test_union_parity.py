"""Golden tests for the production union path used by `membukkit eval`.

The SOTA numbers (LoCoMo 0.877 / LongMemEval 0.81) come from the dual
verbatim+atomic retrieval, unioned. The eval CLI now drives the production
`MemorySystem` union directly (the `library` engine is the single source of
truth; the old bespoke in-CLI retrieval has been removed). These tests pin the
presented fact lines of `MemorySystem.answer(...)` and of the eval task builder
(`_build_all_tasks_lib`) to captured goldens over a fixed synthetic
conversation, so the evaluated path can never silently drift. Determinism comes
from process-stable hashing in the fakes plus seeded KMeans.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np

from membukkit.config import PromptConfig, RetrievalConfig
from membukkit.pipeline import MemorySystem
from membukkit.time_utils import parse_datetime


def _shash(s, mod: int) -> int:
    """Process-stable hash (Python's builtin hash() is salted per run, which would
    make the goldens below non-reproducible)."""
    return int(hashlib.sha1(str(s).encode()).hexdigest(), 16) % mod


class FakeEncoder:
    """Deterministic encoder accepting BOTH call conventions (backend + CLI)."""

    def __init__(self, dim: int = 32):
        self.dim = dim

    def encode(
        self,
        texts,
        normalize=True,
        normalize_embeddings=None,
        show_progress=False,
        show_progress_bar=False,
        **_,
    ):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        vecs = []
        for t in items:
            rng = np.random.default_rng(_shash(t, 2**32))
            v = rng.standard_normal(self.dim).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-8  # always L2-normalize (both APIs ask for it)
            vecs.append(v)
        out = np.vstack(vecs).astype(np.float32)
        return out[0] if single else out


class FakeReranker:
    """Deterministic cross-encoder: token overlap + a stable hash tiebreak."""

    def score(self, query, texts, batch_size: int = 64):
        qs = set(query.lower().split())
        out = []
        for t in texts:
            overlap = len(qs & set(t.lower().split()))
            jitter = _shash(t, 1000) / 1e6
            out.append(overlap + jitter)
        return np.asarray(out, dtype=np.float32)


class FakeDistiller:
    """Deterministic distiller: one unique atomic fact per transcript line.

    `MemorySystem.ingest()` calls `distill(key, transcript, date)`; facts are made
    unique per (transcript, line) to avoid content dedup, and the atom tag is a
    process-stable hash so the goldens above stay reproducible across runs.
    """

    subject = None

    def distill(self, key, transcript, date):
        tag = _shash(transcript, 100000)
        units = []
        for i, line in enumerate(transcript.split("\n")):
            line = line.strip()
            if not line:
                continue
            units.append((i, f"atom{tag}_{i} {line[:40]}"))
        return units

    def save(self):
        pass


class FakeInst:
    """Minimal LongMemEval-shaped instance for both CLI task builders."""

    def __init__(
        self,
        sessions: List[List[Dict[str, str]]],
        dates: List[str],
        question: str,
        ground_truth: str = "gt",
        qid: str = "q1",
    ):
        self._item: Dict[str, Any] = {
            "haystack_sessions": sessions,
            "haystack_dates": dates,
            "question_type": "multi-session",
            "locomo_category": 2,
        }
        self._question = question
        self._gt = ground_truth
        self.question_id = qid
        self.distill_scope_id = "conv1"
        self.ability = "multi_session"
        self.question_date_raw = "2024-06-15"

    def get_facts(self):
        out = []
        for s_idx, session in enumerate(self._item["haystack_sessions"]):
            ts = parse_datetime(self._item["haystack_dates"][s_idx])
            for turn in session:
                c = (turn.get("content", "") or "").strip()
                if not c:
                    continue
                out.append(SimpleNamespace(text=c, timestamp=ts))
        return out

    def get_queries(self):
        return [SimpleNamespace(text=self._question, ground_truth=self._gt)]


_SESSIONS = [
    [
        {"role": "user", "content": "I adopted a dog named Luna in early June."},
        {"role": "assistant", "content": "Congratulations, dogs are wonderful."},
        {"role": "user", "content": "Luna is a golden retriever puppy."},
    ],
    [
        {"role": "user", "content": "I started taking piano lessons last week."},
        {"role": "user", "content": "My piano teacher is named Clara."},
        {"role": "assistant", "content": "That sounds like a lovely hobby."},
    ],
]
_DATES = ["2024-06-01", "2024-06-08"]


def _cfg(top_k: int = 3) -> RetrievalConfig:
    return RetrievalConfig(
        union=True,
        bucket_mode="topic",
        scan_budget=0.3,
        scan_budget_temporal=None,
        num_buckets=24,
        k_proto=0,
        select="hybrid",
        rerank_cap=50,
        top_k=top_k,
        reasoning_top_k=30,
        k_rrf=60,
    )


# Golden union fact lines for the fixed synthetic conversation, captured from the
# production MemorySystem path. Deterministic because the fakes use a
# process-stable hash (see `_shash`) and KMeans is seeded (random_state=0). These
# replace the old library-vs-legacy comparison now that the legacy CLI retrieval
# path has been removed (the library engine is the single source of truth).
_GOLDEN_UNION_ANSWER = [
    "[2024-06-01] Luna is a golden retriever puppy.",
    "[2024-06-08] My piano teacher is named Clara.",
    "[2024-06-08] atom64189_2 [T2] assistant: That sounds like a lovel",
    "[2024-06-08] atom64189_0 [T0] user: I started taking piano lesson",
]

_GOLDEN_TASK_LINES = {
    "coremem_union": [
        "[2024-06-01] Congratulations, dogs are wonderful.",
        "[2024-06-01] I adopted a dog named Luna in early June.",
        "[2024-06-08] That sounds like a lovely hobby.",
        "[2024-06-08] atom64189_2 [T2] assistant: That sounds like a lovel",
        "[2024-06-08] atom64189_0 [T0] user: I started taking piano lesson",
    ],
    "coremem": [
        "[2024-06-01] Congratulations, dogs are wonderful.",
        "[2024-06-01] I adopted a dog named Luna in early June.",
        "[2024-06-08] That sounds like a lovely hobby.",
    ],
    "coremem_atomic": [
        "[2024-06-08] atom64189_2 [T2] assistant: That sounds like a lovel",
        "[2024-06-08] atom64189_0 [T0] user: I started taking piano lesson",
    ],
}


def test_memorysystem_union_matches_golden():
    question = "What kind of dog is Luna?"
    top_k = 3

    mem = MemorySystem(
        encoder=FakeEncoder(),
        reranker=FakeReranker(),
        llm_fn=lambda p: "unused",
        retrieval=_cfg(top_k),
        prompts=PromptConfig.default(),
        distiller=FakeDistiller(),
    )
    mem.ingest(sessions=_SESSIONS, dates=_DATES)

    res = mem.answer(question, generate_answer=False)

    assert res.answer is None
    assert res.facts == _GOLDEN_UNION_ANSWER


def _eval_args(top_k: int = 3):
    return SimpleNamespace(
        top_k=top_k,
        reasoning_top_k=30,
        cand=50,
        scan_budget=0.3,
        scan_budget_temporal=None,
        scan_budget_reason=0.45,
        bucket_mode="topic",
        bucket_k=24,
        bucket_k_proto=0,
        bucket_rerank_cap=50,
        rerank_select="hybrid",
        distill_workers=2,
    )


def test_library_task_builder_golden_across_methods():
    """The library-driven eval builder produces the expected per-method fact lines
    (coremem_union / coremem / coremem_atomic) for the fixed conversation."""
    from membukkit.cli import eval_legacy as cli

    question = "Where does Clara teach piano?"
    methods = ["coremem_union", "coremem", "coremem_atomic"]

    library = cli._build_all_tasks_lib(
        [FakeInst(_SESSIONS, _DATES, question)],
        _cli_adapter(FakeEncoder()),
        FakeReranker(),
        _eval_args(),
        methods,
        FakeDistiller(),
    )
    by_m = {t["method"]: t["fact_lines"] for t in library}

    assert by_m == _GOLDEN_TASK_LINES
    # coremem_union is exactly verbatim ++ atomic; atomic-only differs from verbatim-only
    assert by_m["coremem_union"] == by_m["coremem"] + by_m["coremem_atomic"]
    assert by_m["coremem_atomic"] != by_m["coremem"]


def _cli_adapter(enc):
    from membukkit.cli.eval_legacy import _STAdapter

    # _STAdapter forwards to a SentenceTransformer-style .encode; FakeEncoder
    # already speaks that dialect, so wrap a tiny shim exposing it as `_st`.
    class _ST:
        def encode(self, texts, show_progress_bar=False, normalize_embeddings=True):
            return enc.encode(texts, normalize=normalize_embeddings)

    return _STAdapter(_ST())


def test_union_off_is_single_index_atomic_only():
    """union=False keeps the classic single-index (atomic-only) behaviour."""
    cfg = _cfg()
    cfg.union = False

    enc = FakeEncoder()
    mem = MemorySystem(
        encoder=enc,
        reranker=FakeReranker(),
        llm_fn=lambda p: "unused",
        retrieval=cfg,
        prompts=PromptConfig.default(),
        distiller=FakeDistiller(),
    )
    mem.ingest(sessions=_SESSIONS, dates=_DATES)

    # No verbatim rows are written when union is off.
    assert mem._backend.count_kind("verbatim") == 0
    assert mem._backend.count_kind("atomic") > 0

    res = mem.answer("What kind of dog is Luna?", generate_answer=False)
    assert res.facts  # retrieval still returns atomic facts


def test_select_none_never_calls_cross_encoder():
    """select='none' is the drop-the-reranker ablation arm: plain cosine order,
    the cross-encoder must never be invoked (unlike select='cosine', which
    still lets the cross-encoder order the capped pool)."""

    class ExplodingReranker:
        def score(self, query, texts, batch_size=64):
            raise AssertionError("cross-encoder was called in select='none'")

    cfg = _cfg()
    cfg.select = "none"
    mem = MemorySystem(
        encoder=FakeEncoder(),
        reranker=ExplodingReranker(),
        llm_fn=lambda p: "unused",
        retrieval=cfg,
        prompts=PromptConfig.default(),
        distiller=FakeDistiller(),
    )
    mem.ingest(sessions=_SESSIONS, dates=_DATES)

    res = mem.answer("What kind of dog is Luna?", generate_answer=False)
    assert res.facts  # retrieval works without the reranker ever running
