"""Configuration dataclasses for MEMBUKKIT."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional


@dataclass
class ModelConfig:
    """Model paths and loading configuration.

    Resolution order for model paths:
    1. Explicit path in model_dir / encoder / reranker
    2. MEMBUKKIT_MODEL_DIR environment variable
    3. Default relative paths (models/biencoder_v1, models/reranker_v2/model)
    """

    model_dir: Optional[str] = None
    encoder: str = "biencoder_v1"
    reranker: str = "reranker_v2/model"
    device: Optional[str] = None


@dataclass
class RetrievalConfig:
    """Retrieval pipeline configuration."""

    num_buckets: int = 24
    scan_budget: float = 0.3
    scan_budget_temporal: Optional[float] = None
    scan_budget_reason: float = 0.45
    bucket_mode: str = "topic"
    # Within-region ranking. "hybrid" (shipped): RRF over cosine + cross-encoder
    # ranks. "cosine": cosine-top-`rerank_cap` prefilter, then cross-encoder
    # order. "xenc": cross-encoder order alone. "none": plain cosine, the
    # cross-encoder never runs (the drop-reranker ablation arm).
    select: str = "hybrid"
    top_k: int = 10
    reasoning_top_k: int = 30
    candidate_pool: int = 50
    rerank_cap: int = 50
    k_proto: int = 0
    k_rrf: int = 60
    # --- union (dual verbatim+atomic retrieval) ---
    # When True, ingest stores both raw conversation turns (verbatim lane) and
    # distilled atomic facts (atomic lane); answer/search retrieve each lane
    # independently and concatenate (verbatim then atomic) — the SOTA
    # `coremem_union` behaviour. When False, single-index atomic-only retrieval.
    union: bool = True
    # Which lanes retrieval reads from (storage always writes both under union).
    # Production uses both; the eval sets ("verbatim",) / ("atomic",) for the
    # coremem / coremem_atomic ablations without a second code path.
    union_lanes: tuple = ("verbatim", "atomic")
    # Recommendation/preference queries are routed to the verbatim lane only
    # (atomic facts dilute the preference signal). Matches eval's E20 routing.
    union_recommendation_verbatim_only: bool = True
    # --- optional local BM25 lexical lane (OFF by default) ---
    # Dense topic routing is the shipped method and the one every published
    # number was measured with; leaving this False keeps that path untouched.
    # When True, a BM25 lane retrieves over the whole bank and its hits are
    # added to the routed pool, then fused by RRF alongside cosine and the
    # cross-encoder. Helps exact-string queries the bi-encoder under-weights.
    # Needs the `bm25` extra. Distinct from `bm25_lane` below, which is the
    # Turbopuffer server-side lane and applies only to that backend.
    lexical_lane: bool = False
    lexical_top_k: int = 20  # BM25 hits considered per lane before the union
    # --- service / Turbopuffer retrieval knobs ---
    retrieval_mode: str = "gated"  # "gated" (topic_bucket filter) | "open" (ANN-first)
    pool_size: int = 128  # candidates fetched from the DB before cross-encoder rerank
    bm25_lane: bool = True  # add a BM25 lexical lane fused server-side via RRF


@dataclass
class StorageConfig:
    """Where memory is persisted.

    backend="memory" keeps the original in-RAM behaviour (default; used by
    tests and eval). backend="turbopuffer" persists to a per-owner namespace.
    """

    backend: str = "memory"
    namespace: Optional[str] = None  # Turbopuffer namespace (one per memory owner)
    region: Optional[str] = None  # e.g. "gcp-us-central1"
    api_key: Optional[str] = None  # falls back to TURBOPUFFER_API_KEY env
    vector_dtype: str = "f16"  # stored vector precision ("f16" | "f32")
    recluster_growth_threshold: float = 0.5  # re-cluster when bank grows by this fraction


@dataclass
class PromptConfig:
    """Editable prompt templates and Mem0-style instruction overlays.

    ``None`` / empty for a template field means "use the built-in default".
    Instruction overlays are appended into the active template (default or
    override) without requiring a full copy of the prompt.
    """

    extraction: Optional[str] = None
    extraction_named: Optional[str] = None
    extraction_document: Optional[str] = None
    dated_reader: Optional[str] = None
    recommendation_reader: Optional[str] = None
    reasoning_reader: Optional[str] = None
    abstain_gate: Optional[str] = None
    # Natural-language overlays (Mem0-style); applied when the matching full
    # template override is unused, and also onto overrides if set.
    extraction_instructions: Optional[str] = None
    reader_instructions: Optional[str] = None

    @classmethod
    def default(cls) -> "PromptConfig":
        return cls()

    def is_default(self) -> bool:
        """True when every field is unset (built-in prompts, no overlays)."""
        return all(getattr(self, f.name) in (None, "") for f in fields(self))

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, "")}

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "PromptConfig":
        if not data:
            return cls.default()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known and v not in (None, "")})


@dataclass
class RAGConfig:
    """Configuration for the RAG retrieval pipeline.

    Controls the encoder, retrieval strategy (dense vs CoreMem with buckets,
    multi-hop, decomposition), and reader behaviour.
    """

    encoder: str = "biencoder_v1"  # resolved via models.registry (local/HF Hub/fallback)
    reranker: str = "reranker_v2/model"
    query_prompt: str = ""
    trust_remote_code: bool = False
    max_seq_length: int = 0
    batch_size: int = 64
    method: str = "coremem"
    fusion: str = "cosine"
    budget: float = 0.3
    bucket_k: int = 24
    rerank_cap: int = 100
    top_k: int = 5
    hops: int = 1
    expand_m: int = 3
    expand_mode: str = "entity"
    axes: str = "topic"
    temporal: bool = False
    entity_cap: int = 50
    entity_rank: bool = False
    entity_min: int = 1
    decompose: bool = False
    max_subq: int = 4
    decompose_fuse: str = "interleave"
    decompose_iter: bool = True
    decompose_retrieval: str = "full_cosine"
    reader_verify: bool = False
