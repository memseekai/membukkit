"""RAGSystem — corpus-based retrieval-augmented generation.

The RAG sibling to MemorySystem. Where MemorySystem ingests conversations,
distills atomic facts, and answers with dated/reasoning readers (judge-scored),
RAGSystem ingests a document corpus directly (no distillation), retrieves
passages via dense or CoreMem retrieval, and answers with a short-answer QA
reader scored by EM/F1.

    rag = RAGSystem.from_pretrained(...)
    rag.index([{"title": "...", "text": "..."}, ...])
    result = rag.answer("Who directed the film?")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from membukkit.config import ModelConfig, RAGConfig

logger = logging.getLogger(__name__)


@dataclass
class RAGTrace:
    """Inspectable trace of a RAG retrieval."""

    method: str = "dense"
    n_passages: int = 0
    n_retrieved: int = 0
    ranked_indices: List[int] = field(default_factory=list)
    ranked_titles: List[str] = field(default_factory=list)


@dataclass
class RAGResult:
    """Result of rag.answer()."""

    answer: str = ""
    passages: List[str] = field(default_factory=list)
    trace: RAGTrace = field(default_factory=RAGTrace)


class RAGSystem:
    """Stateful RAG system: index a corpus, answer questions.

    The pipeline:
      1. index: passages as retrieval units (no distillation — the key difference
         from MemorySystem's chat-memory mode).
      2. answer: retrieve top-k passages via dense cosine or CoreMem (bucket gating,
         multi-hop, decomposition) -> QA reader for short-answer extraction.
    """

    def __init__(
        self,
        retriever,
        llm_fn: Optional[Callable[[str], str]],
        cfg: RAGConfig,
    ):
        self._retriever = retriever
        self._llm_fn = llm_fn
        self._cfg = cfg
        self._reader = None
        self._passages_text: List[str] = []
        self._passage_titles: List[str] = []

    @classmethod
    def from_pretrained(
        cls,
        rag_cfg: Optional[RAGConfig] = None,
        llm: str = "openai:gpt-4o-mini",
    ) -> "RAGSystem":
        """Create a RAGSystem with configured encoder + retrieval strategy.

        Args:
            rag_cfg: RAG pipeline configuration. Uses defaults if None.
            llm: LLM spec string like "openai:gpt-4o-mini" for the QA reader.
                 Set to "" or None to skip the reader (retrieval-only mode).
        """
        rag_cfg = rag_cfg or RAGConfig()

        from membukkit.models.registry import resolve_encoder_path, resolve_reranker_path

        rag_cfg.encoder = resolve_encoder_path(ModelConfig(encoder=rag_cfg.encoder))
        rag_cfg.reranker = resolve_reranker_path(ModelConfig(reranker=rag_cfg.reranker))

        from membukkit.retrieval.multihop import SubstrateEncoder, DenseRetriever, CoreMemRetriever

        encoder = SubstrateEncoder(
            rag_cfg.encoder,
            query_prompt=rag_cfg.query_prompt,
            trust_remote_code=rag_cfg.trust_remote_code,
            max_seq_length=rag_cfg.max_seq_length,
            batch_size=rag_cfg.batch_size,
        )

        llm_fn = None
        if llm:
            from membukkit.llm.backends import parse_llm_spec

            llm_fn = parse_llm_spec(llm)

        if rag_cfg.method == "dense":
            retriever = DenseRetriever(encoder)
        else:
            retriever = CoreMemRetriever(
                encoder_path=rag_cfg.encoder,
                reranker_path=rag_cfg.reranker,
                budget=rag_cfg.budget,
                bucket_k=rag_cfg.bucket_k,
                rerank_cap=rag_cfg.rerank_cap,
                fusion=rag_cfg.fusion,
                hops=rag_cfg.hops,
                expand_m=rag_cfg.expand_m,
                expand_mode=rag_cfg.expand_mode,
                axes=rag_cfg.axes,
                temporal=rag_cfg.temporal,
                entity_cap=rag_cfg.entity_cap,
                entity_rank=rag_cfg.entity_rank,
                entity_min=rag_cfg.entity_min,
                decompose=rag_cfg.decompose,
                max_subq=rag_cfg.max_subq,
                decompose_fuse=rag_cfg.decompose_fuse,
                decompose_iter=rag_cfg.decompose_iter,
                decompose_retrieval=rag_cfg.decompose_retrieval,
                llm_fn=llm_fn,
                query_prompt=rag_cfg.query_prompt,
                trust_remote_code=rag_cfg.trust_remote_code,
                max_seq_length=rag_cfg.max_seq_length,
                batch_size=rag_cfg.batch_size,
                encoder=encoder,
            )

        return cls(retriever, llm_fn, rag_cfg)

    def index(self, corpus: List[Dict[str, str]]) -> None:
        """Index a passage corpus for retrieval.

        Args:
            corpus: list of {"title": ..., "text": ...} dicts.
        """
        self._passage_titles = [c["title"] for c in corpus]
        self._passages_text = [f"{c['title']}\n{c['text']}" for c in corpus]
        self._retriever.index(self._passages_text, self._passage_titles)

    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> RAGResult:
        """Answer a question using the indexed corpus.

        Args:
            question: The question text.
            top_k: Number of passages to retrieve (default: cfg.top_k).

        Returns:
            RAGResult with .answer, .passages, .trace
        """
        if not self._passages_text:
            return RAGResult(answer="", passages=[], trace=RAGTrace())

        k = top_k or self._cfg.top_k
        idxs = self._retriever.retrieve(question, k)
        passages = [self._passages_text[i] for i in idxs]
        titles = [self._passage_titles[i] for i in idxs]

        ans = ""
        if self._llm_fn is not None:
            if self._reader is None:
                from membukkit.reading.qa_reader import make_qa_reader

                self._reader = make_qa_reader(self._llm_fn, verify=self._cfg.reader_verify)
            try:
                ans = self._reader(passages, question)
            except Exception:
                ans = ""

        trace = RAGTrace(
            method=getattr(self._retriever, "name", "unknown"),
            n_passages=len(self._passages_text),
            n_retrieved=len(idxs),
            ranked_indices=idxs,
            ranked_titles=titles,
        )
        return RAGResult(answer=ans, passages=passages, trace=trace)
