# MEMBUKKIT RAG Mode

> **Product document ingest**
>
> For uploading contracts, notes, and PDFs with as-of answers and citations,
> use the **[Documents](guide/documents.md)** guide (`membukkit ingest`, GUI drop-zone,
> `contract-qa` demo). This page is the **research/eval** RAG mode (passage index,
> EM/F1), a different door.

MEMBUKKIT now supports two operational modes:

1. **Chat-memory mode** (original), ingest conversations, distill atomic facts, answer with dated/reasoning readers. Evaluated with LLM judge on LongMemEval/LoCoMo.
2. **RAG mode** (new), index a document corpus directly (no distillation), retrieve passages via dense or CoreMem retrieval, answer with a short-answer QA reader. Evaluated with EM/F1 on MuSiQue, 2WikiMultiHopQA, HotpotQA.

## Key Difference

| | Chat-memory | RAG |
|---|---|---|
| Input | Conversation sessions | Document corpus |
| Processing | LLM distillation → atomic facts | Passages as-is (zero LLM cost at index time) |
| Reader | Dated/reasoning readers (prose) | QA reader (short factual answer) |
| Metrics | LLM judge accuracy | EM / F1 / Recall@k |
| Datasets | LongMemEval, LoCoMo | MuSiQue, 2WikiMultiHopQA, HotpotQA |

## Library Usage

### RAG Mode

```python
from membukkit.rag import RAGSystem
from membukkit.config import RAGConfig

cfg = RAGConfig(
    encoder="sentence-transformers/all-mpnet-base-v2",
    method="coremem",       # or "dense"
    fusion="cosine",
    decompose=True,         # iterative query decomposition
)
rag = RAGSystem.from_pretrained(rag_cfg=cfg, llm="openai:gpt-4o-mini")

rag.index([
    {"title": "Albert Einstein", "text": "Albert Einstein was a theoretical physicist..."},
    {"title": "Theory of Relativity", "text": "The theory of relativity..."},
])

result = rag.answer("What field did Einstein work in?")
print(result.answer)    # "theoretical physics"
print(result.passages)  # retrieved passages
```

### Chat-memory Mode (unchanged)

```python
from membukkit.pipeline import MemorySystem

mem = MemorySystem.from_pretrained()
mem.ingest(sessions, dates)
result = mem.answer("What is Alice's favorite food?", question_date="2024/06/01")
```

## CLI, RAG Evaluation

```bash
# Full evaluation (3 datasets x 1000 questions each)
membukkit rag-eval --datasets musique,2wiki,hotpot \
    --methods dense,coremem \
    --embedder nvidia/NV-Embed-v2 \
    --coremem-encoder nvidia/NV-Embed-v2 \
    --coremem-decompose \
    --top-k 5 --workers 16

# CPU smoke test
membukkit rag-eval --datasets 2wiki --methods dense --smoke 20
```

Key flags:
- `--methods dense,coremem`: which retrieval methods to evaluate
- `--coremem-decompose`: enable iterative query decomposition (SOTA)
- `--coremem-fusion cosine|rrf|rerank`: within-region ranking strategy
- `--no-reader`: retrieval-only mode (R@k metrics, zero API cost)
- `--reader-verify`: add a verification pass (re-read and confirm/correct)

## Over HTTP

RAG mode has no HTTP surface: it is a library and CLI feature, driven through
`RAGSystem` or `membukkit rag-eval`. Index a corpus in-process and query it there.

The multi-tenant service is the chat-memory surface, and it is scoped per owner
rather than per corpus:

```bash
pip install membukkit[service]
membukkit serve --port 8080
```

| Method and path | Purpose |
|---|---|
| `POST /v1/{owner}/ingest` | Distill and store conversation sessions |
| `POST /v1/{owner}/answer` | Answer from that owner's memory |
| `DELETE /v1/{owner}` | Delete the owner's namespace |
| `GET /health` | Liveness |

Full request and response models: **[HTTP API reference](reference/http-api.md)**.

## Cost Advantage

CoreMem's RAG mode indexes with **zero LLM calls**, the bi-encoder + KMeans buckets are purely compute-based. This contrasts with graph-based SOTA methods (HippoRAG 2, MultiCube-RAG) which require multiple LLM calls per passage during indexing.

With iterative query decomposition on NV-Embed-v2, MEMBUKKIT achieves SOTA-beating results on all three multi-hop benchmarks while maintaining this zero-index-LLM cost edge.
