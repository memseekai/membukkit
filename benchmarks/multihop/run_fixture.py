"""Run MemBukkit over a QMD-format corpus and emit a QMD-format report.

This is the like-for-like path. Both systems read the *same* directory of
markdown and the *same* ``queries.json``, and both emit the same report schema,
so comparing them needs no trust in this harness:

    python -m benchmarks.multihop.export_corpus --dataset musique --out /tmp/musique
    python -m benchmarks.multihop.run_fixture --corpus /tmp/musique --out mb.json

    qmd init && qmd collection add /tmp/musique/docs --name musique && qmd embed
    qmd bench /tmp/musique/queries.json --collection musique --json > qmd.json

    python -m benchmarks.common.qmd_report membukkit=mb.json qmd=qmd.json

MemBukkit's retrieval modes occupy the ``backends`` slot that QMD fills with
``bm25 / vector / hybrid / full``, so the summary tables line up.

Scoring is QMD's own, ported in ``benchmarks/common/qmd_compat.py``, including
its non-standard ``precision_at_k`` denominator. Any metric QMD does not define
is absent here rather than silently added.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Dict, List

from benchmarks.common import qmd_report
from benchmarks.multihop.run import _make_llm

DEFAULT_MODES = ("dense", "rerank", "chain")


class CappedEncoder:
    """A sentence-transformers model with an explicit sequence and batch cap.

    Large encoders (Qwen3-Embedding and friends) have very long native context;
    at the default batch size attention memory blows past the MPS limit, so both
    caps are set explicitly rather than left to the model defaults.
    """

    def __init__(self, path: str, max_seq_length: int = 384, batch_size: int = 8):
        from sentence_transformers import SentenceTransformer

        self.path = path
        self.model = SentenceTransformer(path)
        if max_seq_length:
            self.model.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def encode(self, texts, normalize: bool = True, show_progress: bool = False):
        import numpy as np

        return np.asarray(self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=normalize,
            show_progress_bar=show_progress))


class AnchoredScorer:
    """Loads a coremem3 `learned_router.BucketScorer` checkpoint standalone.

    The checkpoint is {state, d_in, geo_scale} over a 2-layer MLP, so it is
    reloaded here directly rather than importing coremem3 into the benchmark.
    Scoring must match `BucketScorer._forward` exactly:

        geo_scale * feats[:, 0]  +  MLP(feats[:, 1:])

    with geo_scale a frozen constant and the MLP never seeing column 0.
    """

    def __init__(self, path: str):
        import torch
        import torch.nn as nn

        ckpt = torch.load(path, map_location="cpu")
        self.geo_scale = float(ckpt.get("geo_scale", 10.0))
        d_in = int(ckpt.get("d_in", 4))
        hidden = ckpt["state"]["0.weight"].shape[0]
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.net.load_state_dict(ckpt["state"])
        self.net.eval()
        self._torch = torch

    def score_matrix(self, feats):
        import numpy as np

        with self._torch.no_grad():
            x = self._torch.tensor(np.asarray(feats), dtype=self._torch.float32)
            out = self.geo_scale * x[:, 0] + self.net(x[:, 1:]).squeeze(-1)
        return out.numpy()


def load_corpus(docs_dir: pathlib.Path) -> List:
    """Every markdown file in ``docs_dir``, doc_id being the filename QMD reports."""
    from membukkit.retrieval.rag import Document

    docs = []
    for path in sorted(docs_dir.rglob("*.md")):
        rel = path.relative_to(docs_dir).as_posix()
        docs.append(Document(doc_id=rel, text=path.read_text(), title=path.stem))
    if not docs:
        raise SystemExit(f"no .md files under {docs_dir}")
    return docs


def run_mode(mode: str, docs, queries, collection: str, top_k: int,
             llm_model: str, progress_every: int,
             checkpoint_every: int = 0, on_checkpoint=None,
             encoder=None, embeddings=None, reranker=None,
             scorer=None) -> Dict[str, Dict]:
    from membukkit.retrieval.rag import RagRetriever

    llm = _make_llm(llm_model) if mode == "decompose" else None
    r = RagRetriever(mode=mode, llm=llm, encoder=encoder, reranker=reranker,
                     scorer=scorer)
    t0 = time.perf_counter()
    r.index(docs, embeddings=embeddings)
    print(f"  [{mode}] indexed {len(docs)} docs in {time.perf_counter()-t0:.0f}s", flush=True)
    if embeddings is None:
        embeddings = r._index.embeddings  # reuse for the remaining modes
    return _query_loop(r, mode, queries, collection, top_k, progress_every,
                       checkpoint_every, on_checkpoint), embeddings


def _query_loop(r, mode, queries, collection, top_k, progress_every,
                checkpoint_every, on_checkpoint) -> Dict[str, Dict]:

    runs: Dict[str, Dict] = {}
    t1 = time.perf_counter()
    for i, q in enumerate(queries, start=1):
        t = time.perf_counter()
        hits = r.search(q.get("query", ""), top_k=top_k)
        runs[str(q.get("id"))] = {
            "top_files": [qmd_report.qmd_uri(collection, h.doc_id) for h in hits],
            "latency_ms": (time.perf_counter() - t) * 1000.0,
        }
        if progress_every and i % progress_every == 0:
            el = time.perf_counter() - t1
            eta = (len(queries) - i) * el / i
            print(f"  [{mode}] {i}/{len(queries)} ({el:.0f}s, {el/i:.2f}s/query, "
                  f"eta {eta/60:.0f}m)", flush=True)
        # Score what is finished so far. A long run should show its shape early,
        # not only at the end.
        if checkpoint_every and on_checkpoint and i % checkpoint_every == 0:
            on_checkpoint(mode, queries[:i], dict(runs))
    return runs


def run(corpus: pathlib.Path, modes=DEFAULT_MODES, top_k: int = 10,
        limit: int | None = None, llm_model: str = "",
        progress_every: int = 50, checkpoint_every: int = 0,
        partial_out: pathlib.Path | None = None,
        encoder_path: str = "", encoder_seqlen: int = 384,
        encoder_batch: int = 8, reranker_path: str = "",
        scorer_path: str = "") -> Dict:
    fixture_path = corpus / "queries.json"
    fixture = qmd_report.load_fixture(fixture_path)
    queries = fixture["queries"][: limit or len(fixture["queries"])]
    collection = fixture.get("collection") or corpus.name
    docs = load_corpus(corpus / "docs")

    encoder = None
    if encoder_path:
        print(f"encoder override: {encoder_path} "
              f"(seqlen={encoder_seqlen}, batch={encoder_batch})", flush=True)
        encoder = CappedEncoder(encoder_path, encoder_seqlen, encoder_batch)

    reranker = None
    if reranker_path:
        from membukkit.models.reranker import UtilityReranker

        print(f"reranker override: {reranker_path}", flush=True)
        # Loaded once here rather than lazily per mode: three modes would
        # otherwise load three copies of the same cross-encoder.
        reranker = UtilityReranker.load(reranker_path)

    scorer = None
    if scorer_path:
        print(f"anchored scorer: {scorer_path}", flush=True)
        scorer = AnchoredScorer(scorer_path)

    print(f"{len(queries)} queries over {len(docs)} documents, "
          f"modes={','.join(modes)}", flush=True)

    done_runs: Dict[str, Dict] = {}

    def checkpoint(mode, done_queries, runs):
        partial = qmd_report.build_report(
            str(fixture_path), done_queries, {**done_runs, mode: runs})
        print(f"\n  --- partial: {len(done_queries)}/{len(queries)} queries ---",
              flush=True)
        print(qmd_report.summary_table({"membukkit": partial}), flush=True)
        print("", flush=True)
        if partial_out:
            partial_out.write_text(json.dumps(partial, indent=2) + "\n")

    backend_runs = {}
    shared_embeddings = None
    for mode in modes:
        backend_runs[mode] = run_mode(
            mode, docs, queries, collection, top_k, llm_model, progress_every,
            checkpoint_every=checkpoint_every, on_checkpoint=checkpoint,
            encoder=encoder, embeddings=shared_embeddings, reranker=reranker,
            scorer=scorer)
        backend_runs[mode], shared_embeddings = backend_runs[mode]
        done_runs[mode] = backend_runs[mode]
    index_meta = {}
    index_json = corpus / "index.json"
    if index_json.exists():
        meta = json.loads(index_json.read_text())
        index_meta = {k: meta[k] for k in
                      ("dataset", "source", "n_passages", "qa_sha256", "corpus_sha256")
                      if k in meta}

    return qmd_report.build_report(
        str(fixture_path), queries, backend_runs,
        extra={"system": "membukkit", "dataset": index_meta,
               "config": _config_snapshot(encoder_path, encoder_seqlen,
                                          encoder_batch, llm_model, top_k,
                                          reranker_path, scorer_path),
               "note": "Backends are MemBukkit retrieval modes. Scored with "
                       "QMD's scorer over QMD's fixture format."},
    )


def _config_snapshot(encoder_path, seqlen, batch, llm_model, top_k,
                     reranker_path="", scorer_path="") -> Dict:
    """Whatever a reader needs to know which stack produced these numbers."""
    import membukkit
    from membukkit.config import ModelConfig
    from membukkit.models.registry import resolve_encoder_path, resolve_reranker_path

    return {
        "membukkit_version": getattr(membukkit, "__version__", "unknown"),
        "encoder": encoder_path or resolve_encoder_path(ModelConfig()),
        "encoder_overridden": bool(encoder_path),
        "encoder_max_seq_length": seqlen,
        "encoder_batch_size": batch,
        "reranker": reranker_path or resolve_reranker_path(ModelConfig()),
        "reranker_overridden": bool(reranker_path),
        "anchored_scorer": scorer_path or None,
        "llm": llm_model or None,
        "top_k": top_k,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=pathlib.Path, required=True,
                    help="directory written by benchmarks.multihop.export_corpus")
    ap.add_argument("--modes", default=",".join(DEFAULT_MODES),
                    help="comma-separated: dense,rerank,chain,decompose")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="0 = every query")
    ap.add_argument("--llm", default="", help="decompose only: ollama:<tag> or an OpenAI model id")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--checkpoint-every", type=int, default=0,
                    help="score and print partial results every N queries (0 = off)")
    ap.add_argument("--encoder", default="",
                    help="override the bi-encoder (HF id or path); default is the shipped one")
    ap.add_argument("--reranker", default="",
                    help="override the cross-encoder reranker (path or HF id)")
    ap.add_argument("--scorer", default="",
                    help="anchored residual scorer checkpoint; replaces RRF fusion")
    ap.add_argument("--encoder-seqlen", type=int, default=384)
    ap.add_argument("--encoder-batch", type=int, default=8)
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if "decompose" in modes and not args.llm:
        raise SystemExit("mode 'decompose' needs --llm (e.g. ollama:qwen3:1.7b)")

    partial_out = args.out.with_suffix(".partial.json") if args.checkpoint_every else None
    report = run(args.corpus, modes=modes, top_k=args.top_k,
                 limit=args.limit or None, llm_model=args.llm,
                 checkpoint_every=args.checkpoint_every, partial_out=partial_out,
                 encoder_path=args.encoder, encoder_seqlen=args.encoder_seqlen,
                 encoder_batch=args.encoder_batch, reranker_path=args.reranker,
                 scorer_path=args.scorer)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print()
    print(qmd_report.summary_table({"membukkit": report}))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
