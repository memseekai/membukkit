"""MEMBUKKIT command-line interface.

Entry points:
    membukkit eval          — run evaluation on LongMemEval or LoCoMo
    membukkit train-encoder — fine-tune the bi-encoder
    membukkit train-reranker — fine-tune the cross-encoder reranker
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _pbar(total, desc):
    """A live tqdm progress bar if tqdm is installed, else None (callers no-op)."""
    try:
        from tqdm.auto import tqdm

        return tqdm(total=total, desc=desc, dynamic_ncols=True, unit="q")
    except Exception:
        return None


CORRECT_THRESHOLD = 0.7
LLM_WORKERS = 8

# BEAM per-category reader instructions (answer-form only; official judging
# untouched). Derived from 100K error analysis, see analysis/beam docs:
# - event_ordering is scored by newline-split + LLM equivalence alignment
#   against long, specific rubric items, then Kendall tau-b x F1: terse
#   labels never align (measured F1 0.089), and prose scores ~0.
# - summarization/multi-session/instruction rubrics grade nugget coverage
#   with int() truncation (0.5 partials -> 0), so completeness is binary.
# - contradiction rubrics require SURFACING the conflict, not resolving it.
# event_ordering is handled by the dedicated ordering reader (see
# make_ordering_reader), not a hint: it needs deterministic date-sorting
# and rubric-granularity item construction.
_BEAM_READER_HINTS = {
    "summarization": (
        "Be comprehensive: cover every relevant aspect, decision, and change "
        "mentioned across the conversations, with dates where known."
    ),
    "contradiction_resolution": (
        "If the facts contain conflicting statements about what is asked, do "
        "NOT silently resolve them: explicitly say the information is "
        "contradictory and state both versions with their dates."
    ),
    "multi_session_reasoning": (
        "If the question asks for a count or total, first list every "
        "distinct matching item with its date, then state the final count "
        "on the last line."
    ),
    "instruction_following": (
        "Be complete and explicit: include exact versions or values where "
        "they were mentioned, show step-by-step working when computing "
        "something, and honor any formatting the user previously requested."
    ),
    "knowledge_update": (
        "If a value changed over time, give the MOST RECENT value and note "
        "the date it changed."
    ),
}

# Bump when the built-task schema or fact-line semantics change so stale caches
# are naturally invalidated instead of silently reused.
_TASK_CACHE_VERSION = 1


def _path_sig(path: str) -> dict:
    """Identity signature of a model/data path (weights or file contents can
    change under the same path, so include mtime+size, not just the string)."""
    try:
        st = os.stat(path)
        return {"path": str(path), "mtime": int(st.st_mtime), "size": st.st_size}
    except OSError:
        return {"path": str(path), "exists": False}


def _task_cache_key(args, methods) -> str:
    """Stable hash over everything that affects the built fact_lines.

    The reader/judge are applied AFTER task building, so they're deliberately
    excluded — the whole point is that swapping readers reuses the same tasks.
    """
    from dataclasses import asdict

    dataset = getattr(args, "dataset", "longmemeval")
    key = {
        "v": _TASK_CACHE_VERSION,
        "dataset": dataset,
        "engine": getattr(args, "retrieval_engine", "library"),
        "storage_backend": getattr(args, "storage_backend", "memory"),
        "methods": list(methods),
        "max_instances": getattr(args, "max_instances", 0),
        "retrieval_cfg": asdict(_retrieval_cfg_from_args(args)),
        "bucket_mode": getattr(args, "bucket_mode", None),
        "encoder": _path_sig(args.encoder),
        "reranker": _path_sig(args.reranker),
        "distill_model": getattr(args, "distill_model", ""),
    }
    # Interventions/tracing change the built fact_lines (blocking) or the task
    # payload (trace fields), so they get their own cache entries. Only folded
    # in when active, so pre-existing plain-run caches keep their keys.
    if getattr(args, "block_buckets", "none") not in (None, "none"):
        key["block_buckets"] = args.block_buckets
        key["block_seed"] = getattr(args, "block_seed", 0)
    if getattr(args, "agg_top_k", 0) or getattr(args, "agg_scan_budget", None):
        key["agg_top_k"] = getattr(args, "agg_top_k", 0)
        key["agg_scan_budget"] = getattr(args, "agg_scan_budget", None)
    if getattr(args, "deep_top_k", 0) or getattr(args, "deep_scan_budget", None):
        key["deep_top_k"] = getattr(args, "deep_top_k", 0)
        key["deep_scan_budget"] = getattr(args, "deep_scan_budget", None)
        # Opt-in broad-context cues widen the deep-routing predicate, changing
        # fact_lines under the same deep params; keyed only when on so the
        # LongMemEval scoped-deep caches keep their keys.
        if getattr(args, "deep_broad", False):
            key["deep_broad"] = True
    if getattr(args, "dump_traces", False):
        key["dump_traces"] = True
    # AMB ordering mode appends a verbatim-quote timeline to event-ordering
    # fact_lines; keyed only when non-default so official caches keep keys.
    # The version suffix tracks timeline construction changes.
    if getattr(args, "beam_ordering_mode", "official") != "official":
        key["beam_ordering_mode"] = args.beam_ordering_mode + "+vquote_v1"
    if getattr(args, "only_ids", ""):
        key["only_ids"] = sorted(
            s.strip() for s in args.only_ids.split(",") if s.strip()
        )
        # Targeted-subset runs typically pair with a modified distill cache
        # (e.g. re-distilled evidence sessions), so the cache path must key the
        # tasks; folded in only here so pre-existing full-run caches keep their
        # keys.
        key["distill_cache"] = str(getattr(args, "distill_cache", ""))
    if dataset == "locomo":
        key["locomo"] = _path_sig(getattr(args, "locomo_path", ""))
        key["drop_categories"] = getattr(args, "locomo_drop_categories", "")
    if dataset == "beam":
        key["beam_scale"] = getattr(args, "beam_scale", "100K")
        key["beam_pairs_per_session"] = getattr(args, "beam_pairs_per_session", 6)
        # The turn cap changes transcripts (hence fact_lines), so it must key
        # the tasks. Read from env because the same env var drives the
        # distiller in both the warm and ingest paths.
        key["distill_max_turn_chars"] = os.environ.get(
            "MEMBUKKIT_DISTILL_MAX_TURN_CHARS", "600"
        )
    blob = json.dumps(key, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _task_cache_path(args, methods) -> Path:
    """Location of the persistent, content-addressed built-task cache.

    Keyed on inputs (not --output-dir), so a rerun with a different reader/judge
    or output directory reuses the same expensive retrieval build.
    """
    root = Path(getattr(args, "task_cache_dir", None) or ".membukkit_task_cache")
    return root / f"tasks_{_task_cache_key(args, methods)}.jsonl"


def _eval_cmd(args):
    """Run end-to-end evaluation."""
    from membukkit.llm.backends import GoogleBackend, LocalBackend, resolve_llm
    from membukkit.models.reranker import UtilityReranker
    from membukkit.reading.readers import (
        make_dated_reader,
        make_ordering_reader,
        make_recommendation_reader,
        make_reasoning_reader,
        make_abstain_gate,
        make_mem0_reader,
        make_mem0_judge,
        _normalize_abstain,
    )
    from membukkit.retrieval.router import (
        is_recommendation_query,
        is_reasoning_query,
        is_temporal_query,
    )
    from membukkit.eval.judges import make_judge_fn, make_official_judge

    methods = args.methods.split(",")
    replay = getattr(args, "replay_tasks", None)

    from membukkit.progress import ProgressFileWriter

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    progress_file = ProgressFileWriter(Path(args.output_dir) / "progress.json")
    progress_file.write("start", 0, 0, detail="eval starting", force=True)

    # Persistent, content-addressed cache of built retrieval tasks. Task building
    # (ingest + encode + cluster + rerank per query) is reader-independent and by
    # far the slowest non-LLM phase, so a rerun with a new reader/judge or output
    # dir should reuse it. A cache HIT is treated exactly like --replay-tasks:
    # the encoder/reranker/distiller are never loaded.
    task_cache_file = None
    if not replay and not getattr(args, "no_task_cache", False):
        task_cache_file = _task_cache_path(args, methods)
        if task_cache_file.exists():
            replay = str(task_cache_file)
            logger.info(f"task cache HIT -> reusing built tasks from {task_cache_file}")
        else:
            logger.info(f"task cache MISS -> will build and save to {task_cache_file}")

    if args.distill_cache == "distill_cache.json":
        _ds = getattr(args, "dataset", "longmemeval")
        if _ds == "locomo":
            args.distill_cache = "distill_cache_locomo.json"
        elif _ds == "beam":
            args.distill_cache = f"distill_cache_beam_{getattr(args, 'beam_scale', '100K')}.json"

    need_e16 = any(m.startswith("coremem") for m in methods)
    need_rerank = need_e16

    is_remote_encoder = str(args.encoder).startswith("openai:")
    if not replay:
        if need_e16 and is_remote_encoder:
            e16 = None  # remote embedding API; wrapped in _make_eval_encoder
        elif need_e16:
            from sentence_transformers import SentenceTransformer

            from membukkit.config import ModelConfig
            from membukkit.models.registry import resolve_encoder_path

            e16 = SentenceTransformer(resolve_encoder_path(ModelConfig(encoder=args.encoder)))
        else:
            e16 = None
        if need_rerank:
            from membukkit.config import ModelConfig
            from membukkit.models.registry import resolve_reranker_path

            reranker = UtilityReranker.load(
                resolve_reranker_path(ModelConfig(reranker=args.reranker))
            )
        else:
            reranker = None
    else:
        e16 = reranker = None

    # Distill conversation turns into atomic facts before retrieval (matches the
    # production MemorySystem.ingest path). Without this, retrieval runs over raw
    # dialogue turns, which measurably degrades accuracy.
    distiller = None
    if need_e16 and not replay:
        from membukkit.extraction.distiller import FactDistiller

        distiller = FactDistiller(
            resolve_llm(args.distill_model),
            cache_path=args.distill_cache,
        )

    def _backend_of(spec: str):
        try:
            return resolve_llm(spec)
        except Exception:
            return None

    # Rate-limited hosted providers (Gemini/Gemma free tier) fall over at 8
    # concurrent workers -> 429s that silently become wrong answers; local models
    # (Ollama) run on one machine and just thrash under high parallelism. So
    # auto-throttle for Google and local backends. --llm-workers always wins.
    _rb, _jb = _backend_of(args.reader), _backend_of(args.judge)
    uses_google = isinstance(_rb, GoogleBackend) or isinstance(_jb, GoogleBackend)
    uses_local = isinstance(_rb, LocalBackend) or isinstance(_jb, LocalBackend)
    if getattr(args, "llm_workers", 0):
        llm_workers = args.llm_workers
    elif uses_local or uses_google:
        llm_workers = 2
    else:
        llm_workers = LLM_WORKERS

    # --build-only runs retrieval (and trace dumping) with NO reader/judge LLM
    # calls, so it must not even resolve those backends (no API keys needed).
    build_only = bool(getattr(args, "build_only", False))

    answer_fn = rec_fn = reason_fn = ordering_fn = None
    mem0_reader_fn = None
    mem0_judge_fn = None
    official = getattr(args, "official_judge", False)

    def _zero_judge(*_a, **_k) -> float:
        return 0.0  # placeholder; never called (--build-only returns before judging)

    judge_fn: Callable[..., float] = _zero_judge
    if not build_only:
        dated_tpl = reason_tpl = None
        pv = getattr(args, "reader_prompts", "v1")
        if pv == "v2":
            from membukkit.prompts.reading import (
                DATED_READER_PROMPT_V2,
                REASONING_READER_PROMPT_V2,
            )

            dated_tpl = DATED_READER_PROMPT_V2
            reason_tpl = REASONING_READER_PROMPT_V2
        elif pv == "v3":
            from membukkit.prompts.reading import (
                DATED_READER_PROMPT_V3,
                REASONING_READER_PROMPT_V3,
            )

            dated_tpl = DATED_READER_PROMPT_V3
            reason_tpl = REASONING_READER_PROMPT_V3
        answer_fn = make_dated_reader(resolve_llm(args.reader), prompt_template=dated_tpl)
        rec_fn = make_recommendation_reader(resolve_llm(args.reader))
        reason_fn = make_reasoning_reader(resolve_llm(args.reader), prompt_template=reason_tpl)
        ordering_fn = make_ordering_reader(
            resolve_llm(args.reader),
            amb_mode=getattr(args, "beam_ordering_mode", "official") == "amb",
        )

        reader_protocol = getattr(args, "reader_protocol", "coremem")
        judge_protocol = getattr(args, "judge_protocol", "coremem")
        if reader_protocol == "mem0":
            mem0_reader_fn = make_mem0_reader(resolve_llm(args.reader))
        if judge_protocol == "mem0":
            mem0_judge_fn = make_mem0_judge(resolve_llm(args.judge))

        if official:
            judge_fn = make_official_judge(resolve_llm(args.judge))
        else:
            judge_fn = make_judge_fn(resolve_llm(args.judge))

    # BEAM uses the benchmark's own vendored judge (gpt-4.1-mini @ temp 0,
    # official prompts and aggregation) instead of the LongMemEval judges.
    beam_judge_invoke = None
    if getattr(args, "dataset", "longmemeval") == "beam" and not build_only:
        from membukkit.eval.beam_official import make_openai_judge

        beam_judge_model = args.judge.split(":", 1)[-1]
        if beam_judge_model == "gpt-4o":  # parser default; BEAM's official judge
            beam_judge_model = "gpt-4.1-mini"
        beam_judge_invoke = make_openai_judge(model=beam_judge_model)
        logger.info(f"BEAM official judge: {beam_judge_model} @ temperature 0")

    dataset = getattr(args, "dataset", "longmemeval")
    if replay:
        with open(replay) as f:
            all_tasks = [json.loads(line) for line in f if line.strip()]
        logger.info(f"REPLAY: loaded {len(all_tasks)} cached tasks from {replay}")
    else:
        if dataset == "locomo":
            from membukkit.data.locomo import load_locomo

            drop = (
                [int(x) for x in str(args.locomo_drop_categories).split(",") if x.strip()]
                if getattr(args, "locomo_drop_categories", "")
                else None
            )
            lme = load_locomo(args.locomo_path, drop_categories=drop)
        elif dataset == "beam":
            from membukkit.data.beam import load_beam_qa

            lme = load_beam_qa(
                scale=getattr(args, "beam_scale", "100K"),
                pairs_per_session=getattr(args, "beam_pairs_per_session", 6),
            )
        else:
            from membukkit.data.longmemeval import load_longmemeval

            lme = load_longmemeval(max_instances=None)

        selected = list(lme.instances)
        if args.max_instances:
            selected = selected[: args.max_instances]
        if getattr(args, "only_ids", ""):
            want = {s.strip() for s in args.only_ids.split(",") if s.strip()}
            selected = [i for i in selected if i.question_id in want]
            missing = want - {i.question_id for i in selected}
            if missing:
                raise SystemExit(f"--only-ids: {len(missing)} ids not in dataset: {sorted(missing)[:5]}")
        logger.info(f"{len(selected)} instances; reader={args.reader} judge={args.judge}")

        if getattr(args, "bucket_mode", None) in ("topic", "multiaxis"):
            from membukkit.retrieval.buckets import reset_scan_stats

            reset_scan_stats()

        if distiller is not None:
            _warm_distiller(
                selected,
                distiller,
                getattr(args, "distill_workers", 16),
                progress_file=progress_file,
            )

        if not any(m.startswith("coremem") for m in methods):
            raise SystemExit(
                "membukkit eval requires a coremem* method (coremem_union / coremem / "
                "coremem_atomic); the legacy bm25/dense baselines were removed with the "
                "legacy retrieval engine."
            )
        logger.info("retrieval engine: library (production MemorySystem union)")
        if is_remote_encoder:
            from membukkit.models.openai_encoder import make_openai_encoder

            eval_encoder = make_openai_encoder(args.encoder)
            logger.info(f"encoder: remote {args.encoder} (disk-cached embeddings)")
        else:
            eval_encoder = _STAdapter(e16)
        all_tasks = _build_all_tasks_lib(
            selected,
            eval_encoder,
            reranker,
            args,
            methods,
            distiller,
            progress_file=progress_file,
        )

        if distiller is not None:
            distiller.save()

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        tc = Path(args.output_dir) / "tasks_cache.jsonl"
        with open(tc, "w") as f:
            for t in all_tasks:
                f.write(json.dumps(t) + "\n")
        logger.info(f"Cached {len(all_tasks)} tasks -> {tc}")

        # Also persist to the content-addressed cache so future runs (any reader,
        # any output dir) reuse this build. Write atomically to avoid a partial
        # file being treated as a valid cache hit if interrupted.
        if task_cache_file is not None:
            task_cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = task_cache_file.with_suffix(task_cache_file.suffix + ".tmp")
            with open(tmp, "w") as f:
                for t in all_tasks:
                    f.write(json.dumps(t) + "\n")
            os.replace(tmp, task_cache_file)
            logger.info(f"Saved built tasks to persistent cache -> {task_cache_file}")

    # Per-query retrieval traces (opened buckets, route probs, raw cosines,
    # gold/blocked bucket mapping) — the transparency artifact for audits.
    trace_rows = [t for t in all_tasks if t.get("retrieval_trace") is not None]
    if trace_rows:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        tp = Path(args.output_dir) / "retrieval_trace.jsonl"
        with open(tp, "w") as f:
            for t in trace_rows:
                f.write(
                    json.dumps(
                        {
                            k: t.get(k)
                            for k in (
                                "method",
                                "ability",
                                "instance_id",
                                "question",
                                "gold_buckets",
                                "blocked_buckets",
                                "retrieval_trace",
                            )
                        }
                    )
                    + "\n"
                )
        logger.info(f"Wrote {len(trace_rows)} retrieval traces -> {tp}")

    if build_only:
        logger.info(f"--build-only: built {len(all_tasks)} tasks; skipping reader/judge.")
        return

    logger.info(f"Running {len(all_tasks)} answer+judge tasks ({llm_workers} workers)...")
    progress_file.write("answer", 0, len(all_tasks), force=True)

    # Surface (rather than silently swallow) LLM failures: a rate-limited reader
    # call that falls back to "N/I" is scored wrong, which looks like a bad model
    # but is really an infra problem. We log the first few and tally the rest.
    _fail_lock = threading.Lock()
    _fail_state = {"reader": 0, "judge": 0, "logged": 0}
    _FAIL_LOG_CAP = 5

    def _note_fail(kind: str, err: Exception) -> None:
        with _fail_lock:
            _fail_state[kind] += 1
            if _fail_state["logged"] < _FAIL_LOG_CAP:
                _fail_state["logged"] += 1
                logger.warning(f"{kind} call failed ({type(err).__name__}: {str(err)[:120]})")

    def _run_task(t):
        is_rec = False
        if mem0_reader_fn is not None:
            reader, reader_tag = mem0_reader_fn, "mem0"
        else:
            is_rec = is_recommendation_query(t["question"])
            is_reason_q = (not is_rec) and is_reasoning_query(t["question"])
            if is_rec:
                reader, reader_tag = rec_fn, "rec"
            elif is_reason_q:
                reader, reader_tag = reason_fn, "reason"
            else:
                reader, reader_tag = answer_fn, "dated"
        assert reader is not None  # --build-only returns before tasks run
        reader_question = t["question"]
        # BEAM event_ordering is scored by list alignment over
        # response.split("\n") + Kendall tau; prose answers score near zero
        # regardless of content, so the reader must emit a bare list.
        # BEAM per-category answer instructions. These shape ANSWER FORM only
        # (the judging is untouched official code); category tags are part of
        # the published dataset and the instructions ship in our receipts.
        if dataset == "beam":
            if t.get("qtype") == "event_ordering" and ordering_fn is not None:
                reader, reader_tag = ordering_fn, "ordering"
            else:
                _hint = _BEAM_READER_HINTS.get(t.get("qtype", ""))
                if _hint:
                    reader_question = t["question"] + "\n" + _hint
        reader_err = False
        try:
            ans = (
                reader(t["fact_lines"], reader_question, t.get("qdate", ""))
                if t["fact_lines"]
                else "N/I"
            )
        except Exception as e:  # noqa: BLE001
            ans = "N/I"
            reader_err = True
            _note_fail("reader", e)
        ans = _normalize_abstain(ans)
        judge_err = False
        beam_record = None
        if beam_judge_invoke is not None:
            from membukkit.eval.beam_official import evaluate_question

            category = t.get("qtype", "")
            try:
                beam_record = evaluate_question(
                    category, t.get("beam_rubric") or [], ans, t["question"], beam_judge_invoke
                )
                # Scalar mirror of the official aggregation (report_results.py):
                # tau_norm for event_ordering, nugget mean elsewhere.
                j = (
                    beam_record["tau_norm"]
                    if category == "event_ordering"
                    else beam_record["llm_judge_score"]
                )
            except Exception as e:  # noqa: BLE001
                j = 0.0
                judge_err = True
                _note_fail("judge", e)
            return {
                **t,
                "answer": ans,
                "judge": j,
                "beam_record": beam_record,
                "reader": reader_tag,
                "_reader_err": reader_err,
                "_judge_err": judge_err,
            }
        try:
            if mem0_judge_fn is not None:
                j = mem0_judge_fn(
                    t["question"], t["ground_truth"], ans, category=t.get("locomo_category")
                )
            elif official:
                j = judge_fn(
                    t["question"],
                    t["ground_truth"],
                    ans,
                    question_type=t.get("qtype", ""),
                    abstention=t.get("is_abstention", False),
                )
            else:
                j = judge_fn(t["question"], ans, t["ground_truth"])
        except Exception as e:  # noqa: BLE001
            j = 0.0
            judge_err = True
            _note_fail("judge", e)
        return {
            **t,
            "answer": ans,
            "judge": j,
            "reader": reader_tag,
            "_reader_err": reader_err,
            "_judge_err": judge_err,
        }

    results = []
    correct = 0.0
    fails = 0
    bar = _pbar(total=len(all_tasks), desc="answer+judge")
    with ThreadPoolExecutor(max_workers=llm_workers) as ex:
        futs = [ex.submit(_run_task, t) for t in all_tasks]
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            correct += float(r.get("judge") or 0.0)
            if r.get("_reader_err") or r.get("_judge_err"):
                fails += 1
            if bar is not None:
                bar.update(1)
                bar.set_postfix(acc=f"{correct / len(results):.3f}", fail=fails, refresh=False)
            elif len(results) % 50 == 0:
                logger.info(f"  completed {len(results)}/{len(all_tasks)} (fails={fails})")
            progress_file.write(
                "answer",
                len(results),
                len(all_tasks),
                detail=f"acc={correct / len(results):.3f}",
                force=(len(results) >= len(all_tasks)),
            )
    if bar is not None:
        bar.close()
    progress_file.write("done", len(results), len(all_tasks), force=True)

    if fails:
        logger.warning(
            f"{fails}/{len(results)} tasks hit an LLM error "
            f"(reader={_fail_state['reader']}, judge={_fail_state['judge']}) and were scored "
            f"N/I or 0 — likely rate limits. Lower --llm-workers or check your API quota; "
            f"results above are NOT reliable."
        )

    _report(results, methods, args)


def _warm_distiller(selected, distiller, workers, progress_file=None):
    """Parallel pre-distill every session before retrieval (ports coremem3 run()).

    Without this the CLI distills lazily and serially inside the retrieval loop,
    which is prohibitively slow on LongMemEval (hundreds of independent haystacks).
    Jobs are deduped by content-hash key so shared-haystack datasets (LoCoMo, where
    many QA share one conversation) distill each session exactly once. The key
    scheme MUST match MemorySystem.ingest (make_key("ingest", s_idx, transcript))
    so the warmed entries are cache hits when the library engine ingests.
    """
    from membukkit.extraction.distiller import build_transcript, make_key
    from membukkit.time_utils import parse_datetime, format_prompt_date

    jobs = []
    seen = set()
    for inst in selected:
        item = getattr(inst, "_item", {})
        sessions = item.get("haystack_sessions", [])
        dates = item.get("haystack_dates", [])
        for s_idx, session in enumerate(sessions):
            turns = [(t.get("role", "user"), t.get("content", "")) for t in session]
            transcript = build_transcript(turns, numbered=True)
            if not transcript.strip():
                continue
            key = make_key("ingest", s_idx, transcript)
            if key in seen:
                continue
            seen.add(key)
            ts = parse_datetime(dates[s_idx] if s_idx < len(dates) else None)
            jobs.append((key, transcript, format_prompt_date(ts)))
    if jobs:
        logger.info(
            f"Distilling atomic facts ({len(jobs)} unique session jobs, workers={workers})..."
        )
        bar_state: Dict[str, Any] = {"bar": None}

        def _cb(done: int, total: int) -> None:
            if bar_state["bar"] is None:
                bar_state["bar"] = _pbar(total=total, desc="distill")
            b = bar_state["bar"]
            if b is not None:
                b.update(1)
            if progress_file is not None:
                progress_file.write("distill", done, total, force=(done >= total))

        distiller.warm(jobs, workers=workers, progress_cb=_cb)
        if bar_state["bar"] is not None:
            bar_state["bar"].close()


def _fmt_qdate(inst):
    from membukkit.time_utils import format_prompt_date, parse_datetime

    return format_prompt_date(parse_datetime(inst.question_date_raw))


class _STAdapter:
    """Expose a raw SentenceTransformer through the membukkit Encoder API.

    Lets the library path (`MemorySystem`/`InMemoryBackend`, which call
    ``encode(texts, normalize=True)``) reuse the single already-loaded model.
    Byte-identical to `membukkit.models.encoder.Encoder`.
    """

    def __init__(self, st):
        self._st = st

    def encode(self, texts, normalize: bool = True, show_progress: bool = False):
        return np.asarray(
            self._st.encode(texts, show_progress_bar=show_progress, normalize_embeddings=normalize)
        )


def _lanes_for_method(method: str):
    """Map an eval method to the union lanes retrieval reads from."""
    if method == "coremem_atomic":
        return ("atomic",)
    if method == "coremem":
        return ("verbatim",)
    return ("verbatim", "atomic")  # coremem_union (and any coremem* default)


def _retrieval_cfg_from_args(args):
    """Build a RetrievalConfig from eval args, matching the eval defaults."""
    from membukkit.config import RetrievalConfig

    return RetrievalConfig(
        union=True,
        bucket_mode=getattr(args, "bucket_mode", None) or "none",
        scan_budget=args.scan_budget,
        scan_budget_temporal=getattr(args, "scan_budget_temporal", None),
        scan_budget_reason=getattr(args, "scan_budget_reason", 0.45),
        num_buckets=getattr(args, "bucket_k", 24),
        k_proto=getattr(args, "bucket_k_proto", 0),
        select=getattr(args, "rerank_select", "hybrid"),
        rerank_cap=getattr(args, "bucket_rerank_cap", 50),
        candidate_pool=getattr(args, "cand", 50),
        top_k=args.top_k,
        reasoning_top_k=args.reasoning_top_k,
        k_rrf=60,
    )


def _make_lib_backend(args, encoder, conv_key: str, cfg):
    """Backend for one conversation's MemorySystem in the library engine.

    memory -> None (MemorySystem defaults to InMemoryBackend). turbopuffer ->
    a TurbopufferBackend over an EPHEMERAL namespace derived from conv_key, so
    the real production path (per-kind partitions, gated routing, RRF) is
    exercised end-to-end. The caller deletes the namespace afterwards.
    """
    if getattr(args, "storage_backend", "memory") != "turbopuffer":
        return None
    from membukkit.config import StorageConfig
    from membukkit.storage.turbopuffer import TurbopufferBackend

    region = getattr(args, "region", None) or os.environ.get("TURBOPUFFER_REGION")
    api_key = os.environ.get("TURBOPUFFER_API_KEY")
    if not api_key:
        raise SystemExit(
            "storage-backend=turbopuffer requires TURBOPUFFER_API_KEY (and a region "
            "via --region or TURBOPUFFER_REGION)."
        )
    prefix = getattr(args, "namespace_prefix", "evaltmp_")
    ns = f"{prefix}{hashlib.sha1(str(conv_key).encode()).hexdigest()[:16]}"
    storage = StorageConfig(
        backend="turbopuffer",
        namespace=ns,
        region=region,
        api_key=api_key,
        vector_dtype=getattr(args, "vector_dtype", "f16"),
    )
    return TurbopufferBackend(cfg, encoder, storage)


def _lane_views(backend) -> Dict[str, Dict]:
    """Per-lane inspection views with id/text -> bucket lookup maps.

    Built once per ingested conversation (the partition is bank-level, not
    query-level) and reused for every question in the group.
    """
    if not hasattr(backend, "lane_view"):
        return {}
    views: Dict[str, Dict] = {}
    for kind in ("verbatim", "atomic"):
        v = backend.lane_view(kind)
        if not v:
            continue
        v["_id_to_bucket"] = dict(zip(v["ids"], v["labels"]))
        v["_text_to_bucket"] = {
            " ".join(t.lower().split()): lbl for t, lbl in zip(v["texts"], v["labels"])
        }
        views[kind] = v
    return views


def _gold_bucket_map(inst, lane_views: Dict[str, Dict]) -> Dict[str, List[int]]:
    """Map an instance's gold evidence to lane-local topic buckets.

    Verbatim lane: each gold turn maps to its stored row by content id (text
    fallback covers turns whose id shifted through date normalisation or
    dedup), then to the partition label routing actually uses. Atomic lane:
    any distilled fact whose source ingest session is a gold session.
    """
    if not hasattr(inst, "get_gold_fact_indices"):
        return {}
    from membukkit.storage.base import content_id

    out: Dict[str, List[int]] = {}
    facts = inst.get_facts()
    gold_idx = inst.get_gold_fact_indices()
    vv = lane_views.get("verbatim")
    if vv is not None:
        buckets = set()
        for i in gold_idx:
            f = facts[i]
            b = vv["_id_to_bucket"].get(content_id(f.text, None, f.timestamp, "verbatim"))
            if b is None:
                b = vv["_text_to_bucket"].get(" ".join(f.text.lower().split()))
            if b is not None:
                buckets.add(int(b))
        out["verbatim"] = sorted(buckets)
    av = lane_views.get("atomic")
    # Source ids are `ingest:{idx}:{hash}`; the hash makes provenance unique
    # across ingest calls. Older stores wrote a bare `ingest:{idx}`, so match
    # the session-index component rather than the whole string.
    gold_idxs = {str(s) for s in getattr(inst, "_gold_session_set", set())}

    def _is_gold(src: str) -> bool:
        parts = (src or "").split(":")
        return len(parts) >= 2 and parts[0] == "ingest" and parts[1] in gold_idxs

    if av is not None:
        out["atomic"] = sorted(
            {int(lbl) for src, lbl in zip(av["sources"], av["labels"]) if _is_gold(src)}
        )
    return out


def _blocked_buckets(
    gold_map: Dict[str, List[int]],
    mode: str,
    lane_views: Dict[str, Dict],
    seed_key: str,
) -> Optional[Dict[str, List[int]]]:
    """Pick the buckets to close for one question under a --block-buckets arm.

    gold: close exactly the buckets holding gold evidence (causal test).
    random: close the SAME NUMBER of non-gold buckets per lane, seeded by the
    question id (specificity control — removing arbitrary topics should not
    hurt).
    """
    if mode == "none" or not gold_map:
        return None
    if mode == "gold":
        blocked = {lane: ids for lane, ids in gold_map.items() if ids}
        return blocked or None
    rng = random.Random(f"block:{seed_key}")
    blocked = {}
    for lane, gold in gold_map.items():
        if not gold:
            continue
        k_eff = int((lane_views.get(lane) or {}).get("k_eff", 0))
        pool = [b for b in range(k_eff) if b not in set(gold)]
        if not pool:
            continue
        blocked[lane] = sorted(rng.sample(pool, min(len(gold), len(pool))))
    return blocked or None


def _beam_verbatim_timeline(mem, question, k=80, snip=220):
    """Dated verbatim turn openers, relevance-ranked over the verbatim lane.

    BEAM event-ordering gold items are the generator's plan bullets, and the
    conversation's turn openers echo them near-verbatim (measured: ~70% of
    expected items concentrate in a single turn's opening lines, both roles).
    Fact distillation compresses those beats away, so the ordering reader
    gets them back as quotes: a FULL cosine scan of the verbatim lane (the
    raw turns already ingested by the union method), top-k openers, dated.
    Deliberately bypasses topic routing — the scan budget would drop
    sessions, and this lane is small enough to scan whole.
    """
    import numpy as np

    backend = getattr(mem, "_backend", None)
    idxs = getattr(backend, "_kind_idx", {}).get("verbatim", []) if backend else []
    embs = getattr(backend, "_embs", None) if backend else None
    if embs is None or not idxs:
        return []
    qe = np.asarray(backend._encoder.encode(question, normalize=True), dtype=np.float32)
    sub = embs[idxs]
    order = np.argsort(sub @ qe)[::-1][: min(k, len(idxs))]
    lines, seen = [], set()
    for loc in order:
        g = idxs[int(loc)]
        head = " ".join((backend._texts[g] or "").split())[:snip]
        if not head or head in seen:
            continue
        seen.add(head)
        ts = backend._times[g] if g < len(backend._times) else None
        date = ts.strftime("%Y-%m-%d") if ts else ""
        lines.append(f"[{date}] QUOTE: {head}")
    return lines


def _build_all_tasks_lib(
    selected, encoder, reranker, args, methods, distiller=None, progress_file=None
):
    """Build retrieval tasks by driving the production `MemorySystem`.

    This is the convergence path: retrieval goes through exactly the same
    ingest + union `answer()` the service/dashboard use, so the eval can never
    silently diverge from production. Facts come from
    ``answer(generate_answer=False)``; the reader/judge are applied downstream
    in ``_run_task`` (protocol-specific), identical to the legacy path.

    With --storage-backend turbopuffer each haystack is ingested into an
    ephemeral cloud namespace (per-kind partitions warmed via recluster), queried,
    then deleted — a real smoke test of the production Turbopuffer path.
    """
    from membukkit.config import PromptConfig
    from membukkit.pipeline import MemorySystem
    from membukkit.progress import ProgressEvent

    if getattr(args, "storage_backend", "memory") == "turbopuffer":
        logger.info(
            "storage backend: turbopuffer (ephemeral per-conversation namespaces, prefix=%s)",
            getattr(args, "namespace_prefix", "evaltmp_"),
        )

    block_mode = getattr(args, "block_buckets", "none") or "none"
    dump_traces = bool(getattr(args, "dump_traces", False))
    want_gold = block_mode != "none" or dump_traces
    if block_mode != "none" and getattr(args, "storage_backend", "memory") != "memory":
        raise SystemExit(
            "--block-buckets requires --storage-backend memory (gold->bucket mapping "
            "uses the in-memory lane views)."
        )
    if block_mode != "none":
        logger.info(
            "bucket intervention: block-%s (seed=%s)", block_mode, getattr(args, "block_seed", 0)
        )

    # Group instances by conversation so each haystack is ingested once (LoCoMo
    # shares one haystack across ~150 QA; LongMemEval has one per instance).
    groups: Dict[str, List] = {}
    for inst in selected:
        conv_key = getattr(inst, "distill_scope_id", None) or inst.question_id
        groups.setdefault(conv_key, []).append(inst)

    all_tasks = []
    prompts = PromptConfig.default()
    bar = _pbar(total=len(selected), desc="retrieval (build tasks)")
    if progress_file is not None:
        progress_file.write("retrieve", 0, len(selected), force=True)
    done_q = 0
    for gi, (conv_key, insts) in enumerate(groups.items()):
        item = getattr(insts[0], "_item", {})
        sessions = item.get("haystack_sessions", [])
        dates = item.get("haystack_dates", [])
        if not sessions:
            if bar is not None:
                bar.update(len(insts))
            done_q += len(insts)
            if progress_file is not None:
                progress_file.write("retrieve", done_q, len(selected))
            continue

        cfg = _retrieval_cfg_from_args(args)
        backend = _make_lib_backend(args, encoder, conv_key, cfg)
        mem = MemorySystem(
            encoder=encoder,
            reranker=reranker,
            llm_fn=lambda p: "",  # reader skipped (generate_answer=False)
            retrieval=cfg,
            prompts=prompts,
            distiller=distiller,
            backend=backend,
        )
        try:
            def _ingest_progress(ev: ProgressEvent, _ck=conv_key):
                if progress_file is None:
                    return
                progress_file.write(
                    "retrieve",
                    done_q,
                    len(selected),
                    detail=f"{_ck}: {ev.phase} {ev.done}/{ev.total}",
                )

            mem.ingest(
                sessions=sessions,
                dates=dates,
                on_progress=_ingest_progress if progress_file is not None else None,
            )
            if backend is not None:
                # Warm per-kind topic partitions so gated routing exercises the
                # per-kind centroids (not a cold full-kind scan).
                backend.maybe_recluster()

            # Lane views (id/text -> bucket) are bank-level: build once per
            # conversation, reuse for every question in the group.
            lane_views = _lane_views(mem._backend) if want_gold else {}

            for inst in insts:
                if bar is not None:
                    bar.update(1)
                done_q += 1
                if progress_file is not None:
                    progress_file.write("retrieve", done_q, len(selected))
                queries = inst.get_queries()
                if not queries:
                    continue
                q = queries[0]
                gt = q.ground_truth or ""
                if not gt:
                    continue

                iitem = getattr(inst, "_item", {})
                qdate = _fmt_qdate(inst)
                qtype = iitem.get("question_type", "")
                is_abs = str(inst.question_id).endswith("_abs")

                gold_map = _gold_bucket_map(inst, lane_views) if want_gold else {}
                blocked = _blocked_buckets(
                    gold_map,
                    block_mode,
                    lane_views,
                    f"{getattr(args, 'block_seed', 0)}:{inst.question_id}",
                )
                # Aggregation routing experiment: counting queries need every
                # mention, so widen retrieval (deeper top-k / bigger scan) for
                # them only. cfg is this conversation's private config and task
                # building is sequential, so per-query mutation is safe.
                agg_top_k = getattr(args, "agg_top_k", 0)
                agg_scan = getattr(args, "agg_scan_budget", None)
                is_agg = False
                if agg_top_k or agg_scan:
                    from membukkit.retrieval.router import has_aggregation_cues

                    is_agg = has_aggregation_cues(q.text)
                # Deep routing experiment: needle queries (router-TEMPORAL, or
                # assistant-recall by query-surface cue) are recall-bound, so
                # widen retrieval for them only; counting/preference queries
                # keep the standard depth because extra distractors measurably
                # hurt them (see deep_54targeted_full per-type results).
                deep_top_k = getattr(args, "deep_top_k", 0)
                deep_scan = getattr(args, "deep_scan_budget", None)
                is_deep = False
                if deep_top_k or deep_scan:
                    from membukkit.retrieval.router import (
                        has_assistant_recall_cues,
                        has_broad_context_cues,
                        is_temporal_query,
                    )

                    is_deep = is_temporal_query(q.text) or has_assistant_recall_cues(
                        q.text
                    )
                    # --deep-broad: coverage-bound synthesis queries
                    # ("summarize...", "how has X evolved") also route deep.
                    if not is_deep and getattr(args, "deep_broad", False):
                        is_deep = has_broad_context_cues(q.text)

                for method in methods:
                    trace_payload = None
                    if method.startswith("coremem"):
                        cfg.union_lanes = _lanes_for_method(method)
                        saved = (
                            cfg.scan_budget,
                            cfg.scan_budget_temporal,
                            cfg.scan_budget_reason,
                            cfg.reasoning_top_k,
                            cfg.candidate_pool,
                            cfg.rerank_cap,
                        )
                        if is_agg:
                            if agg_scan:
                                cfg.scan_budget = agg_scan
                                cfg.scan_budget_temporal = agg_scan
                            if agg_top_k:
                                cfg.reasoning_top_k = max(cfg.reasoning_top_k, agg_top_k)
                        if is_deep:
                            if deep_scan:
                                cfg.scan_budget = deep_scan
                                cfg.scan_budget_temporal = deep_scan
                                cfg.scan_budget_reason = deep_scan
                            if deep_top_k:
                                cfg.reasoning_top_k = max(cfg.reasoning_top_k, deep_top_k)
                                # A deeper top-k is useless if the candidate pool
                                # or rerank cap clips the list first.
                                cfg.candidate_pool = max(cfg.candidate_pool, 2 * deep_top_k)
                                cfg.rerank_cap = max(cfg.rerank_cap, 2 * deep_top_k)
                        try:
                            res = mem.answer(
                                q.text, generate_answer=False, exclude_buckets=blocked
                            )
                        finally:
                            (
                                cfg.scan_budget,
                                cfg.scan_budget_temporal,
                                cfg.scan_budget_reason,
                                cfg.reasoning_top_k,
                                cfg.candidate_pool,
                                cfg.rerank_cap,
                            ) = saved
                        lines = res.facts
                        if dump_traces:
                            trace_payload = {
                                "lanes": res.trace.lanes,
                                "scan_fraction": res.trace.scan_fraction,
                                "n_facts": res.trace.n_facts,
                                "n_scanned": res.trace.n_scanned,
                            }
                    else:
                        lines = []
                    task = {
                        "method": method,
                        "ability": inst.ability,
                        "instance_id": inst.question_id,
                        "question": q.text,
                        "ground_truth": gt,
                        "fact_lines": lines,
                        "qdate": qdate,
                        "qtype": qtype,
                        "is_abstention": is_abs,
                        "locomo_category": iitem.get("locomo_category"),
                    }
                    if getattr(inst, "beam_rubric", None) is not None:
                        task["beam_rubric"] = inst.beam_rubric
                        task["beam_conv"] = iitem.get("beam_conv", "")
                        if (
                            inst.ability == "event_ordering"
                            and getattr(args, "beam_ordering_mode", "official") == "amb"
                        ):
                            task["fact_lines"] = list(lines) + _beam_verbatim_timeline(
                                mem, q.text
                            )
                    if want_gold:
                        task["gold_buckets"] = gold_map
                        task["blocked_buckets"] = blocked
                    if trace_payload is not None:
                        task["retrieval_trace"] = trace_payload
                    all_tasks.append(task)
        finally:
            if backend is not None and not getattr(args, "keep_namespaces", False):
                try:
                    backend.delete()  # drop the ephemeral cloud namespace
                except Exception as e:
                    logger.warning("failed to delete eval namespace: %s", e)
    if bar is not None:
        bar.close()
    return all_tasks


def _beam_report(results, args):
    """Official BEAM aggregation (report_results.py semantics) + receipts.

    Per conversation: mean over that conversation's records per category
    (tau_norm for event_ordering, llm_judge_score otherwise — already
    mirrored into r["judge"]). Scale score: mean over conversations, then
    mean over the 10 categories.
    """
    from membukkit.eval.beam_official import aggregate_scores

    per_conv: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r.get("beam_record") is None:
            continue
        per_conv[r.get("beam_conv", "?")][r.get("qtype", "?")].append(r["beam_record"])

    convs = sorted(per_conv)
    agg = aggregate_scores([per_conv[c] for c in convs])

    print(f"\n{'=' * 72}\nBEAM OFFICIAL SCORING (scale={getattr(args, 'beam_scale', '?')}, "
          f"{len(convs)} conversations)\n{'=' * 72}")
    for cat in sorted(k for k in agg if k != "average"):
        print(f"{cat:28s} {agg[cat]:.3f}")
    print(f"{'-' * 40}\n{'AVERAGE':28s} {agg['average']:.3f}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "beam_summary.json").write_text(
        json.dumps(
            {
                "scale": getattr(args, "beam_scale", None),
                "n_conversations": len(convs),
                "n_questions": sum(len(v) for c in per_conv.values() for v in c.values()),
                "scores": agg,
                "per_conversation": {
                    c: aggregate_scores([per_conv[c]]) for c in convs
                },
            },
            indent=2,
        )
    )
    logger.info(f"Wrote {out / 'beam_summary.json'}")


def _report(results, methods, args):
    """Print evaluation report and save summary."""
    if any(r.get("beam_record") is not None for r in results):
        _beam_report(results, args)
    by_method = defaultdict(list)
    by_am = defaultdict(list)
    for r in results:
        by_method[r["method"]].append(r["judge"])
        by_am[(r["ability"], r["method"])].append(r["judge"])

    print(
        f"\n{'=' * 72}\nMEMBUKKIT EVALUATION (acc@{CORRECT_THRESHOLD})  reader={args.reader} judge={args.judge}\n{'=' * 72}"
    )
    print(f"{'method':16s} {'acc':>8s} {'judge':>8s} {'n':>6s}")
    for m in methods:
        s = by_method.get(m, [])
        if s:
            acc = np.mean([x >= CORRECT_THRESHOLD for x in s])
            print(f"{m:16s} {acc:8.3f} {np.mean(s):8.3f} {len(s):6d}")

    print(f"\n{'-' * 72}\nBY ABILITY\n{'-' * 72}")
    abil = sorted({a for (a, _) in by_am})
    print(f"{'ability':18s}" + "".join(f"{m[:14]:>15s}" for m in methods))
    for a in abil:
        row = f"{a:18s}"
        for m in methods:
            s = by_am.get((a, m), [])
            row += f"{np.mean([x >= CORRECT_THRESHOLD for x in s]):15.3f}" if s else f"{'-':>15s}"
        print(row)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _jsonable(v):
        return isinstance(v, (str, int, float, bool, type(None), list, dict))

    summary = {
        "config": {
            k: v
            for k, v in vars(args).items()
            if not k.startswith("_") and k != "func" and _jsonable(v)
        },
        "overall": {
            m: {
                "acc": float(np.mean([x >= CORRECT_THRESHOLD for x in by_method[m]])),
                "judge_mean": float(np.mean(by_method[m])),
                "n": len(by_method[m]),
            }
            for m in methods
            if by_method.get(m)
        },
        "by_ability": {
            f"{a}|{m}": float(np.mean([x >= CORRECT_THRESHOLD for x in by_am[(a, m)]]))
            for (a, m) in by_am
        },
    }
    (out / "e2e_summary.json").write_text(json.dumps(summary, indent=2))
    with open(out / "e2e_results.jsonl", "w") as f:
        for r in results:
            f.write(
                json.dumps(
                    {
                        k: r.get(k)
                        for k in (
                            "method",
                            "ability",
                            "qtype",
                            "is_abstention",
                            "instance_id",
                            "question",
                            "ground_truth",
                            "answer",
                            "judge",
                            "beam_conv",
                            "beam_record",
                        )
                        if not (k.startswith("beam_") and r.get(k) is None)
                    }
                )
                + "\n"
            )
    logger.info(f"Wrote {out / 'e2e_summary.json'} and e2e_results.jsonl")


RAG_LLM_WORKERS = 16
RAG_READER_MODEL = "gpt-4o-mini"


def _rag_eval_cmd(args):
    """Run multi-hop RAG evaluation (EM/F1/Recall@k)."""
    import time
    import random
    from membukkit.data.multihop import load_multihop, _ALIASES
    from membukkit.retrieval.multihop import SubstrateEncoder, DenseRetriever, CoreMemRetriever
    from membukkit.eval.qa_scorer import score_qa, score_retrieval, exact_match as _em
    from membukkit.reading.qa_reader import make_qa_reader, _call_with_retry

    try:
        from tqdm.auto import tqdm as _tqdm_cls

        _tqdm: Any = _tqdm_cls
    except Exception:
        _tqdm = None

    def _pbar(iterable, total, desc):
        if _tqdm is not None:
            return _tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=False)
        return iterable

    def _canon(name):
        key = _ALIASES.get(name.strip().lower())
        if key is None:
            raise ValueError(f"Unknown dataset '{name}'.")
        return key

    from membukkit.llm.backends import make_llm_backend

    need_llm = (not args.no_reader) or args.coremem_decompose
    llm_fn = (
        make_llm_backend("openai", model=RAG_READER_MODEL, temperature=0.0) if need_llm else None
    )
    if args.no_reader:
        reader = None
    else:
        assert llm_fn is not None
        reader = make_qa_reader(llm_fn, verify=args.reader_verify)

    datasets = [_canon(x) for x in args.datasets.split(",") if x.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    workers = max(1, int(args.workers))

    all_results = {}

    for dataset_key in datasets:
        ds = load_multihop(dataset_key, data_dir=args.data_dir, max_questions=(args.smoke or None))

        _enc_cache = {}

        def _substrate_encoder():
            if "e" not in _enc_cache:
                _enc_cache["e"] = SubstrateEncoder(
                    args.embedder,
                    query_prompt=args.query_prompt,
                    trust_remote_code=args.trust_remote_code,
                    max_seq_length=args.max_seq_len,
                    batch_size=args.encode_batch_size,
                )
            return _enc_cache["e"]

        ds_results = {}
        for method in methods:
            coremem_shares = (
                method == "coremem"
                and args.coremem_encoder == args.embedder
                and args.coremem_query_prompt == args.query_prompt
            )
            encoder = _substrate_encoder() if (method == "dense" or coremem_shares) else None

            if method == "dense":
                assert encoder is not None
                retriever = DenseRetriever(encoder)
            elif method == "coremem":
                reuse = encoder
                retriever = CoreMemRetriever(
                    encoder_path=args.coremem_encoder,
                    reranker_path=args.coremem_reranker,
                    budget=args.coremem_budget,
                    bucket_k=args.coremem_bucket_k,
                    rerank_cap=args.coremem_rerank_cap,
                    fusion=args.coremem_fusion,
                    hops=args.coremem_hops,
                    expand_m=args.coremem_expand_m,
                    expand_mode=args.coremem_expand_mode,
                    axes=args.coremem_axes,
                    temporal=args.coremem_temporal,
                    entity_cap=args.coremem_entity_cap,
                    entity_rank=args.coremem_entity_rank,
                    entity_min=args.coremem_entity_min,
                    decompose=args.coremem_decompose,
                    max_subq=args.coremem_max_subq,
                    decompose_fuse=args.coremem_decompose_fuse,
                    decompose_iter=(not args.coremem_decompose_no_iter),
                    decompose_retrieval=args.coremem_decompose_retrieval,
                    llm_fn=llm_fn,
                    query_prompt=args.coremem_query_prompt,
                    trust_remote_code=args.coremem_trust_remote_code,
                    max_seq_length=args.max_seq_len,
                    batch_size=args.encode_batch_size,
                    encoder=reuse,
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            logger.info(f"[{ds.dataset_name}/{method}] indexing ({len(ds.corpus)} passages)...")
            _t_idx = time.time()
            retriever.index(ds.passages_text, ds.passage_titles)
            index_s = time.time() - _t_idx

            n = len(ds.instances)
            _t_ret = time.time()
            retrieve_threaded = method == "coremem" and args.coremem_decompose and workers > 1

            def _retrieve_one(inst):
                q = inst.get_queries()[0].text
                return retriever.retrieve(q, args.top_k)

            if retrieve_threaded:
                retrieved_idx: List[List[int]] = [[] for _ in range(n)]
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futs = {ex.submit(_retrieve_one, ds.instances[j]): j for j in range(n)}
                    for fut in _pbar(as_completed(futs), n, f"{ds.dataset_name}/{method} retrieve"):
                        retrieved_idx[futs[fut]] = fut.result()
            else:
                retrieved_idx = [
                    _retrieve_one(inst)
                    for inst in _pbar(ds.instances, n, f"{ds.dataset_name}/{method} retrieve")
                ]
            retrieve_s = time.time() - _t_ret
            retrieved_titles = [[ds.passage_titles[i] for i in idxs] for idxs in retrieved_idx]

            gold_answers = [inst.gold_answers for inst in ds.instances]
            gold_titles = [inst.gold_titles for inst in ds.instances]
            ret = score_retrieval(retrieved_titles, gold_titles, ks=(2, 5))

            timing = {
                "n_passages": len(ds.corpus),
                "n_queries": n,
                "index_seconds": round(index_s, 2),
                "retrieve_seconds": round(retrieve_s, 2),
                "retrieve_ms_per_query": round(1000.0 * retrieve_s / max(1, n), 2),
            }

            if args.no_reader:
                ds_results[method] = {"em": None, "f1": None, **ret, "timing": timing}
                logger.info(
                    f"[{ds.dataset_name}/{method}] NO-READER  R@2={ret['recall@2']:.1f} "
                    f"R@5={ret['recall@5']:.1f}"
                )
                continue

            preds = [""] * n
            n_em = 0

            def _read(args_tuple):
                j, idxs = args_tuple
                passages = [ds.passages_text[i] for i in idxs]
                if not passages:
                    return j, ""
                q = ds.instances[j].get_queries()[0].text
                assert reader is not None
                try:
                    pred = _call_with_retry(reader, passages, q)
                except Exception:
                    pred = ""
                return j, pred

            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_read, (j, idxs)) for j, idxs in enumerate(retrieved_idx)]
                bar = _pbar(as_completed(futs), len(futs), f"{ds.dataset_name}/{method} read")
                done = 0
                for fut in bar:
                    j, pred = fut.result()
                    preds[j] = pred
                    done += 1
                    n_em += _em(gold_answers[j], pred)
                    if _tqdm is not None and hasattr(bar, "set_postfix"):
                        bar.set_postfix(EM=f"{100.0 * n_em / done:.1f}", refresh=False)

            qa = score_qa(gold_answers, preds)
            ds_results[method] = {**qa, **ret, "timing": timing}
            logger.info(
                f"[{ds.dataset_name}/{method}] EM={qa['em']:.1f} F1={qa['f1']:.1f} "
                f"R@2={ret['recall@2']:.1f} R@5={ret['recall@5']:.1f}"
            )

        all_results[dataset_key] = {"provenance": ds.provenance(), "results": ds_results}

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(all_results, indent=2))
    logger.info(f"Wrote {out / 'results.json'}")

    print(f"\n{'=' * 72}\nMEMBUKKIT RAG EVALUATION\n{'=' * 72}")
    for d in datasets:
        res = all_results[d]["results"]
        for m in methods:
            r = res.get(m, {})
            em = "—" if r.get("em") is None else f"{r['em']:.1f}"
            f1 = "—" if r.get("f1") is None else f"{r['f1']:.1f}"
            print(
                f"  {d:20s} {m:12s} EM={em:>5s} F1={f1:>5s} "
                f"R@2={r.get('recall@2', 0):.1f} R@5={r.get('recall@5', 0):.1f}"
            )


def _train_encoder_cmd(args):
    """Dispatch to train_encoder script."""
    import subprocess

    script = Path(__file__).parent.parent.parent / "scripts" / "train_encoder.py"
    cmd = [sys.executable, str(script)] + sys.argv[2:]
    subprocess.run(cmd, check=True)


def _train_reranker_cmd(args):
    """Dispatch to train_reranker script."""
    import subprocess

    script = Path(__file__).parent.parent.parent / "scripts" / "train_reranker.py"
    cmd = [sys.executable, str(script)] + sys.argv[2:]
    subprocess.run(cmd, check=True)


def _serve_cmd(args):
    """Launch the multi-tenant memory service."""
    import uvicorn
    from membukkit.config import ModelConfig, RetrievalConfig
    from membukkit.service import MemoryService, ServiceConfig, create_app

    cfg = ServiceConfig(
        models=ModelConfig(model_dir=args.model_dir, encoder=args.encoder, reranker=args.reranker),
        retrieval=RetrievalConfig(retrieval_mode=args.retrieval_mode, union=args.union),
        llm=args.llm,
        region=args.region,
        vector_dtype=args.vector_dtype,
        namespace_prefix=args.namespace_prefix,
        telemetry=not args.no_telemetry,
        environment=args.environment,
        capture_content=args.capture_content,
    )
    app = create_app(MemoryService(cfg))
    logger.info(
        "serving MEMBUKKIT memory on %s:%d (mode=%s, union=%s)",
        args.host,
        args.port,
        args.retrieval_mode,
        args.union,
    )
    uvicorn.run(app, host=args.host, port=args.port)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # These emit one INFO line per LLM call ("HTTP Request: POST ...200 OK" from
    # httpx/openai, "AFC is enabled with max remote calls: 10" from google-genai),
    # which drowns out phase progress. Keep warnings/errors, drop the chatter.
    for _noisy in (
        "httpx",
        "httpcore",
        "openai",
        "urllib3",
        "google_genai",
        "google.genai",
        "google",
    ):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(prog="membukkit", description="MEMBUKKIT CLI")
    sub = parser.add_subparsers(dest="command")

    # eval
    ev = sub.add_parser("eval", help="Run evaluation on a benchmark")
    ev.add_argument("--encoder", default="biencoder_v1")
    ev.add_argument("--reranker", default="reranker_v2/model")
    ev.add_argument("--reader", default="gpt-4o-mini")
    ev.add_argument("--judge", default="gpt-4o")
    ev.add_argument("--official-judge", action="store_true")
    ev.add_argument("--top-k", type=int, default=10)
    ev.add_argument("--reasoning-top-k", type=int, default=30)
    ev.add_argument("--cand", type=int, default=50)
    ev.add_argument("--bucket-mode", default="topic", choices=[None, "topic", "multiaxis"])
    ev.add_argument("--scan-budget", type=float, default=0.3)
    ev.add_argument("--scan-budget-reason", type=float, default=0.45)
    ev.add_argument("--scan-budget-temporal", type=float, default=None)
    ev.add_argument("--bucket-k", type=int, default=24)
    ev.add_argument("--bucket-rerank-cap", type=int, default=50)
    ev.add_argument("--bucket-k-proto", type=int, default=0)
    ev.add_argument("--rerank-select", default="hybrid", choices=["cosine", "xenc", "hybrid"])
    ev.add_argument("--methods", default="coremem_union")
    ev.add_argument(
        "--retrieval-engine",
        default="library",
        choices=["library"],
        help="Retrieval is always the production MemorySystem union (single source "
        "of truth). Kept for command compatibility; the legacy in-CLI dual-bank "
        "path has been removed.",
    )
    ev.add_argument(
        "--storage-backend",
        default="memory",
        choices=["memory", "turbopuffer"],
        help="Backend for the library engine's per-conversation MemorySystem. "
        "memory: in-RAM (default, fast). turbopuffer: persist each haystack to an "
        "ephemeral cloud namespace (needs TURBOPUFFER_API_KEY + region) to smoke-test "
        "the production Turbopuffer path end-to-end; namespaces are deleted after use.",
    )
    ev.add_argument("--region", default=None, help="Turbopuffer region (or TURBOPUFFER_REGION env)")
    ev.add_argument(
        "--namespace-prefix",
        default="evaltmp_",
        help="Prefix for the ephemeral per-conversation Turbopuffer namespaces.",
    )
    ev.add_argument("--vector-dtype", default="f16", choices=["f16", "f32"])
    ev.add_argument(
        "--keep-namespaces",
        action="store_true",
        help="Do NOT delete the ephemeral Turbopuffer namespaces after the run (debug).",
    )
    ev.add_argument(
        "--block-buckets",
        default="none",
        choices=["none", "gold", "random"],
        help="Bucket intervention arm: 'gold' closes the topic buckets holding each "
        "question's gold evidence (causal-faithfulness test); 'random' closes the same "
        "number of non-gold buckets (specificity control). Requires the in-memory backend.",
    )
    ev.add_argument("--block-seed", type=int, default=0, help="Seed for the random-blocking arm.")
    ev.add_argument(
        "--dump-traces",
        action="store_true",
        help="Write retrieval_trace.jsonl (per-query opened buckets with route probs and "
        "raw centroid cosines, per lane, plus the gold->bucket mapping).",
    )
    ev.add_argument(
        "--build-only",
        action="store_true",
        help="Stop after building tasks (and writing traces): no reader/judge LLM calls. "
        "For retrieval-only analyses, e.g. trace-correctness audits.",
    )
    ev.add_argument("--replay-tasks", default=None)
    ev.add_argument(
        "--task-cache-dir",
        default=".membukkit_task_cache",
        help="Directory for the persistent, content-addressed built-task cache. "
        "Reused across runs with different readers/judges/output dirs whenever the "
        "retrieval inputs (dataset, engine, methods, retrieval config, encoder, "
        "reranker, distill model) are unchanged.",
    )
    ev.add_argument(
        "--no-task-cache",
        action="store_true",
        help="Disable reading/writing the persistent built-task cache (always rebuild).",
    )
    ev.add_argument("--reader-protocol", default="coremem", choices=["coremem", "mem0"])
    ev.add_argument(
        "--agg-top-k",
        type=int,
        default=0,
        help="Aggregation routing experiment: top-k for queries with counting/"
        "totaling cues (0 = off). Counting needs every mention, not the best "
        "few, so these queries get deeper evidence.",
    )
    ev.add_argument(
        "--agg-scan-budget",
        type=float,
        default=None,
        help="Scan budget for aggregation-cue queries (e.g. 1.0 = full scan); "
        "default None keeps the standard budget.",
    )
    ev.add_argument(
        "--reader-prompts",
        default="v1",
        choices=["v1", "v2", "v3"],
        help="Reader prompt set. v2 softens the abstention bar and forces "
        "date/enumeration scratchwork (tuned for strong readers); v3 adds "
        "strict direct-answer formatting and an entity-match abstention "
        "check; v1 is the paper-headline set. Does not affect the task cache.",
    )
    ev.add_argument("--judge-protocol", default="coremem", choices=["coremem", "mem0"])
    ev.add_argument(
        "--dataset", default="longmemeval", choices=["longmemeval", "locomo", "beam"]
    )
    ev.add_argument(
        "--beam-scale",
        default="100K",
        choices=["100K", "500K", "1M", "10M"],
        help="BEAM conversation-length split (--dataset beam).",
    )
    ev.add_argument(
        "--beam-pairs-per-session",
        type=int,
        default=6,
        help="Turn pairs per ingested session when chunking BEAM batches "
        "(batches are 30-40K tokens; the distiller needs smaller windows). "
        "Also set MEMBUKKIT_DISTILL_MAX_TURN_CHARS=4000 for BEAM runs: its turns "
        "average ~1,900 chars vs the 600-char LongMemEval-tuned default cap.",
    )
    ev.add_argument("--locomo-path", default="locomo10.json")
    ev.add_argument("--locomo-drop-categories", default="")
    ev.add_argument("--max-instances", type=int, default=0)
    ev.add_argument("--output-dir", default="results/eval")
    ev.add_argument("--distill-cache", default="distill_cache.json")
    ev.add_argument(
        "--deep-top-k",
        type=int,
        default=0,
        help="Deep routing experiment: top-k for needle queries (router-TEMPORAL "
        "or assistant-recall cues); 0 = off. These queries are recall-bound "
        "(evidence ranked out of the standard list), unlike counting queries "
        "which lose precision with extra distractors.",
    )
    ev.add_argument(
        "--deep-scan-budget",
        type=float,
        default=None,
        help="Scan budget for needle queries (e.g. 1.0 = full scan); default "
        "None keeps the standard budget.",
    )
    ev.add_argument(
        "--deep-broad",
        action="store_true",
        help="Also deep-route coverage-bound synthesis queries (summarize/"
        "overview/how-has-X-evolved surface cues). Opt-in so existing "
        "scoped-deep task caches keep their keys; used for BEAM, whose "
        "summarization rubrics grade nugget coverage.",
    )
    ev.add_argument(
        "--beam-ordering-mode",
        choices=["official", "amb"],
        default="official",
        help="Event-ordering answer style: 'official' truncates to the "
        "question's requested item count (official BEAM scorer); 'amb' "
        "enumerates ALL topics chronologically (Hindsight's AMB judge "
        "grades against the full expected-order list).",
    )
    ev.add_argument(
        "--only-ids",
        default="",
        help="Comma-separated question_ids to evaluate (diagnostic runs on a "
        "specific subset, e.g. re-reading only previously missed questions "
        "after a targeted re-distill). Gets its own task-cache entry.",
    )
    ev.add_argument("--distill-model", default="gpt-4o-mini")
    ev.add_argument("--distill-workers", type=int, default=16)
    ev.add_argument(
        "--llm-workers",
        type=int,
        default=0,
        help="Concurrency for answer+judge LLM calls. 0 = auto "
        "(2 for rate-limited Google/Gemini backends, 8 otherwise).",
    )
    ev.set_defaults(func=_eval_cmd)

    # rag-eval
    _PKG = Path(__file__).resolve().parent
    re_ = sub.add_parser("rag-eval", help="Run multi-hop RAG evaluation (EM/F1/Recall)")
    re_.add_argument("--datasets", default="musique,2wiki,hotpot")
    re_.add_argument("--methods", default="dense,coremem")
    re_.add_argument("--embedder", default="sentence-transformers/all-mpnet-base-v2")
    re_.add_argument("--query-prompt", default="")
    re_.add_argument("--trust-remote-code", action="store_true")
    re_.add_argument("--max-seq-len", type=int, default=0)
    re_.add_argument("--encode-batch-size", type=int, default=64)
    re_.add_argument("--top-k", type=int, default=5)
    re_.add_argument("--workers", type=int, default=RAG_LLM_WORKERS)
    re_.add_argument("--data-dir", default=None)
    re_.add_argument(
        "--coremem-encoder", default=str(_PKG.parent.parent.parent / "models" / "biencoder_v1")
    )
    re_.add_argument(
        "--coremem-reranker",
        default=str(_PKG.parent.parent.parent / "models" / "reranker_v2" / "model"),
    )
    re_.add_argument("--coremem-budget", type=float, default=0.3)
    re_.add_argument("--coremem-bucket-k", type=int, default=24)
    re_.add_argument("--coremem-rerank-cap", type=int, default=100)
    re_.add_argument("--coremem-fusion", choices=["rrf", "rerank", "cosine"], default="cosine")
    re_.add_argument("--coremem-hops", type=int, default=1)
    re_.add_argument("--coremem-expand-m", type=int, default=3)
    re_.add_argument(
        "--coremem-expand-mode", choices=["entity", "passage", "both"], default="entity"
    )
    re_.add_argument("--coremem-axes", choices=["topic", "multi"], default="topic")
    re_.add_argument("--coremem-temporal", action="store_true")
    re_.add_argument("--coremem-entity-cap", type=int, default=50)
    re_.add_argument("--coremem-entity-rank", action="store_true")
    re_.add_argument("--coremem-entity-min", type=int, default=1)
    re_.add_argument("--coremem-decompose", action="store_true")
    re_.add_argument("--coremem-max-subq", type=int, default=4)
    re_.add_argument(
        "--coremem-decompose-fuse", choices=["interleave", "maxpool"], default="interleave"
    )
    re_.add_argument("--coremem-decompose-no-iter", action="store_true")
    re_.add_argument(
        "--coremem-decompose-retrieval",
        choices=["full_cosine", "bucket"],
        default="full_cosine",
    )
    re_.add_argument("--coremem-query-prompt", default="")
    re_.add_argument("--coremem-trust-remote-code", action="store_true")
    re_.add_argument("--no-reader", action="store_true")
    re_.add_argument("--reader-verify", action="store_true")
    re_.add_argument("--output-dir", default="results/rag_eval")
    re_.add_argument("--smoke", type=int, default=0)
    re_.set_defaults(func=_rag_eval_cmd)

    # train-encoder
    te = sub.add_parser("train-encoder", help="Fine-tune bi-encoder")
    te.set_defaults(func=_train_encoder_cmd)

    # train-reranker
    tr = sub.add_parser("train-reranker", help="Fine-tune cross-encoder reranker")
    tr.set_defaults(func=_train_reranker_cmd)

    # serve — multi-tenant memory service (Turbopuffer-backed)
    sv = sub.add_parser("serve", help="Run the multi-tenant memory service (FastAPI)")
    sv.add_argument("--host", default="0.0.0.0")
    sv.add_argument("--port", type=int, default=8080)
    sv.add_argument("--model-dir", default=None, help="Path to encoder/reranker weights")
    sv.add_argument("--encoder", default="biencoder_v1")
    sv.add_argument("--reranker", default="reranker_v2/model")
    sv.add_argument("--llm", default="openai:gpt-4o-mini")
    sv.add_argument("--region", default=None, help="Turbopuffer region (or TURBOPUFFER_REGION)")
    sv.add_argument(
        "--namespace-prefix",
        default="mem_",
        help="Prefix for owner->namespace mapping (default 'mem_'); "
        "pass '' to use the owner id as the namespace verbatim",
    )
    sv.add_argument("--retrieval-mode", default="gated", choices=["gated", "open"])
    sv.add_argument(
        "--union",
        dest="union",
        action="store_true",
        default=True,
        help="Dual verbatim+atomic retrieval (SOTA coremem_union; default on)",
    )
    sv.add_argument(
        "--no-union",
        dest="union",
        action="store_false",
        help="Single-index atomic-only retrieval (disables the union lanes)",
    )
    sv.add_argument("--vector-dtype", default="f16", choices=["f16", "f32"])
    sv.add_argument("--environment", default=None, help="Deployment env tag for telemetry")
    sv.add_argument(
        "--capture-content",
        action="store_true",
        help="Log raw fact/query/LLM text to telemetry (PII; debug only)",
    )
    sv.add_argument(
        "--no-telemetry", action="store_true", help="Disable Logfire/OTel instrumentation"
    )
    sv.set_defaults(func=_serve_cmd)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
