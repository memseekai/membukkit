"""High-level evaluation harness.

Runs a MemorySystem over a dataset of instances and scores answers with a
chosen judge protocol. This is the convenience wrapper behind:

    from membukkit.eval import evaluate
    report = evaluate(mem, load_longmemeval(variant="longmemeval_s"), judge="official")
"""

from __future__ import annotations

import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

CORRECT_THRESHOLD = 0.7


def _make_judge(judge: str, llm_fn: Callable[[str], str]):
    """Resolve a judge name to a scoring callable."""
    from membukkit.eval.judges import make_judge_fn, make_official_judge
    from membukkit.reading.readers import make_mem0_judge

    if judge == "official":
        fn = make_official_judge(llm_fn)
        return ("official", fn)
    if judge == "mem0":
        fn = make_mem0_judge(llm_fn)
        return ("mem0", fn)
    return ("generic", make_judge_fn(llm_fn))


def evaluate(
    mem,
    dataset,
    judge: str = "official",
    judge_llm: str = "openai:gpt-4o",
    max_instances: Optional[int] = None,
    workers: int = 8,
) -> Dict[str, Any]:
    """Evaluate a MemorySystem on a dataset.

    Args:
        mem: a MemorySystem instance.
        dataset: a LongMemEvalDataset (or anything with `.instances`).
        judge: "official" (LongMemEval binary), "mem0" (LoCoMo J-score), or "generic".
        judge_llm: LLM spec for the judge.
        max_instances: cap the number of instances evaluated.
        workers: parallel answer+judge workers.

    Returns:
        A report dict with overall accuracy and a per-ability breakdown.
    """
    from membukkit.llm.backends import parse_llm_spec

    instances = list(dataset.instances)
    if max_instances:
        instances = instances[:max_instances]

    judge_kind, judge_fn = _make_judge(judge, parse_llm_spec(judge_llm))

    def _run(inst):
        queries = inst.get_queries()
        if not queries:
            return None
        q = queries[0]
        gt = q.ground_truth or ""
        if not gt:
            return None

        item = getattr(inst, "_item", {})
        # Per-instance haystack: reset then ingest then answer.
        mem.reset()
        sessions = item.get("haystack_sessions", [])
        dates = item.get("haystack_dates", [])
        if sessions:
            mem.ingest(sessions=sessions, dates=dates)
        res = mem.answer(q.text, question_date=inst.question_date_raw)

        if judge_kind == "official":
            score = judge_fn(
                q.text,
                gt,
                res.answer,
                question_type=item.get("question_type", ""),
                abstention=str(inst.question_id).endswith("_abs"),
            )
        elif judge_kind == "mem0":
            score = judge_fn(q.text, gt, res.answer, category=item.get("locomo_category"))
        else:
            score = judge_fn(q.text, res.answer, gt)
        return {"ability": inst.ability, "judge": float(score), "answer": res.answer}

    results: List[Dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_run, inst) for inst in instances]
        for k, fut in enumerate(as_completed(futs)):
            r = fut.result()
            if r is not None:
                results.append(r)
            if (k + 1) % 25 == 0:
                logger.info(f"  evaluated {k + 1}/{len(instances)}")

    scores = [r["judge"] for r in results]
    by_ability = defaultdict(list)
    for r in results:
        by_ability[r["ability"]].append(r["judge"])

    return {
        "judge": judge,
        "n": len(results),
        "accuracy": float(np.mean([s >= CORRECT_THRESHOLD for s in scores])) if scores else 0.0,
        "judge_mean": float(np.mean(scores)) if scores else 0.0,
        "by_ability": {
            a: float(np.mean([s >= CORRECT_THRESHOLD for s in v])) for a, v in by_ability.items()
        },
    }
