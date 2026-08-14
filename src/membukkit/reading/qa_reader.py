"""Short-answer QA reader for multi-hop RAG.

Produces short factual answers (name, entity, date, phrase) scored by EM/F1,
distinct from the chat-memory dated/reasoning readers which produce longer
prose answers scored by LLM judge.
"""

from __future__ import annotations

import random
import time
from typing import Callable, List


_READER_PROMPT = """You are a careful reading-comprehension assistant for multi-hop question answering.
Use ONLY the numbered passages below. Reason briefly across them, then give the SHORT factual answer
(a name, entity, date, or phrase) — no explanation in the final answer, no full sentence.

Passages:
{passages}

Question: {question}

Respond in exactly this format:
Thought: <one or two sentences connecting the passages>
Answer: <short answer>"""

_VERIFY_PROMPT = """You are checking a candidate answer for a multi-hop question.
Re-read the numbered passages. If the candidate is correct AND fully supported, repeat it
verbatim. Otherwise give the correct SHORT factual answer (a name, entity, date, or phrase).

Passages:
{passages}

Question: {question}
Candidate answer: {cand}

Respond in exactly this format:
Check: <one sentence: supported or what's wrong>
Answer: <short answer>"""

READER_RETRIES = 6


def _parse_answer(out: str) -> str:
    out = (out or "").strip()
    if "Answer:" in out:
        out = out.rsplit("Answer:", 1)[1].strip()
    return out.splitlines()[0].strip() if out else ""


def _call_with_retry(fn: Callable, *fn_args, retries: int = READER_RETRIES):
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return fn(*fn_args)
        except Exception as e:
            last = e
            time.sleep(min(30.0, (2**attempt) * 0.5) + random.uniform(0, 0.5))
    if last is not None:
        raise last
    raise RuntimeError("call failed with no attempts")


def make_qa_reader(llm_fn: Callable[[str], str], verify: bool = False):
    """Build a QA reader function: (passages, question) -> short answer string."""

    def answer(passages: List[str], question: str) -> str:
        block = "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages)) or "(none)"
        cand = _parse_answer(llm_fn(_READER_PROMPT.format(passages=block, question=question)))
        if not verify or not cand:
            return cand
        verified = _parse_answer(
            llm_fn(_VERIFY_PROMPT.format(passages=block, question=question, cand=cand))
        )
        return verified or cand

    return answer
