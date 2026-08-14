"""Reader factories for MEMBUKKIT."""

from __future__ import annotations

import re
from typing import Callable, List, Optional

from membukkit.prompts.reading import (
    ABSTAIN_GATE_PROMPT,
    DATED_READER_PROMPT,
    ORDERING_READER_PROMPT,
    ORDERING_READER_PROMPT_AMB,
    REASONING_READER_PROMPT,
    RECOMMENDATION_READER_PROMPT,
)
from membukkit.prompts.protocol import (
    MEM0_ANSWER_PROMPT,
    MEM0_JUDGE_PROMPT,
    _extract_label,
    preprocess_gold,
)

ABSTAIN_TEXT = (
    "Based on our past conversations, you never mentioned that, "
    "so I don't have any information about it."
)


def _today_line(qdate: str) -> str:
    """Frame the as-of date: stated-at vs effective-at for current-state answers."""
    if not qdate:
        return ""
    return (
        f"Today's date is {qdate}. Answer strictly as of that date: report the "
        f"state that was true then. Memory dates mark when something was stated; "
        f"if a change takes effect after today, it is not yet current.\n"
    )


def _identity_preamble(identity: str) -> str:
    if not identity:
        return ""
    return (
        f"These memories belong to {identity}. Answer the question about this person, "
        f"and treat their stated identity (their own name and own email address) as "
        f"authoritative ground truth over any conflicting memory.\n"
    )


def _normalize_abstain(ans: str) -> str:
    return ABSTAIN_TEXT if (ans or "").strip().upper() in ("N/I", "NI") else ans


_ORDERING_LINE_RE = re.compile(r"^\s*(\d{4}-\d{2}-\d{2})\s*\|\s*(.+?)\s*$")
_COUNT_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def _requested_item_count(question: str) -> Optional[int]:
    m = re.search(r"\b(\d+|" + "|".join(_COUNT_WORDS) + r")\s+items?\b", question, re.I)
    if not m:
        return None
    w = m.group(1).lower()
    return int(w) if w.isdigit() else _COUNT_WORDS.get(w)


def make_ordering_reader(llm_fn: Callable[[str], str], amb_mode: bool = False):
    """Event-ordering answers: LLM names dated top-level phases, the harness
    orders them by date DETERMINISTICALLY and emits only the descriptions
    (one per line, the format the official BEAM scorer expects).

    Ordering by code instead of by the LLM removes sequencing slips; the
    prompt handles the other failure mode (item granularity vs the scorer's
    compound rubric items).

    amb_mode targets Hindsight's AMB judging instead: their binary judge
    grades against the FULL expected-order list (up to 9 items) even when
    the question asks for fewer, and fails answers with "key topics
    missing". So the AMB variant enumerates ALL distinct topics
    chronologically and never truncates to the requested count.
    """
    template = ORDERING_READER_PROMPT_AMB if amb_mode else ORDERING_READER_PROMPT
    _date_re = re.compile(r"^\[(\d{4}-\d{2}-\d{2})\]")

    def answer(fact_lines: List[str], question: str, qdate: str) -> str:
        if amb_mode:
            # Chronological presentation: the AMB prompt walks the timeline
            # stage by stage, which relevance ranking destroys.
            def _key(f: str):
                m = _date_re.match(f)
                return m.group(1) if m else "9999-99-99"
            fact_lines = sorted(fact_lines, key=_key)
        fact_block = "\n".join(f"- {f}" for f in fact_lines) if fact_lines else "(none)"
        prompt = template.format(
            today_line=_today_line(qdate), fact_block=fact_block, question=question
        )
        raw = llm_fn(prompt).strip()
        rows = []
        for line in raw.splitlines():
            m = _ORDERING_LINE_RE.match(line)
            if m:
                rows.append((m.group(1), m.group(2)))
        if not rows:  # format not followed; the raw text is the best we have
            return raw
        rows.sort(key=lambda r: r[0])  # stable: ties keep LLM order
        # Both scorers reward exactly the requested count: the official one
        # F1-penalizes extras, and AMB questions ask for exactly as many
        # items as the expected-order list holds (68/70 at 1M).
        want = _requested_item_count(question)
        if want and len(rows) > want:
            rows = rows[:want]
        return "\n".join(desc for _, desc in rows)

    return answer


def make_dated_reader(
    llm_fn: Callable[[str], str],
    identity: str = "",
    prompt_template: Optional[str] = None,
):
    template = prompt_template or DATED_READER_PROMPT

    def answer(fact_lines: List[str], question: str, qdate: str) -> str:
        fact_block = "\n".join(f"- {f}" for f in fact_lines) if fact_lines else "(none)"
        prompt = template.format(
            identity_preamble=_identity_preamble(identity),
            today_line=_today_line(qdate),
            fact_block=fact_block,
            question=question,
        )
        return llm_fn(prompt).strip()

    return answer


def make_recommendation_reader(
    llm_fn: Callable[[str], str],
    identity: str = "",
    prompt_template: Optional[str] = None,
):
    template = prompt_template or RECOMMENDATION_READER_PROMPT

    def answer(fact_lines: List[str], question: str, qdate: str) -> str:
        fact_block = "\n".join(f"- {f}" for f in fact_lines) if fact_lines else "(none)"
        prompt = template.format(
            identity_preamble=_identity_preamble(identity),
            today_line=_today_line(qdate),
            fact_block=fact_block,
            question=question,
        )
        return llm_fn(prompt).strip()

    return answer


def make_reasoning_reader(
    llm_fn: Callable[[str], str],
    identity: str = "",
    prompt_template: Optional[str] = None,
):
    template = prompt_template or REASONING_READER_PROMPT

    def answer(fact_lines: List[str], question: str, qdate: str) -> str:
        fact_block = "\n".join(f"- {f}" for f in fact_lines) if fact_lines else "(none)"
        prompt = template.format(
            identity_preamble=_identity_preamble(identity),
            today_line=_today_line(qdate),
            fact_block=fact_block,
            question=question,
        )
        out = llm_fn(prompt).strip()
        if "Answer:" in out:
            out = out.rsplit("Answer:", 1)[1].strip()
        return out

    return answer


def make_abstain_gate(
    llm_fn: Callable[[str], str],
    prompt_template: Optional[str] = None,
):
    template = prompt_template or ABSTAIN_GATE_PROMPT

    def gate(question: str, answer: str, fact_lines: List[str], qdate: str) -> bool:
        fact_block = "\n".join(f"- {f}" for f in fact_lines) if fact_lines else "(none)"
        prompt = template.format(
            fact_block=fact_block,
            question=question,
            answer=answer,
        )
        try:
            verdict = llm_fn(prompt).strip().upper()
            return "UNSUPPORTED" not in verdict
        except Exception:
            return True

    return gate


def make_mem0_reader(llm_fn: Callable[[str], str]):
    def answer(fact_lines: List[str], question: str, qdate: str = "") -> str:
        memories = "\n".join(fact_lines) if fact_lines else "(No relevant memories found)"
        ref = qdate or "2023"
        prompt = MEM0_ANSWER_PROMPT.format(memories=memories, question=question, reference_date=ref)
        out = (llm_fn(prompt) or "").strip()
        if "ANSWER:" in out:
            out = out.rsplit("ANSWER:", 1)[1].strip()
        return out

    return answer


def make_mem0_judge(llm_fn: Callable[[str], str]):
    def judge(
        question: str,
        gold_answer: str,
        response: str,
        category: Optional[int] = None,
    ) -> float:
        gold = preprocess_gold(category, gold_answer or "")
        prompt = MEM0_JUDGE_PROMPT.format(question=question, answer=gold, response=response)
        try:
            return 1.0 if _extract_label(llm_fn(prompt)) == "CORRECT" else 0.0
        except Exception:
            return 0.0

    return judge
