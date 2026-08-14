"""Jina-style DeepSearch / DeepResearch over MEMBUKKIT memory.

This module keeps external search out of v1. The "search" and "read" actions
both resolve to `MemorySystem.search`, because MEMBUKKIT stores already-atomic
dated facts. Pydantic AI is imported lazily so the core package remains usable
without the optional `membukkit[agent]` extra.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

DEFAULT_MODEL = "openai-responses:gpt-5.4"


@dataclass
class MemoryResearchDeps:
    """Runtime dependencies and budgets for memory deep research."""

    memory: Any
    identity: str = ""
    question_date: Optional[str] = None
    max_steps: int = 12
    max_gap_questions: int = 8
    max_bad_attempts: int = 2
    token_budget: int = 120_000
    evidence_top_k: int = 12
    min_evidence: int = 2


@dataclass
class EvidenceNote:
    """One evidence item promoted into the research knowledge ledger."""

    ref: str
    fact: str
    text: str
    query: str
    timestamp: Optional[str] = None
    source_id: str = ""


@dataclass
class DiaryEntry:
    """Compact record of one DeepSearch transition."""

    step: int
    action: str
    question: str
    summary: str
    queries: List[str] = field(default_factory=list)
    refs: List[str] = field(default_factory=list)
    accepted: bool = True


@dataclass
class DeepSearchState:
    """Mutable state for the DeepSearch loop."""

    original_question: str
    identity: str = ""
    question_date: Optional[str] = None
    gap_queue: List[str] = field(default_factory=list)
    knowledge_ledger: List[EvidenceNote] = field(default_factory=list)
    action_diary: List[DiaryEntry] = field(default_factory=list)
    attempted_queries: List[str] = field(default_factory=list)
    bad_attempts: int = 0
    total_steps: int = 0
    token_usage: int = 0
    token_budget: int = 120_000
    max_steps: int = 12
    max_gap_questions: int = 8
    max_bad_attempts: int = 2
    evidence_top_k: int = 12
    min_evidence: int = 2
    last_answer_failed: bool = False
    final_answer: str = ""
    final_citations: List[str] = field(default_factory=list)
    unresolved_gaps: List[str] = field(default_factory=list)
    trace_stats: Dict[str, Any] = field(default_factory=dict)

    def next_question(self) -> str:
        if not self.gap_queue:
            self.gap_queue.append(self.original_question)
        return self.gap_queue.pop(0)


@dataclass
class DeepSearchAction:
    """Structured action selected by the Pydantic AI controller."""

    action: Literal["reflect", "search", "answer"]
    rationale: str = ""
    gap_questions: List[str] = field(default_factory=list)
    search_queries: List[str] = field(default_factory=list)
    answer: str = ""
    citations: List[str] = field(default_factory=list)


@dataclass
class AnswerEvaluation:
    """Structured verdict from the answer evaluator agent."""

    passes: bool
    reason: str = ""
    missing_questions: List[str] = field(default_factory=list)
    follow_up_queries: List[str] = field(default_factory=list)


@dataclass
class DeepSearchResult:
    """Final answer from one DeepSearch run."""

    question: str
    answer: str
    citations: List[str] = field(default_factory=list)
    evidence: List[EvidenceNote] = field(default_factory=list)
    unresolved_gaps: List[str] = field(default_factory=list)
    diary: List[DiaryEntry] = field(default_factory=list)
    trace_stats: Dict[str, Any] = field(default_factory=dict)
    passed: bool = False


@dataclass
class ResearchSectionPlan:
    """One planned report section."""

    title: str
    question: str


@dataclass
class ResearchPlan:
    """Structured DeepResearch plan."""

    sections: List[ResearchSectionPlan] = field(default_factory=list)


@dataclass
class ResearchSection:
    """Completed report section."""

    title: str
    question: str
    answer: str
    citations: List[str] = field(default_factory=list)
    unresolved_gaps: List[str] = field(default_factory=list)


@dataclass
class ResearchReport:
    """DeepResearch report assembled from section-level DeepSearch runs."""

    question: str
    report: str
    sections: List[ResearchSection] = field(default_factory=list)
    evidence: List[EvidenceNote] = field(default_factory=list)
    unresolved_gaps: List[str] = field(default_factory=list)
    trace_stats: Dict[str, Any] = field(default_factory=dict)
    diary: List[DiaryEntry] = field(default_factory=list)


def _require_pydantic_ai():
    try:
        from pydantic_ai import Agent, RunContext
        from pydantic_ai.capabilities import AbstractCapability
        from pydantic_ai.toolsets import AgentToolset, FunctionToolset
    except ImportError as exc:
        raise ImportError(
            "Pydantic AI is required for memory deep research. "
            "Install it with `pip install -e '.[agent]'`."
        ) from exc
    return Agent, RunContext, AbstractCapability, AgentToolset, FunctionToolset


def _normalize_query(query: str) -> str:
    query = re.sub(r"\s+", " ", (query or "").strip().lower())
    return re.sub(r"[^\w\s@.+:-]", "", query)


def _token_set(query: str) -> set:
    return {tok for tok in _normalize_query(query).split() if tok}


def _too_similar(a: str, b: str, threshold: float = 0.9) -> bool:
    ta, tb = _token_set(a), _token_set(b)
    if not ta or not tb:
        return _normalize_query(a) == _normalize_query(b)
    return len(ta & tb) / len(ta | tb) >= threshold


def dedupe_queries(queries: List[str], existing: Optional[List[str]] = None) -> List[str]:
    """Drop repeated or near-identical queries while preserving order."""

    existing = existing or []
    out: List[str] = []
    for raw in queries:
        q = re.sub(r"\s+", " ", (raw or "").strip())
        if not q:
            continue
        seen = existing + out
        if any(_too_similar(q, other) for other in seen):
            continue
        out.append(q)
    return out


def rewrite_search_queries(
    queries: List[str],
    *,
    current_question: str,
    original_question: str,
    attempted_queries: Optional[List[str]] = None,
    max_queries: int = 3,
) -> List[str]:
    """Small deterministic rewrite/expansion before memory search."""

    candidates = list(queries or [])
    if current_question:
        candidates.insert(0, current_question)
    if original_question and not _too_similar(current_question, original_question):
        candidates.append(f"{original_question} {current_question}")
    return dedupe_queries(candidates, attempted_queries)[:max_queries]


def validate_citations(
    citations: List[str], evidence: List[EvidenceNote]
) -> Tuple[List[str], List[str]]:
    """Return (valid, invalid) citation refs against the current ledger."""

    known = {e.ref for e in evidence}
    valid, invalid = [], []
    for ref in citations:
        if ref in known and ref not in valid:
            valid.append(ref)
        elif ref not in known and ref not in invalid:
            invalid.append(ref)
    return valid, invalid


def allowed_actions(state: DeepSearchState) -> List[str]:
    """Gate actions based on evidence and prior failures."""

    actions = ["reflect", "search"]
    if (
        len(state.knowledge_ledger) >= state.min_evidence
        and not state.last_answer_failed
        and state.bad_attempts <= state.max_bad_attempts
    ):
        actions.append("answer")
    return actions


def rotate_gap_questions(state: DeepSearchState, gap_questions: List[str]) -> List[str]:
    """Push new gaps to the front and the original question behind them."""

    new_gaps = dedupe_queries(
        gap_questions,
        existing=state.gap_queue + [state.original_question],
    )
    room = max(0, state.max_gap_questions - len(state.gap_queue))
    new_gaps = new_gaps[:room]
    if new_gaps:
        state.gap_queue = new_gaps + state.gap_queue
    state.gap_queue.append(state.original_question)
    return new_gaps


def _trace_summary(result: Any) -> Dict[str, Any]:
    trace = getattr(result, "trace", None)
    return {
        "backend": getattr(trace, "backend", ""),
        "n_facts": getattr(trace, "n_facts", 0),
        "n_scanned": getattr(trace, "n_scanned", 0),
        "scan_fraction": getattr(trace, "scan_fraction", 0.0),
    }


def _merge_trace_stats(state: DeepSearchState, search_result: Any) -> None:
    trace = _trace_summary(search_result)
    stats = state.trace_stats
    stats["searches"] = stats.get("searches", 0) + 1
    stats["n_scanned_total"] = stats.get("n_scanned_total", 0) + trace.get("n_scanned", 0)
    stats["last_backend"] = trace.get("backend", "")
    stats["last_scan_fraction"] = trace.get("scan_fraction", 0.0)
    max_scan = stats.get("max_scan_fraction", 0.0)
    stats["max_scan_fraction"] = max(max_scan, trace.get("scan_fraction", 0.0))


def add_search_results(
    state: DeepSearchState, query: str, search_result: Any
) -> List[EvidenceNote]:
    """Promote new memory hits into the ledger."""

    known = {item.ref for item in state.knowledge_ledger}
    added: List[EvidenceNote] = []
    for hit in getattr(search_result, "hits", []):
        if hit.ref in known:
            continue
        note = EvidenceNote(
            ref=hit.ref,
            fact=hit.fact,
            text=hit.text,
            query=query,
            timestamp=hit.timestamp,
            source_id=hit.source_id,
        )
        state.knowledge_ledger.append(note)
        known.add(note.ref)
        added.append(note)
    _merge_trace_stats(state, search_result)
    return added


def _usage_tokens(result: Any) -> int:
    usage = getattr(result, "usage", None)
    if callable(usage):
        try:
            usage = usage()
        except TypeError:
            pass
    for name in ("total_tokens", "total", "request_tokens"):
        value = getattr(usage, name, None)
        if isinstance(value, int):
            return value
    return 0


def _agent_output(result: Any) -> Any:
    return getattr(result, "output", result)


def _knowledge_context(evidence: List[EvidenceNote], limit: int = 40) -> str:
    if not evidence:
        return "(no evidence yet)"
    rows = []
    for item in evidence[-limit:]:
        rows.append(f"- {item.ref} {item.fact}")
    return "\n".join(rows)


def _diary_context(diary: List[DiaryEntry], limit: int = 12) -> str:
    if not diary:
        return "(no prior actions)"
    rows = []
    for item in diary[-limit:]:
        status = "ok" if item.accepted else "rejected"
        refs = f" refs={','.join(item.refs)}" if item.refs else ""
        queries = f" queries={'; '.join(item.queries)}" if item.queries else ""
        rows.append(f"- step {item.step} {item.action} [{status}]: {item.summary}{queries}{refs}")
    return "\n".join(rows)


def _controller_prompt(state: DeepSearchState, current_question: str, actions: List[str]) -> str:
    return (
        "You are controlling a memory-only DeepSearch loop.\n"
        "Choose exactly one next action from the allowed actions. Do not invent citations.\n\n"
        f"<original_question>{state.original_question}</original_question>\n"
        f"<current_question>{current_question}</current_question>\n"
        f"<identity>{state.identity or '(unknown)'}</identity>\n"
        f"<question_date>{state.question_date or '(not provided)'}</question_date>\n"
        f"<allowed_actions>{', '.join(actions)}</allowed_actions>\n"
        f"<gap_queue>{state.gap_queue}</gap_queue>\n"
        f"<bad_attempts>{state.bad_attempts}</bad_attempts>\n"
        f"<attempted_queries>{state.attempted_queries}</attempted_queries>\n\n"
        f"<knowledge>\n{_knowledge_context(state.knowledge_ledger)}\n</knowledge>\n\n"
        f"<diary>\n{_diary_context(state.action_diary)}\n</diary>\n\n"
        "Action policy:\n"
        "- reflect: add focused gap_questions when evidence is missing, contradictory, "
        "or a facet of the question is still uncovered (identity questions -> role, "
        "relationship, recent activity).\n"
        "- search: propose 1-3 concrete memory search queries, including alternate "
        "wording (names, emails, event titles).\n"
        "- answer: only when the evidence is enough. Write directly in the subject's "
        "assistant voice; state concrete facts (named people, dates, events) rather "
        "than describing the memory; never write 'memory says' or 'related memories'. "
        "Cite refs from <knowledge>.\n"
    )


def _evaluate_prompt(
    question: str, answer: str, citations: List[str], state: DeepSearchState
) -> str:
    return (
        "Evaluate whether this memory-grounded answer is concise, well-synthesized, and supported.\n"
        "Pass if it answers the question, its claims are backed by cited refs, and it "
        "names the actual people, places, and events (grouped/organized when the question "
        "asks to recommend or list). A concise synthesis that omits some individual "
        "evidence items is fine. Fail only for unsupported or fabricated citations, vague "
        "gestures ('several meetings') where specifics exist, retrieval narration ('memory "
        "says'), raw calendar dumps, or a missing facet the question clearly asks for.\n\n"
        f"<question>{question}</question>\n"
        f"<answer>{answer}</answer>\n"
        f"<citations>{citations}</citations>\n\n"
        f"<knowledge>\n{_knowledge_context(state.knowledge_ledger)}\n</knowledge>\n"
    )


def _coerce_action(
    action: DeepSearchAction, allowed: List[str], current_question: str
) -> DeepSearchAction:
    if action.action in allowed:
        return action
    if "search" in allowed:
        return DeepSearchAction(
            action="search",
            rationale=f"Model chose gated action {action.action!r}; falling back to search.",
            search_queries=[current_question],
        )
    return DeepSearchAction(
        action="reflect",
        rationale=f"Model chose gated action {action.action!r}; falling back to reflect.",
        gap_questions=[current_question],
    )


def _fallback_answer(question: str, state: DeepSearchState) -> str:
    if not state.knowledge_ledger:
        return (
            "I could not find enough relevant memory evidence to answer this. "
            "No memory citations are available."
        )
    lines = [
        "I could not synthesize a confident answer, but the most relevant memory is:",
        "",
    ]
    for item in state.knowledge_ledger[:6]:
        snippet = re.sub(r"\s+", " ", item.fact).strip()
        if len(snippet) > 140:
            snippet = snippet[:137].rstrip() + "..."
        lines.append(f"- {snippet} ({item.ref})")
    if state.unresolved_gaps:
        lines.append("")
        lines.append("Unresolved gaps: " + "; ".join(state.unresolved_gaps[:5]))
    return "\n".join(lines)


def _make_controller(model: str):
    Agent, _, _, _, _ = _require_pydantic_ai()
    return Agent(
        model,
        output_type=DeepSearchAction,
        instructions=(
            "You are a disciplined DeepSearch controller operating over one person's "
            "dated personal memory (calendar events and conversation-derived facts). "
            "Return exactly one structured action.\n"
            "- Prefer `search` when evidence is thin, `reflect` when the question has "
            "facets you have not yet covered, and `answer` only when the cited memory "
            "evidence is genuinely sufficient.\n"
            "- For identity / 'who is' questions, cover role, relationship to the "
            "subject, and recent interactions before answering; a name alone is not an "
            "answer.\n"
            "When you answer, write in a direct, natural voice as the subject's "
            "assistant, and be concise. Name the specific people, places, and events "
            "involved, but SYNTHESIZE across the evidence rather than restating each "
            "memory item: when the question asks to recommend, list, compare, or "
            "categorize, aggregate the relevant facts and organize the answer (for "
            "example group restaurants by lunch vs dinner) instead of emitting one line "
            "per calendar entry. Do not quote raw calendar text. A short grouped list or "
            "a few tight sentences is ideal. Never narrate the retrieval ('memory says', "
            "'supporting memory', 'related memories associate') and never summarize "
            "vaguely ('several meetings'). Cite refs inline from <knowledge>."
        ),
    )


def _make_evaluator(model: str):
    Agent, _, _, _, _ = _require_pydantic_ai()
    return Agent(
        model,
        output_type=AnswerEvaluation,
        instructions=(
            "You are an evaluator for memory-grounded answers about one person's life. "
            "Reward concise, well-synthesized answers: a short grouped list or a few "
            "tight sentences that names the actual people, places, and events and, for "
            "recommend/list/compare questions, organizes them (e.g. lunch vs dinner) is "
            "a GOOD answer even though it does not restate every cited fact. Do not "
            "require the answer to enumerate all evidence. Fail an answer only if it is "
            "unsupported, cites refs absent from <knowledge>, or genuinely fails the "
            "question by: (a) staying vague where it should name specifics ('several "
            "meetings', 'various places') even though the cited refs hold them; (b) "
            "narrating the retrieval ('memory says', 'related memories'); (c) dumping raw "
            "calendar text instead of synthesizing; or (d) omitting a facet the question "
            "clearly asks for. Return missing_questions and follow_up_queries when you fail."
        ),
    )


def make_memory_search_capability():
    """Create a Pydantic AI capability exposing MEMBUKKIT memory search."""

    _, RunContext, AbstractCapability, AgentToolset, FunctionToolset = _require_pydantic_ai()
    toolset = FunctionToolset()

    @toolset.tool
    def search_memory(
        ctx: RunContext[MemoryResearchDeps],
        query: str,
        top_k: Optional[int] = None,
        reason: str = "",
    ) -> Dict[str, Any]:
        """Search dated MEMBUKKIT memory facts and return cited evidence."""

        del reason
        k = top_k or ctx.deps.evidence_top_k
        result = ctx.deps.memory.search(query, top_k=k)
        return memory_search_result_to_dict(result)

    @dataclass
    class MemorySearchCapability(AbstractCapability[MemoryResearchDeps]):
        id: str = "membukkit_memory_search"
        description: str = "Searches MEMBUKKIT memory and returns dated fact citations."

        def get_toolset(self) -> AgentToolset[MemoryResearchDeps] | None:
            return toolset

    return MemorySearchCapability()


def memory_search_result_to_dict(result: Any) -> Dict[str, Any]:
    return {
        "query": getattr(result, "query", ""),
        "hits": [asdict(hit) for hit in getattr(result, "hits", [])],
        "trace": _trace_summary(result),
    }


def run_memory_deep_search(
    question: str,
    deps: MemoryResearchDeps,
    *,
    model: str = DEFAULT_MODEL,
) -> DeepSearchResult:
    """Run the memory-only DeepSearch loop for one question."""

    controller = _make_controller(model)
    evaluator = _make_evaluator(model)
    state = DeepSearchState(
        original_question=question,
        identity=deps.identity,
        question_date=deps.question_date,
        gap_queue=[question],
        token_budget=deps.token_budget,
        max_steps=deps.max_steps,
        max_gap_questions=deps.max_gap_questions,
        max_bad_attempts=deps.max_bad_attempts,
        evidence_top_k=deps.evidence_top_k,
        min_evidence=deps.min_evidence,
    )

    while (
        state.total_steps < state.max_steps
        and state.bad_attempts <= state.max_bad_attempts
        and state.token_usage < state.token_budget
        and not state.final_answer
    ):
        state.total_steps += 1
        current_question = state.next_question()
        actions = allowed_actions(state)
        prompt = _controller_prompt(state, current_question, actions)
        action_result = controller.run_sync(prompt)
        state.token_usage += _usage_tokens(action_result)
        action = _coerce_action(_agent_output(action_result), actions, current_question)

        if action.action == "reflect":
            new_gaps = rotate_gap_questions(state, action.gap_questions)
            state.last_answer_failed = False
            state.action_diary.append(
                DiaryEntry(
                    step=state.total_steps,
                    action="reflect",
                    question=current_question,
                    summary=action.rationale or "reflected on missing memory evidence",
                    queries=new_gaps,
                )
            )
            continue

        if action.action == "search":
            queries = rewrite_search_queries(
                action.search_queries,
                current_question=current_question,
                original_question=question,
                attempted_queries=state.attempted_queries,
            )
            if not queries:
                queries = dedupe_queries([current_question], state.attempted_queries)
            refs: List[str] = []
            for query in queries:
                state.attempted_queries.append(query)
                search_result = deps.memory.search(query, top_k=state.evidence_top_k)
                added = add_search_results(state, query, search_result)
                refs.extend(note.ref for note in added)
            state.last_answer_failed = False
            state.action_diary.append(
                DiaryEntry(
                    step=state.total_steps,
                    action="search",
                    question=current_question,
                    summary=action.rationale
                    or f"searched memory; added {len(refs)} evidence notes",
                    queries=queries,
                    refs=refs,
                )
            )
            if not refs and current_question != question:
                state.unresolved_gaps.append(current_question)
            continue

        valid, invalid = validate_citations(action.citations, state.knowledge_ledger)
        eval_result = evaluator.run_sync(_evaluate_prompt(question, action.answer, valid, state))
        state.token_usage += _usage_tokens(eval_result)
        evaluation = _agent_output(eval_result)
        passes = bool(valid) and not invalid and bool(action.answer.strip()) and evaluation.passes
        if passes:
            state.final_answer = action.answer.strip()
            state.final_citations = valid
            state.action_diary.append(
                DiaryEntry(
                    step=state.total_steps,
                    action="answer",
                    question=current_question,
                    summary=evaluation.reason or "answer accepted",
                    refs=valid,
                    accepted=True,
                )
            )
            break

        state.bad_attempts += 1
        state.last_answer_failed = True
        followups = list(evaluation.missing_questions or []) + list(
            evaluation.follow_up_queries or []
        )
        if invalid:
            followups.append("Find support for the answer claims using valid memory citations.")
        if followups:
            rotate_gap_questions(state, followups)
        state.action_diary.append(
            DiaryEntry(
                step=state.total_steps,
                action="answer",
                question=current_question,
                summary=evaluation.reason or "answer rejected",
                refs=valid,
                accepted=False,
            )
        )

    passed = bool(state.final_answer)
    answer = state.final_answer or _fallback_answer(question, state)
    citations = state.final_citations
    if not citations:
        citations = [item.ref for item in state.knowledge_ledger[:8]]
    if not passed and not state.unresolved_gaps:
        state.unresolved_gaps.extend(state.gap_queue[:5])
    return DeepSearchResult(
        question=question,
        answer=answer,
        citations=citations,
        evidence=state.knowledge_ledger,
        unresolved_gaps=dedupe_queries(state.unresolved_gaps),
        diary=state.action_diary,
        trace_stats=state.trace_stats,
        passed=passed,
    )


def _make_planner(model: str):
    Agent, _, _, _, _ = _require_pydantic_ai()
    return Agent(
        model,
        output_type=ResearchPlan,
        instructions=(
            "Create a compact memory-only research plan with 3 to 5 sections over one "
            "person's dated personal memory. Decompose the question into distinct, "
            "non-overlapping facets (e.g. identity/role, relationships, timeline of "
            "recent activity, open questions). Each section needs a title and a "
            "specific question answerable from personal memory."
        ),
    )


def _default_plan(question: str) -> ResearchPlan:
    return ResearchPlan(
        sections=[
            ResearchSectionPlan(title="Direct Answer", question=question),
            ResearchSectionPlan(
                title="Supporting Evidence", question=f"What memory evidence supports: {question}"
            ),
            ResearchSectionPlan(
                title="Gaps and Caveats", question=f"What is missing or uncertain about: {question}"
            ),
        ]
    )


def _dedupe_evidence(evidence: List[EvidenceNote]) -> List[EvidenceNote]:
    seen = set()
    out = []
    for item in evidence:
        if item.ref in seen:
            continue
        seen.add(item.ref)
        out.append(item)
    return out


def _assemble_report(
    question: str, sections: List[ResearchSection], evidence: List[EvidenceNote]
) -> str:
    lines = ["# Memory DeepResearch Report", "", f"Question: {question}", ""]
    for section in sections:
        lines.append(f"## {section.title}")
        lines.append(section.answer.strip() or "No supported answer found for this section.")
        if section.citations:
            lines.append("")
            lines.append("Citations: " + ", ".join(section.citations))
        if section.unresolved_gaps:
            lines.append("")
            lines.append("Open gaps: " + "; ".join(section.unresolved_gaps))
        lines.append("")
    if evidence:
        lines.append("## Evidence Index")
        for item in evidence:
            lines.append(f"- {item.ref}: {item.fact}")
    return "\n".join(lines).strip()


def _coherence_pass(model: str, draft: str) -> str:
    Agent, _, _, _, _ = _require_pydantic_ai()
    agent = Agent(
        model,
        output_type=str,
        instructions=(
            "Improve this memory-grounded report for coherence, consistent terminology, "
            "and reduced redundancy. Preserve all citations exactly; do not add facts."
        ),
    )
    result = agent.run_sync(draft)
    return str(_agent_output(result)).strip() or draft


def run_memory_deep_research(
    question: str,
    deps: MemoryResearchDeps,
    *,
    report_mode: bool = True,
    model: str = DEFAULT_MODEL,
) -> Union[DeepSearchResult, ResearchReport]:
    """Run DeepSearch directly, or DeepResearch report mode over sections."""

    if not report_mode:
        return run_memory_deep_search(question, deps, model=model)

    planner = _make_planner(model)
    try:
        plan = _agent_output(planner.run_sync(question))
    except Exception:
        plan = _default_plan(question)
    if not plan.sections:
        plan = _default_plan(question)
    plan.sections = plan.sections[:5]

    completed: List[ResearchSection] = []
    all_evidence: List[EvidenceNote] = []
    all_diary: List[DiaryEntry] = []
    unresolved: List[str] = []
    trace_stats: Dict[str, Any] = {"sections": len(plan.sections)}

    for section in plan.sections:
        result = run_memory_deep_search(section.question, deps, model=model)
        completed.append(
            ResearchSection(
                title=section.title,
                question=section.question,
                answer=result.answer,
                citations=result.citations,
                unresolved_gaps=result.unresolved_gaps,
            )
        )
        all_evidence.extend(result.evidence)
        all_diary.extend(result.diary)
        unresolved.extend(result.unresolved_gaps)
        trace_stats["searches"] = trace_stats.get("searches", 0) + result.trace_stats.get(
            "searches", 0
        )
        trace_stats["n_scanned_total"] = trace_stats.get(
            "n_scanned_total", 0
        ) + result.trace_stats.get("n_scanned_total", 0)

    all_evidence = _dedupe_evidence(all_evidence)
    draft = _assemble_report(question, completed, all_evidence)
    try:
        report = _coherence_pass(model, draft)
    except Exception:
        report = draft
    return ResearchReport(
        question=question,
        report=report,
        sections=completed,
        evidence=all_evidence,
        unresolved_gaps=dedupe_queries(unresolved),
        trace_stats=trace_stats,
        diary=all_diary,
    )
