"""Official BEAM scoring, vendored from the benchmark's reference code.

Source: github.com/mohammadtavakoli78/BEAM, src/evaluation/compute_metrics.py
and src/evaluation/report_results.py (main branch, retrieved 2026-08-03).

The point of this module is protocol fidelity: prompts, parsing, score
casting, alignment, and aggregation are byte-identical in behavior to the
official code, including its quirks, so our numbers are directly comparable
to the paper's tables:

  * Judge model: gpt-4.1-mini, temperature 0 (per src/llm.py).
  * Per rubric nugget, the judge returns {"score": 1.0|0.5|0.0, "reason": ...}.
  * For the 9 non-ordering categories the official code casts the score with
    int(), so partial 0.5 scores truncate to 0. We keep that.
  * The judge prompt template contains a `<question>` placeholder that the
    official code never fills. We keep that too.
  * event_ordering: system list = response.split("\\n"); LLM pairwise
    equivalence alignment against the rubric; score = normalized Kendall
    tau-b x F1 ("final_score"). The official code also makes an extract_facts
    LLM call whose result is immediately discarded; we skip that dead call
    (zero effect on scores).
  * Aggregation (report_results.py): per conversation, mean over the 2
    questions per category; event_ordering reports `tau_norm` (NOT
    final_score), all others report `llm_judge_score`. Scale score = mean
    over conversations, then mean over the 10 categories.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from scipy.stats import kendalltau

# Verbatim official judge prompt (spliced from src/prompts.py; do not edit).
UNIFIED_LLM_JUDGE_BASE_PROMPT = """
You are an expert evaluator tasked with judging whether the LLM's response demonstrates compliance with the specified RUBRIC CRITERION.

## EVALUATION INPUTS
- QUESTION (what the user asked): <question>
- RUBRIC CRITERION (what to check): <rubric_item>
- RESPONSE TO EVALUATE: <llm_response>

## EVALUATION RUBRIC:
The rubric defines a specific requirement, constraint, or expected behavior that the LLM response should demonstrate. 

**IMPORTANT**: Pay careful attention to whether the rubric specifies:
- **Positive requirements** (things the response SHOULD include/do)
- **Negative constraints** (things the response SHOULD NOT include/do, often indicated by "no", "not", "avoid", "absent")

## RESPONSIVENESS REQUIREMENT (anchored to the QUESTION)
A compliant response must be **on-topic with respect to the QUESTION** and attempt to answer it.
- If the response does not address the QUESTION, score **0.0** and stop.
- For negative constraints, both must hold: (a) the response is responsive to the QUESTION, and (b) the prohibited element is absent.

## SEMANTIC TOLERANCE RULES:
Judge by meaning, not exact wording.
- Accept **paraphrases** and **synonyms** that preserve intent.
- **Case/punctuation/whitespace** differences must be ignored.
- **Numbers/currencies/dates** may appear in equivalent forms (e.g., “$68,000”, “68k”, “68,000 USD”, or “sixty-eight thousand dollars”). Treat them as equal when numerically equivalent.
- If the rubric expects a number or duration, prefer **normalized comparison** (extract and compare values) over string matching.

## STYLE NEUTRALITY (prevents style contamination):
Ignore tone, politeness, length, and flourish unless the rubric explicitly requires a format/structure (e.g., “itemized list”, “no citations”, “one sentence”).
- Do **not** penalize hedging, voice, or verbosity if content satisfies the rubric.
- Only evaluate format when the rubric **explicitly** mandates it.

## SCORING SCALE:
- **1.0 (Complete Compliance)**: Fully complies with the rubric criterion.
  - Positive: required element present, accurate, properly executed (allowing semantic equivalents).
  - Negative: prohibited element **absent** AND response is **responsive**.
  
- **0.5 (Partial Compliance)**: Partially complies.
  - Positive: element present but minor inaccuracies/incomplete execution.
  - Negative: generally responsive and mostly avoids the prohibited element but with minor/edge violations.
  
- **0.0 (No Compliance)**: Fails to comply.
  - Positive: required element missing or incorrect.
  - Negative: prohibited element present **or** response is non-responsive/evasive even if the element is absent.

## EVALUATION INSTRUCTIONS:
1. **Understand the Requirement**: Determine if the rubric is asking for something to be present (positive) or absent (negative/constraint).

2. **Parse Compound Statements**: If the rubric contains multiple elements connected by "and" or commas, evaluate whether:
   - **All elements** must be present for full compliance (1.0)
   - **Some elements** present indicates partial compliance (0.5)
   - **No elements** present indicates no compliance (0.0)
   
3. **Check Compliance**: 
   - For positive requirements: Look for the presence and quality of the required element
   - For negative constraints: Look for the absence of the prohibited element

4. **Assign Score**: Based on compliance with the specific rubric criterion according to the scoring scale above.

5. **Provide Reasoning**: Explain whether the rubric criterion was satisfied and justify the score.

## OUTPUT FORMAT:
Return your evaluation in JSON format with two fields:

{
   "score": [your score: 1.0, 0.5, or 0.0],
   "reason": "[detailed explanation of whether the rubric criterion was satisfied and why this justified the assigned score]"
}

NOTE: ONLY output the json object, without any explanation before or after that
"""

# invoke(messages) -> str, where messages is a list of {"role", "content"}
LLMInvoke = Callable[[List[Dict[str, str]]], str]


def parse_json_response(response: str):
    """Verbatim port of the official parse_json_response."""
    response = response.strip()

    if response.startswith("```"):
        match = re.search(
            r"```(?:json)?\s*(\[.*\]|\{.*\})\s*```", response, re.DOTALL)
        if match:
            response = match.group(1).strip()

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    match = re.search(r'(\{.*?\}|\[.*?\])', response, re.DOTALL)
    if match:
        json_part = match.group(1)
        try:
            return json.loads(json_part)
        except Exception as e:
            raise ValueError(
                f"Found possible JSON but failed to parse it: {e}")

    raise ValueError("No valid JSON found in response.")


def _judge_one_nugget(rubric_item: str, llm_response: str, invoke: LLMInvoke) -> Dict:
    prompt = UNIFIED_LLM_JUDGE_BASE_PROMPT \
        .replace("<rubric_item>", rubric_item) \
        .replace("<llm_response>", llm_response)
    raw = invoke([{"role": "user", "content": prompt}]).strip()
    try:
        return parse_json_response(raw)
    except Exception:
        from json_repair import repair_json
        return json.loads(repair_json(raw))


def _rubric_judge(
    rubric: Sequence[str],
    llm_response: str,
    invoke: LLMInvoke,
    cast: Callable[[Any], float],
) -> Dict:
    """Common body of the official evaluate_<category> functions."""
    responses = []
    score = 0.0
    for item in rubric:
        r = _judge_one_nugget(item, llm_response, invoke)
        score += cast(r["score"])
        responses.append(r)
    return dict(
        llm_judge_score=score / len(rubric),
        llm_judge_responses=responses,
    )


def llm_equivalence(first_paragraph: str, second_paragraph: str, invoke: LLMInvoke) -> bool:
    """Verbatim port (system + user messages, 'yes' substring check)."""
    messages = [
        {
            "role": "system",
            "content": """
            You are a binary classifier.
            If the TWO snippets describe the SAME event/fact, reply **YES**
            Otherwise reply **NO**. No extra words.
            DO NOT provide any exaplanation.
        """,
        },
        {
            "role": "user",
            "content": f"""First snippet: {first_paragraph} \n
                       Second snippet: {second_paragraph}
                    """,
        },
    ]
    return "yes" in invoke(messages).lower()


def align_with_llm(
    reference: List[str], system: List[str], invoke: LLMInvoke
) -> Tuple[List[str], List[str]]:
    used = set()
    system_out = []
    for s in system:
        matched_index = None
        for index, r in enumerate(reference):
            if index in used:
                continue
            if llm_equivalence(r, s, invoke):
                matched_index = index
                break
        if matched_index is not None:
            system_out.append(reference[matched_index])
            used.add(matched_index)
        else:
            system_out.append(s)
    return reference, system_out


def event_ordering_score(
    reference_list: List[str], system_list: List[str], invoke: LLMInvoke
) -> Dict:
    """Verbatim port with align_type='llm' (the official evaluation path)."""
    reference_canon, system_canon = align_with_llm(reference_list, system_list, invoke)

    tp = len(set(reference_canon) & set(system_canon))
    fp = len([x for x in system_canon if x not in reference_canon])
    fn = len([x for x in reference_canon if x not in system_canon])

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    union = list(dict.fromkeys(reference_canon + system_canon))
    tie_rank = len(union) + 1

    def to_rank(seq):
        r = {item: i + 1 for i, item in enumerate(seq)}
        return [r.get(u, tie_rank) for u in union]

    tau_b, _ = kendalltau(
        to_rank(reference_canon), to_rank(system_canon), variant="b", method="auto"
    )
    tau_b_norm = (tau_b + 1) / 2 if tau_b is not None else 0

    return dict(
        precision=precision,
        recall=recall,
        f1=f1,
        tau_norm=tau_b_norm,
        final_score=tau_b_norm * f1,
    )


def evaluate_question(
    category: str,
    rubric: Sequence[str],
    llm_response: str,
    probing_question: str,
    invoke: LLMInvoke,
) -> Dict:
    """Score one answered question exactly as the official code does.

    Returns the same record shape as the official evaluate_<category>
    functions (event_ordering additionally carries tau/f1 fields).
    """
    if category == "event_ordering":
        system_list = llm_response.split("\n")
        record = event_ordering_score(list(rubric), system_list, invoke)
        judged = _rubric_judge(rubric, llm_response, invoke, cast=float)
        record.update(judged)
        return record
    # All other 9 categories: int() cast (0.5 truncates to 0), per official code.
    return _rubric_judge(rubric, llm_response, invoke, cast=lambda s: float(int(s)))


def aggregate_scores(
    per_conversation: List[Dict[str, List[Dict]]],
    categories: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Replicates report_results.py aggregation.

    per_conversation: one dict per conversation mapping category ->
    list of judge records (as returned by evaluate_question).

    Returns per-category means across conversations plus "average"
    (mean of the 10 category columns, as reported in the paper).
    """
    if categories is None:
        categories = sorted({c for conv in per_conversation for c in conv})

    per_cat: Dict[str, float] = {}
    for cat in categories:
        conv_means = []
        for conv in per_conversation:
            records = conv.get(cat, [])
            if not records:
                continue
            key = "tau_norm" if cat == "event_ordering" else "llm_judge_score"
            conv_means.append(sum(r[key] for r in records) / len(records))
        per_cat[cat] = sum(conv_means) / len(conv_means) if conv_means else 0.0

    per_cat["average"] = sum(per_cat[c] for c in categories) / len(categories)
    return per_cat


def make_openai_judge(
    model: str = "gpt-4.1-mini",
    api_key: Optional[str] = None,
    max_retries: int = 5,
) -> LLMInvoke:
    """Official judge: gpt-4.1-mini @ temperature 0 (per BEAM src/llm.py)."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, max_retries=max_retries)

    def invoke(messages: List[Dict[str, str]]) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        return resp.choices[0].message.content or ""

    return invoke
