"""Judge functions for MEMBUKKIT evaluation."""

from __future__ import annotations

import re
from typing import Callable, List

from membukkit.eval.metrics import f1_score
from membukkit.reading.readers import make_mem0_judge


def make_judge_fn(llm_fn: Callable[[str], str]) -> Callable[..., float]:
    """Create an LLM judge function for scoring answers (generic 0-1 scorer)."""

    def judge(question: str, answer: str, ground_truth: str) -> float:
        prompt = (
            "You are evaluating a memory-based QA system.\n"
            "Rate how well the answer addresses the question compared to the ground truth.\n"
            "Score from 0.0 to 1.0:\n"
            "- 1.0: perfect match in meaning\n"
            "- 0.7-0.9: mostly correct, minor omissions\n"
            "- 0.3-0.6: partially correct\n"
            "- 0.0-0.2: wrong or irrelevant\n\n"
            f"Question: {question}\n"
            f"Ground truth: {ground_truth}\n"
            f"System answer: {answer}\n\n"
            "Score (just the number):"
        )
        try:
            resp = llm_fn(prompt).strip()
            match = re.search(r"(\d+\.?\d*)", resp)
            if match:
                return min(1.0, max(0.0, float(match.group(1))))
        except Exception:
            pass
        return f1_score(answer, ground_truth)

    return judge


def _official_anscheck_prompt(
    task: str, question: str, answer: str, response: str, abstention: bool = False
) -> str:
    """Verbatim LongMemEval judge prompts (xiaowu0162/LongMemEval evaluate_qa.py)."""
    if abstention:
        return (
            "I will give you an unanswerable question, an explanation, and a response "
            "from a model. Please answer yes if the model correctly identifies the question "
            "as unanswerable. The model could say that the information is incomplete, or some "
            "other information is given but the asked information is not.\n\n"
            f"Question: {question}\n\nExplanation: {answer}\n\nModel Response: {response}\n\n"
            "Does the model correctly identify the question as unanswerable? Answer yes or no only."
        )
    if task in ("single-session-user", "single-session-assistant", "multi-session"):
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response is equivalent to the correct answer or contains all the intermediate "
            "steps to get the correct answer, you should also answer yes. If the response only "
            "contains a subset of the information required by the answer, answer no.\n\n"
            f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    if task == "temporal-reasoning":
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response is equivalent to the correct answer or contains all the intermediate "
            "steps to get the correct answer, you should also answer yes. If the response only "
            "contains a subset of the information required by the answer, answer no. In addition, "
            "do not penalize off-by-one errors for the number of days. If the question asks for the "
            "number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., "
            "predicting 19 days when the answer is 18), the model's response is still correct.\n\n"
            f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    if task == "knowledge-update":
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response contains some previous information along with an updated answer, the "
            "response should be considered as correct as long as the updated answer is the required "
            f"answer.\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    if task == "single-session-preference":
        return (
            "I will give you a question, a rubric for desired personalized response, and a response "
            "from a model. Please answer yes if the response satisfies the desired response. Otherwise, "
            "answer no. The model does not need to reflect all the points in the rubric. The response is "
            "correct as long as it recalls and utilizes the user's personal information correctly.\n\n"
            f"Question: {question}\n\nRubric: {answer}\n\nModel Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    return _official_anscheck_prompt("single-session-user", question, answer, response)


def make_official_judge(llm_fn: Callable[[str], str]) -> Callable[..., float]:
    """Official LongMemEval binary judge (1.0 = yes, 0.0 = no), task-routed.

    Signature: judge(question, answer, response, question_type="", abstention=False).
    """

    def judge(
        question: str, answer: str, response: str, question_type: str = "", abstention: bool = False
    ) -> float:
        prompt = _official_anscheck_prompt(question_type, question, answer, response, abstention)
        try:
            resp = llm_fn(prompt).strip().lower()
            return 1.0 if "yes" in resp else 0.0
        except Exception:
            return 0.0

    return judge
