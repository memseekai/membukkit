"""Evaluation metrics for MEMBUKKIT."""


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = set(prediction.lower().split())
    gt_tokens = set(ground_truth.lower().split())
    if not gt_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    common = pred_tokens & gt_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
