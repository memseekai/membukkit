"""Fine-tune a bi-encoder (DPR-style) on PerLTQA query->gold-fact pairs.

The learned metric places a query near its gold evidence (attacks asymmetric
retrieval). Train on PerLTQA characters; the routing-headroom diagnostic
evaluates on disjoint held-out characters.

Loss: MultipleNegativesRankingLoss (in-batch negatives).
Base: all-mpnet-base-v2 so frozen-vs-finetuned is apples-to-apples.

Data setup:
    Download PerLTQA from HuggingFace and point --data-dir at the folder.
    The script expects {data_dir}/perltqa_*.json or a huggingface-hub loadable
    variant. See docs/REPRODUCE.md for full instructions.

Usage:
    membukkit train-encoder --data-dir data/perltqa --output models/biencoder_v1
    # or directly:
    python scripts/train_encoder.py --train-offset 0 --train-chars 15 \\
        --epochs 3 --batch-size 32 --output models/biencoder_v1
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def build_pairs(offset: int, n_chars: int, data_dir: str = "") -> List[Tuple[str, str]]:
    """Build (query, gold_fact) pairs from PerLTQA characters [offset, offset+n)."""
    try:
        from membukkit.data.base import FactInput, QueryInput
    except ImportError:
        pass

    # PerLTQA loading: expects a JSON file with character profiles + QA
    pairs: List[Tuple[str, str]] = []
    data_path = Path(data_dir) if data_dir else Path("data/perltqa")
    if not data_path.exists():
        logger.warning(
            f"PerLTQA data not found at {data_path}. "
            "See docs/REPRODUCE.md for data setup instructions."
        )
        return pairs

    import json

    for f in sorted(data_path.glob("*.json")):
        raw = json.loads(f.read_text())
        chars = raw if isinstance(raw, list) else [raw]
        for char in chars[offset : offset + n_chars]:
            facts = char.get("facts", [])
            for qa in char.get("qa", []):
                gold_indices = qa.get("gold_fact_indices", [])
                qt = (qa.get("question", "") or "").strip()
                if not qt:
                    continue
                for gi in gold_indices:
                    if 0 <= gi < len(facts):
                        ft = (
                            facts[gi].get("text", "")
                            if isinstance(facts[gi], dict)
                            else str(facts[gi])
                        ).strip()
                        if ft:
                            pairs.append((qt, ft))
    logger.info(
        f"Built {len(pairs)} (query, gold_fact) pairs from chars [{offset}:{offset + n_chars})."
    )
    return pairs


def run(args) -> None:
    import torch
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer
    from torch.optim.lr_scheduler import LambdaLR

    pairs = build_pairs(args.train_offset, args.train_chars, args.data_dir)
    if not pairs:
        raise SystemExit("No training pairs found. Check --data-dir.")

    model = SentenceTransformer(args.base)
    device = model.device
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scale = 20.0
    bs = args.batch_size
    rng = np.random.default_rng(0)
    n = len(pairs)
    steps_per_epoch = (n - bs + 1 + bs - 1) // bs
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = max(1, int(0.1 * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    sched = LambdaLR(opt, lr_lambda)
    logger.info(
        f"Fine-tuning {args.base}: {n} pairs, {args.epochs} epochs, bs={bs}, "
        f"lr={args.lr}, total_steps={total_steps}, warmup={warmup_steps}"
    )

    def embed(texts):
        feats = model.tokenize(texts)
        feats = {k: v.to(device) for k, v in feats.items()}
        return model(feats)["sentence_embedding"]

    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(n)
        total, steps = 0.0, 0
        for s in range(0, n - bs + 1, bs):
            idx = order[s : s + bs]
            q_texts = [pairs[i][0] for i in idx]
            p_texts = [pairs[i][1] for i in idx]
            q = F.normalize(embed(q_texts), dim=-1)
            p = F.normalize(embed(p_texts), dim=-1)
            scores = (q @ p.T) * scale
            labels = torch.arange(len(idx), device=device)
            loss = F.cross_entropy(scores, labels)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            sched.step()
            total += loss.item()
            steps += 1
            if steps % 100 == 0:
                logger.info(
                    f"  epoch {ep + 1}/{args.epochs} step {steps} "
                    f"loss={total / steps:.4f} lr={sched.get_last_lr()[0]:.2e}"
                )
        logger.info(f"epoch {ep + 1}/{args.epochs} done: mean loss={total / max(1, steps):.4f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    logger.info(f"Saved fine-tuned bi-encoder to {out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="DPR-style bi-encoder fine-tune on PerLTQA")
    p.add_argument("--base", default="sentence-transformers/all-mpnet-base-v2")
    p.add_argument("--data-dir", default="data/perltqa")
    p.add_argument("--train-offset", type=int, default=0)
    p.add_argument("--train-chars", type=int, default=15)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--output", default="models/biencoder_v1")
    args = p.parse_args()
    run(args)
