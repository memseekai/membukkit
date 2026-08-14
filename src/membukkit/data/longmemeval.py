"""LongMemEval dataset loader for MEMBUKKIT."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from membukkit.data.instance import LongMemEvalDataset, LongMemEvalInstance

logger = logging.getLogger(__name__)


def load_longmemeval(
    variant: str = "longmemeval_s",
    cache_dir: Optional[str] = None,
    max_instances: Optional[int] = None,
) -> LongMemEvalDataset:
    """Load LongMemEval from HuggingFace.

    Args:
        variant: which file to load — "longmemeval_s" (default, ~115k tokens/instance)
                 or "longmemeval_m" (~500 sessions/instance)
        cache_dir: HuggingFace cache directory
        max_instances: cap number of instances (for dev/debug)

    Returns:
        LongMemEvalDataset with all instances.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("pip install huggingface-hub to use LongMemEval")

    VARIANT_TO_FILE = {
        "longmemeval_s": "longmemeval_s_cleaned.json",
        "longmemeval_m": "longmemeval_m_cleaned.json",
        "longmemeval_oracle": "longmemeval_oracle.json",
    }
    filename = VARIANT_TO_FILE.get(variant, f"{variant}.json")
    logger.info(f"Downloading LongMemEval ({filename}) from HuggingFace...")

    local_path = hf_hub_download(
        repo_id="xiaowu0162/longmemeval-cleaned",
        filename=filename,
        repo_type="dataset",
        cache_dir=cache_dir,
    )

    logger.info(f"Loading LongMemEval from {local_path}...")
    with open(local_path, "r") as f:
        raw_data = json.load(f)

    if max_instances is not None:
        raw_data = raw_data[:max_instances]

    instances = [LongMemEvalInstance(item) for item in raw_data]
    dataset = LongMemEvalDataset(instances)

    logger.info(
        f"LongMemEval loaded: {len(instances)} instances, "
        f"abilities: {dataset.summary()}"
    )
    return dataset
