"""One-time: publish the fine-tuned encoder/reranker weights to the HF Hub.

Requires `hf auth login` with write access to the `MemseekAI` org.

    python scripts/publish_weights.py --models-dir /path/to/models
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

REPOS = {
    "biencoder_v1": "MemseekAI/membukkit-biencoder-v1",
    "reranker_v2/model": "MemseekAI/membukkit-reranker-v2",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    api = HfApi()
    for local, repo_id in REPOS.items():
        folder = Path(args.models_dir) / local
        if not folder.exists():
            raise SystemExit(f"missing: {folder}")
        api.create_repo(repo_id, exist_ok=True, private=args.private)
        api.upload_folder(folder_path=str(folder), repo_id=repo_id)
        print(f"uploaded {folder} -> {repo_id}")


if __name__ == "__main__":
    main()
