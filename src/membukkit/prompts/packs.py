"""Load use-case prompt packs into PromptConfig."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

from membukkit.config import PromptConfig

_PACKS_DIR = Path(__file__).resolve().parent / "packs"


def list_prompt_packs() -> List[Dict[str, str]]:
    """Built-in packs: ``[{id, title, description}, ...]``."""
    out = []
    if not _PACKS_DIR.is_dir():
        return out
    for path in sorted(_PACKS_DIR.glob("*.yaml")):
        meta = _read_yaml(path)
        out.append(
            {
                "id": path.stem,
                "title": str(meta.get("title") or path.stem.replace("_", " ").title()),
                "description": str(meta.get("description") or ""),
            }
        )
    return out


def load_prompt_pack(name_or_path: Union[str, Path]) -> PromptConfig:
    """Load a shipped pack id (e.g. ``customer_support``) or a YAML file path."""
    path = Path(name_or_path)
    if path.suffix in {".yaml", ".yml"} and path.is_file():
        data = _read_yaml(path)
    else:
        candidate = _PACKS_DIR / f"{name_or_path}.yaml"
        if not candidate.is_file():
            known = ", ".join(p["id"] for p in list_prompt_packs()) or "(none)"
            raise FileNotFoundError(
                f"unknown prompt pack {name_or_path!r}; known: {known}"
            )
        data = _read_yaml(candidate)
    # Drop pack metadata keys that are not PromptConfig fields.
    data.pop("title", None)
    data.pop("description", None)
    return PromptConfig.from_dict(data)


def _read_yaml(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ImportError:
        return _parse_simple_yaml(text)
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError(f"prompt pack {path} must be a mapping")
    return data


def _parse_simple_yaml(text: str) -> dict:
    """Minimal YAML subset for pack files when PyYAML is not installed.

    Supports ``key: value`` and ``key: |`` multiline blocks. Good enough for
    the shipped instruction-only packs; full templates should use PyYAML.
    """
    out: dict = {}
    key = None
    multiline = False
    buf: List[str] = []
    for raw in text.splitlines():
        if multiline:
            if raw.startswith("  ") or raw.startswith("\t") or raw.strip() == "":
                buf.append(raw[2:] if raw.startswith("  ") else raw)
                continue
            out[key] = "\n".join(buf).rstrip("\n")
            multiline = False
            buf = []
            key = None
            # fall through to parse this line
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        k, _, rest = line.partition(":")
        k = k.strip()
        rest = rest.strip()
        if rest == "|" or rest == ">":
            key = k
            multiline = True
            buf = []
            continue
        if (rest.startswith('"') and rest.endswith('"')) or (
            rest.startswith("'") and rest.endswith("'")
        ):
            rest = rest[1:-1]
        out[k] = rest
    if multiline and key is not None:
        out[key] = "\n".join(buf).rstrip("\n")
    return out
