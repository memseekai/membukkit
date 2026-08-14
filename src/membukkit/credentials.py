"""Local LLM API credentials for GUI / CLI (outside shell ``export``).

Keys may come from the process environment (highest priority) or from a
local file under ``~/.membukkit/credentials.env`` (mode ``0600``), typically
set from the GUI when a user has no shell key configured.

Never log or return full secret values over the API — only presence + a short
mask (last 4 chars).
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SUPPORTED_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OLLAMA_HOST",
)

_FILE_KEYS = SUPPORTED_KEYS

# Populated by bootstrap_credentials(): keys already set before file load.
_PREEXISTING_ENV: set[str] = set()
_BOOTSTRAPPED = False


def membukkit_home() -> Path:
    home = os.environ.get("MEMBUKKIT_HOME")
    return Path(home).expanduser() if home else Path.home() / ".membukkit"


def credentials_path() -> Path:
    return membukkit_home() / "credentials.env"


def _mask(value: str) -> str:
    v = (value or "").strip()
    if not v:
        return ""
    if len(v) <= 8:
        return "••••"
    return f"…{v[-4:]}"


def _parse_env_file(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if key in _FILE_KEYS and val:
            out[key] = val
    return out


def read_credentials_file() -> Dict[str, str]:
    path = credentials_path()
    if not path.is_file():
        return {}
    try:
        return _parse_env_file(path.read_text(encoding="utf-8"))
    except OSError:
        return {}


def write_credentials_file(values: Dict[str, str]) -> Path:
    """Merge non-empty supported keys; empty string deletes that key. Mode 0600."""
    path = credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = read_credentials_file()
    for key in _FILE_KEYS:
        if key not in values:
            continue
        val = (values.get(key) or "").strip()
        if val:
            merged[key] = val
        else:
            merged.pop(key, None)
    lines = [
        "# MemBukkit local credentials — do not commit.",
        "# Set from the GUI or edit carefully. Loaded when env vars are unset.",
        "",
    ]
    for key in _FILE_KEYS:
        if key in merged:
            lines.append(f"{key}={merged[key]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600
    except OSError:
        pass
    return path


def load_credentials_file_into_environ() -> List[str]:
    """Load file keys into env only where the env var is currently unset."""
    applied: List[str] = []
    for key, val in read_credentials_file().items():
        if key not in SUPPORTED_KEYS:
            continue
        v = (val or "").strip()
        if not v or (os.environ.get(key) or "").strip():
            continue
        os.environ[key] = v
        applied.append(key)
    return applied


def bootstrap_credentials() -> List[str]:
    """Call once at process start: remember preexisting env, then load file."""
    global _PREEXISTING_ENV, _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return []
    _BOOTSTRAPPED = True
    _PREEXISTING_ENV = {k for k in SUPPORTED_KEYS if (os.environ.get(k) or "").strip()}
    return load_credentials_file_into_environ()


def _provider_ready(llm_spec: str) -> Tuple[str, bool]:
    """Return (needs_id, ready) for the active LLM spec."""
    spec = (llm_spec or "").lower()
    openai_set = bool((os.environ.get("OPENAI_API_KEY") or "").strip())
    anthropic_set = bool((os.environ.get("ANTHROPIC_API_KEY") or "").strip())
    google_set = bool(
        (os.environ.get("GEMINI_API_KEY") or "").strip()
        or (os.environ.get("GOOGLE_API_KEY") or "").strip()
    )

    if spec.startswith("anthropic:") or spec.startswith("claude"):
        return "anthropic", anthropic_set
    if spec.startswith("google:") or spec.startswith("gemini:") or spec.startswith("vertex:"):
        return "google", google_set
    if spec.startswith("ollama:"):
        return "ollama", True
    if spec.startswith("local:") or spec.startswith("compat:"):
        return "other", True
    # Default openai:* / bare model names
    return "openai", openai_set


def _source_for(key: str) -> str:
    val = (os.environ.get(key) or "").strip()
    if not val:
        return "none"
    if key in _PREEXISTING_ENV:
        return "env"
    file_val = (read_credentials_file().get(key) or "").strip()
    if file_val and file_val == val:
        return "file"
    return "env"


def key_status(llm_spec: str = "") -> Dict:
    """Public status payload for the GUI (no full secrets)."""
    needs, ready = _provider_ready(llm_spec)

    def _entry(key: str) -> Dict:
        val = (os.environ.get(key) or "").strip()
        return {
            "set": bool(val),
            "mask": _mask(val) if val else "",
            "source": _source_for(key),
        }

    openai = _entry("OPENAI_API_KEY")
    anthropic = _entry("ANTHROPIC_API_KEY")
    gemini = _entry("GEMINI_API_KEY")
    google_alt = _entry("GOOGLE_API_KEY")
    google = gemini if gemini["set"] else google_alt
    if gemini["set"] or google_alt["set"]:
        google = {
            "set": True,
            "mask": gemini["mask"] or google_alt["mask"],
            "source": gemini["source"] if gemini["set"] else google_alt["source"],
        }
    else:
        google = {"set": False, "mask": "", "source": "none"}

    return {
        "llm": llm_spec or "",
        "needs": needs,
        "ready": bool(ready),
        "credentials_path": str(credentials_path()),
        "providers": {
            "openai": openai,
            "anthropic": anthropic,
            "google": google,
            "ollama": _entry("OLLAMA_HOST"),
        },
    }


def set_keys(
    *,
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    ollama_host: Optional[str] = None,
    persist: bool = True,
) -> Tuple[List[str], Optional[str]]:
    """Apply keys to the process (and optionally the credentials file).

    Empty string clears that key from the file (and from env if it matched file).
    Returns (applied_keys, credentials_path or None).
    """
    mapping = {
        "OPENAI_API_KEY": openai_api_key,
        "ANTHROPIC_API_KEY": anthropic_api_key,
        "GEMINI_API_KEY": gemini_api_key,
        "OLLAMA_HOST": ollama_host,
    }
    to_file: Dict[str, str] = {}
    applied: List[str] = []
    for key, val in mapping.items():
        if val is None:
            continue
        stripped = val.strip()
        to_file[key] = stripped
        if stripped:
            os.environ[key] = stripped
            applied.append(key)
            _PREEXISTING_ENV.discard(key)  # treat as file-origin after GUI set
        else:
            file_vals = read_credentials_file()
            if os.environ.get(key) == file_vals.get(key):
                os.environ.pop(key, None)
            _PREEXISTING_ENV.discard(key)

    path_str = None
    if persist:
        path = write_credentials_file(to_file)
        path_str = str(path)
    return applied, path_str
