"""
LLM atomic-fact distiller (the "better fact representation" lever).

Raw facts are whole conversation turns: long, chatty, with the signal buried
and coreference unresolved. This module distills turns into ATOMIC,
self-contained, dated, user-centric facts so that (a) retrieval units are
clean and (b) the answer is explicit for the reader.

Design:
  - Distill at SESSION granularity (within-session context resolves coreference).
  - One LLM call per session -> a list of atomic user-facts (cached to disk by
    content hash, so re-runs share work and cost nothing twice).
  - Each atomic fact inherits the session's date as its timestamp.

Usage is via `FactDistiller`:
  d = FactDistiller(make_llm_backend("openai", model="gpt-4o-mini"), cache_path=...)
  d.warm(jobs, workers=16)
  facts = d.distill(key, transcript_lines, date_str)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from membukkit.config import PromptConfig
from membukkit.prompts.resolve import (
    DOC_PROMPT_VERSION,
    NAMED_PROMPT_VERSION,
    PROMPT_VERSION,
    resolve_extraction_template,
)

logger = logging.getLogger(__name__)

# Re-export version tags for callers/tests that import them from this module.
__all_prompt_versions__ = (PROMPT_VERSION, NAMED_PROMPT_VERSION, DOC_PROMPT_VERSION)

# Per-turn transcript cap. 600 chars fits LongMemEval/LoCoMo turns; BEAM turns
# average ~1,900 chars, so BEAM runs set MEMBUKKIT_DISTILL_MAX_TURN_CHARS=4000.
# Changing the cap changes transcripts and therefore distill cache keys, so
# existing caches are never silently mixed across cap values.
_MAX_TURN_CHARS = int(os.environ.get("MEMBUKKIT_DISTILL_MAX_TURN_CHARS", "600"))
_DISTILL_ATTEMPTS = 3


class DistillationError(RuntimeError):
    """An LLM distillation call failed after retries.

    The failed session is NOT cached, so a later ingest/warm retries it.
    (Previously failures were silently cached as an empty fact list, which
    permanently lost the session's facts.)
    """


_BULLET = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s*")
_TURN_TAG = re.compile(r"^\s*(?:\[?T)?(\d+)\]?\s*[|:\-]\s*(.+)$")


def _clean_line(line: str) -> str:
    return _BULLET.sub("", line).strip().strip('"')


def parse_facts(raw: str) -> List[List]:
    """Parse '<turn_idx> | <fact>' lines into [[turn_idx:int, fact:str], ...].

    Falls back to turn_idx=-1 for lines that don't carry an index.
    """
    out: List[List] = []
    for line in (raw or "").splitlines():
        s = line.strip()
        if not s or s.upper() == "NONE":
            continue
        m = _TURN_TAG.match(s)
        if m:
            idx = int(m.group(1))
            fact = _clean_line(m.group(2))
        else:
            idx = -1
            fact = _clean_line(s)
        if len(fact) < 4:
            continue
        out.append([idx, fact])
    return out


def build_transcript(turns: List[Tuple[str, str]], numbered: bool = False) -> str:
    """turns: list of (role, content) -> a compact transcript string.

    When numbered, each turn is prefixed with its 0-based index as [T{i}] so the
    distiller can backpointer each fact to a source turn.
    """
    lines = []
    for i, (role, content) in enumerate(turns):
        c = (content or "").strip().replace("\n", " ")
        if len(c) > _MAX_TURN_CHARS:
            c = c[:_MAX_TURN_CHARS] + "…"
        if c:
            lines.append(f"[T{i}] {role}: {c}" if numbered else f"{role}: {c}")
    return "\n".join(lines)


def make_key(*parts) -> str:
    h = hashlib.sha1("||".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return h[:16]


class FactDistiller:
    def __init__(
        self,
        llm_fn: Callable[[str], str],
        cache_path: Optional[str] = None,
        subject: Optional[str] = None,
        prompts: Optional[PromptConfig] = None,
    ):
        self.llm_fn = llm_fn
        self.subject = subject
        self.prompts = prompts or PromptConfig.default()
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: Dict[str, List[List]] = {}
        self._lock = threading.Lock()
        self._calls = 0
        if self.cache_path and self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text())
                logger.info(
                    f"FactDistiller: loaded {len(self._cache)} cached sessions from {self.cache_path}"
                )
            except Exception:
                self._cache = {}
        elif self.cache_path:
            logger.warning(
                "FactDistiller: distill cache not found at %s (cwd=%s); "
                "starting with an empty cache — use an absolute path or run from "
                "the directory that contains the cache file",
                self.cache_path.resolve(),
                Path.cwd(),
            )

    def _template_and_version(self, mode: str = "chat") -> Tuple[str, str]:
        return resolve_extraction_template(self.prompts, mode=mode, subject=self.subject)

    def _distill_uncached(
        self, transcript: str, date_str: str, mode: str = "chat"
    ) -> List[List]:
        template, _ = self._template_and_version(mode)
        if mode != "document" and self.subject:
            prompt = template.format(
                subject=self.subject, date=date_str or "unknown date", transcript=transcript
            )
        else:
            prompt = template.format(
                date=date_str or "unknown date", transcript=transcript
            )
        last_err: Optional[Exception] = None
        for attempt in range(1, _DISTILL_ATTEMPTS + 1):
            try:
                raw = self.llm_fn(prompt)
                self._calls += 1
                return parse_facts(raw)
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.warning(f"distill attempt {attempt}/{_DISTILL_ATTEMPTS} failed: {e}")
        # Raise instead of returning [] — an empty list means "the LLM found no
        # facts" and gets cached; a failure must stay uncached and retryable.
        raise DistillationError(
            f"distillation failed after {_DISTILL_ATTEMPTS} attempts: {last_err}"
        ) from last_err

    def _vkey(self, key: str, mode: str = "chat") -> str:
        _, ver = self._template_and_version(mode)
        if mode == "document":
            return f"{ver}:{key}"
        if self.subject:
            return f"{ver}:{self.subject}:{key}"
        return f"{ver}:{key}"

    def distill(
        self, key: str, transcript: str, date_str: str, mode: str = "chat"
    ) -> List[List]:
        """Distill one session's transcript into atomic facts.

        ``mode`` selects the extraction prompt: "chat" (default) is the
        user↔assistant prompt (or the named-subject variant when ``subject``
        is set); "document" is the subject-agnostic prompt for documents,
        records, and multi-speaker chats. Each mode has its own prompt-version
        cache prefix, so switching modes never reads the other mode's entries.
        """
        vk = self._vkey(key, mode)
        with self._lock:
            if vk in self._cache:
                return self._cache[vk]
        facts = self._distill_uncached(transcript, date_str, mode)
        with self._lock:
            self._cache[vk] = facts
        return facts

    def warm(
        self,
        jobs: List[Tuple[str, str, str]],
        workers: int = 16,
        progress_every: int = 500,
        progress_cb: Optional[Callable[[int, int], None]] = None,
        save_every: int = 200,
        save_interval_s: float = 30.0,
    ) -> None:
        """Parallel pre-fill. jobs = list of (key, transcript, date_str).

        If ``progress_cb`` is given it is called once per completed session as
        ``progress_cb(done, total)`` (total = number of uncached sessions), which
        callers use to drive a progress bar; the periodic log line is then
        suppressed to avoid duplicate/noisy output.

        Progress is checkpointed to disk incrementally — every ``save_every``
        completed sessions or every ``save_interval_s`` seconds, whichever comes
        first — and once more on exit (including on KeyboardInterrupt). A long
        distill run can therefore be interrupted and resumed with at most a
        handful of sessions redone, since the cache is content-addressed.
        """
        todo = [j for j in jobs if self._vkey(j[0]) not in self._cache]
        if not todo:
            logger.info("FactDistiller.warm: all cached, nothing to do")
            return
        logger.info(
            f"FactDistiller.warm: {len(todo)} sessions to distill "
            f"({len(jobs) - len(todo)} already cached), workers={workers}"
        )
        done = 0
        failed = 0
        last_save = time.monotonic()
        interrupted = False

        def _do(job):
            key, transcript, date_str = job
            facts = self._distill_uncached(transcript, date_str)
            return self._vkey(key), facts

        ex = ThreadPoolExecutor(max_workers=workers)
        futs = [ex.submit(_do, j) for j in todo]
        try:
            for fut in as_completed(futs):
                try:
                    vk, facts = fut.result()
                except DistillationError as e:
                    # Leave the session uncached so the next warm/ingest retries it.
                    failed += 1
                    logger.warning(f"FactDistiller.warm: session failed, not cached: {e}")
                    if progress_cb is not None:
                        progress_cb(done, len(todo))
                    continue
                with self._lock:
                    self._cache[vk] = facts
                done += 1
                if progress_cb is not None:
                    progress_cb(done, len(todo))
                elif done % progress_every == 0:
                    logger.info(f"  distilled {done}/{len(todo)}")
                now = time.monotonic()
                if done % save_every == 0 or (now - last_save) >= save_interval_s:
                    self.save()
                    last_save = now
        except KeyboardInterrupt:
            interrupted = True
            logger.warning(
                "FactDistiller.warm: interrupted after %d/%d sessions — "
                "checkpointing cache before exit (safe to resume)",
                done,
                len(todo),
            )
        finally:
            # Don't block on the remaining pool on interrupt; just persist what
            # we have. shutdown() is idempotent so the happy path is unaffected.
            ex.shutdown(wait=not interrupted, cancel_futures=interrupted)
            self.save()

        if interrupted:
            raise KeyboardInterrupt
        if failed:
            logger.warning(
                f"FactDistiller.warm: {failed}/{len(todo)} sessions failed and were "
                "NOT cached — re-run warm/ingest to retry them"
            )
        logger.info(f"FactDistiller.warm: done ({self._calls} live LLM calls)")

    def drop_empty_entries(self) -> int:
        """Remove cached sessions with zero facts; returns how many were dropped.

        Caches written before failures raised may hold `[]` for sessions whose
        LLM call failed — indistinguishable from a genuine "no facts" result, so
        this repair is opt-in. Dropped sessions re-distill on the next
        ingest/warm (the distiller is content-hash cached, so unaffected
        sessions cost nothing).
        """
        with self._lock:
            empty = [k for k, v in self._cache.items() if not v]
            for k in empty:
                del self._cache[k]
        if empty:
            self.save()
            logger.info(f"FactDistiller: dropped {len(empty)} empty cache entries")
        return len(empty)

    def save(self) -> None:
        if not self.cache_path:
            return
        with self._lock:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            # Write to a temp file then atomically replace, so an interrupt mid
            # write can never leave a truncated/corrupt cache on disk.
            tmp = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
            tmp.write_text(json.dumps(self._cache))
            os.replace(tmp, self.cache_path)
