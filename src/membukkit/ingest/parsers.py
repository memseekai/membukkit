"""Parsers that turn user files into ingestable sessions.

Everything MemBukkit ingests becomes the same shape — a list of *sessions*
(each a list of ``{"role", "content"}`` turns) with optional per-session ISO
dates. Chat-like files map naturally; documents are chunked into passages that
ride in as turns. The mapping is deliberately simple and inspectable: the raw
file content is kept by the LocalStore, and every fact carries a
``source_ref`` back into these sessions.

Supported: .json (chat exports and generic records), .csv, .txt, .md, .pdf,
.zip (ChatGPT/Claude ``conversations*.json``).
"""

from __future__ import annotations

import csv
import json
import re
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union

SUPPORTED_SUFFIXES = {
    ".json",
    ".jsonl",
    ".csv",
    ".txt",
    ".md",
    ".markdown",
    ".pdf",
    ".zip",
}

# Documents get chunked into pseudo-turns of at most this many characters, and
# pseudo-sessions of at most this many turns — sized so the distiller sees
# coherent, prompt-sized windows.
_MAX_TURN_CHARS = 1500
_TURNS_PER_SESSION = 8
_CSV_ROWS_PER_SESSION = 20

_DATE_KEYS = ("date", "created_at", "timestamp", "time", "datetime", "created")
_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


@dataclass
class ParsedDoc:
    """One source document, normalized to ingestable sessions."""

    name: str
    doc_type: str  # "chat" | "records" | "document"
    sessions: List[List[Dict[str, str]]] = field(default_factory=list)
    dates: List[Optional[str]] = field(default_factory=list)
    origin: str = ""  # original filesystem path

    @property
    def n_turns(self) -> int:
        return sum(len(s) for s in self.sessions)


def parse_path(path: str | Path) -> List[ParsedDoc]:
    """Parse a file or directory (recursively) into ParsedDocs."""
    p = Path(path).expanduser()
    if p.is_dir():
        docs: List[ParsedDoc] = []
        for child in sorted(p.rglob("*")):
            if not child.is_file():
                continue
            suf = child.suffix.lower()
            if suf == ".zip":
                docs.extend(_parse_export_zip(child))
            elif suf in SUPPORTED_SUFFIXES:
                docs.append(parse_file(child))
        return docs
    if p.suffix.lower() == ".zip":
        return _parse_export_zip(p)
    return [parse_file(p)]


def parse_file(path: str | Path) -> ParsedDoc:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(str(p))
    suffix = p.suffix.lower()
    if suffix == ".zip":
        docs = _parse_export_zip(p)
        if not docs:
            raise ValueError(
                f"zip {p.name!r} has no conversations.json / conversations-*.json"
            )
        if len(docs) == 1:
            return docs[0]
        # Merge multi-file ChatGPT split exports into one doc.
        merged = ParsedDoc(
            name=p.name, doc_type="chat", origin=str(p), sessions=[], dates=[]
        )
        for d in docs:
            merged.sessions.extend(d.sessions)
            merged.dates.extend(d.dates)
        return merged
    if suffix in (".json", ".jsonl"):
        return _parse_json(p)
    if suffix == ".csv":
        return _parse_csv(p)
    if suffix == ".pdf":
        return _parse_pdf(p)
    if suffix in (".txt", ".md", ".markdown"):
        return _parse_text(p)
    raise ValueError(f"unsupported file type {suffix!r} ({p.name}); supported: "
                     + ", ".join(sorted(SUPPORTED_SUFFIXES)))


def _parse_export_zip(p: Path) -> List[ParsedDoc]:
    """Extract ChatGPT/Claude conversations*.json from a data-export ZIP."""
    docs: List[ParsedDoc] = []
    try:
        with zipfile.ZipFile(p) as zf:
            names = [
                n
                for n in zf.namelist()
                if Path(n).name.startswith("conversations")
                and n.lower().endswith(".json")
                and not n.endswith("/")
            ]
            if not names:
                return []
            with tempfile.TemporaryDirectory(prefix="membukkit-export-") as tmp:
                tmp_path = Path(tmp)
                for name in sorted(names):
                    target = tmp_path / Path(name).name
                    target.write_bytes(zf.read(name))
                    doc = _parse_json(target)
                    doc.origin = f"{p}::{name}"
                    doc.name = f"{p.stem}/{Path(name).name}"
                    docs.append(doc)
    except zipfile.BadZipFile as e:
        raise ValueError(f"invalid zip archive {p.name!r}: {e}") from e
    return docs


# ---------------------------------------------------------------------- JSON


def _find_date(obj: Dict) -> Optional[str]:
    for key in _DATE_KEYS:
        val = obj.get(key)
        if isinstance(val, str):
            m = _ISO_DATE_RE.search(val)
            if m:
                return m.group(0)
        if isinstance(val, (int, float)) and key in ("create_time", "created_at", "timestamp"):
            return _unix_to_date(val)
    return None


def _unix_to_date(ts: Union[int, float]) -> Optional[str]:
    try:
        # ChatGPT sometimes uses ms; treat large values as ms.
        sec = float(ts)
        if sec > 1e12:
            sec /= 1000.0
        return datetime.fromtimestamp(sec, tz=timezone.utc).date().isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def _is_turn(obj) -> bool:
    return isinstance(obj, dict) and "content" in obj and isinstance(obj.get("content"), str)


def _clean_turn(obj: Dict) -> Dict[str, str]:
    return {"role": str(obj.get("role", "user")), "content": obj["content"]}


def _is_chatgpt_conversation(obj: object) -> bool:
    return (
        isinstance(obj, dict)
        and isinstance(obj.get("mapping"), dict)
        and bool(obj.get("mapping"))
    )


def _is_claude_conversation(obj: object) -> bool:
    if not isinstance(obj, dict):
        return False
    msgs = obj.get("chat_messages")
    if not isinstance(msgs, list) or not msgs:
        return False
    sample = msgs[0]
    return isinstance(sample, dict) and (
        "sender" in sample or sample.get("role") in ("human", "assistant", "user")
    )


def _chatgpt_parts_text(content: object) -> str:
    if not isinstance(content, dict):
        return str(content) if content else ""
    parts = content.get("parts") or []
    out: List[str] = []
    for part in parts:
        if isinstance(part, str):
            if part.strip():
                out.append(part)
        elif isinstance(part, dict):
            text = part.get("text") or part.get("content") or ""
            if isinstance(text, str) and text.strip():
                out.append(text)
    return "\n\n".join(out).strip()


def _walk_chatgpt_thread(convo: Dict) -> List[Dict[str, str]]:
    """Follow current_node → parent to reconstruct the visible thread."""
    mapping = convo.get("mapping") or {}
    node_id = convo.get("current_node")
    if not node_id:
        # Fallback: pick a leaf with no children / highest create_time.
        leaves = [
            nid
            for nid, node in mapping.items()
            if isinstance(node, dict) and not (node.get("children") or [])
        ]
        node_id = leaves[-1] if leaves else None
    path_ids: List[str] = []
    seen = set()
    while node_id and node_id not in seen:
        seen.add(node_id)
        path_ids.append(node_id)
        node = mapping.get(node_id) or {}
        node_id = node.get("parent")
    path_ids.reverse()
    turns: List[Dict[str, str]] = []
    for nid in path_ids:
        node = mapping.get(nid) or {}
        msg = node.get("message")
        if not isinstance(msg, dict):
            continue
        author = msg.get("author") or {}
        role = str(author.get("role") or "user")
        if role in ("system", "tool"):
            continue
        text = _chatgpt_parts_text(msg.get("content"))
        if not text:
            continue
        if role not in ("user", "assistant"):
            role = "user"
        turns.append({"role": role, "content": text})
    return turns


def _parse_chatgpt_export(data, name: str, origin: str) -> Optional[ParsedDoc]:
    convos: List[Dict] = []
    if _is_chatgpt_conversation(data):
        convos = [data]  # type: ignore[list-item]
    elif isinstance(data, list) and data and all(_is_chatgpt_conversation(x) for x in data):
        convos = data  # type: ignore[assignment]
    else:
        return None
    doc = ParsedDoc(name=name, doc_type="chat", origin=origin)
    for convo in convos:
        turns = _walk_chatgpt_thread(convo)
        if not turns:
            continue
        doc.sessions.append(turns)
        ts = convo.get("create_time") or convo.get("update_time")
        if isinstance(ts, (int, float)):
            doc.dates.append(_unix_to_date(ts))
        else:
            doc.dates.append(_find_date(convo) if isinstance(convo, dict) else None)
    return doc if doc.sessions else None


def _claude_message_text(msg: Dict) -> str:
    if isinstance(msg.get("text"), str) and msg["text"].strip():
        return msg["text"].strip()
    content = msg.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type") or ""
            if btype in ("thinking", "tool_use", "tool_result"):
                continue
            text = block.get("text") or block.get("content") or ""
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return "\n\n".join(parts).strip()
    return ""


def _parse_claude_export(data, name: str, origin: str) -> Optional[ParsedDoc]:
    convos: List[Dict] = []
    if _is_claude_conversation(data):
        convos = [data]  # type: ignore[list-item]
    elif isinstance(data, list) and data and all(_is_claude_conversation(x) for x in data):
        convos = data  # type: ignore[assignment]
    else:
        return None
    doc = ParsedDoc(name=name, doc_type="chat", origin=origin)
    for convo in convos:
        turns: List[Dict[str, str]] = []
        for msg in convo.get("chat_messages") or []:
            if not isinstance(msg, dict):
                continue
            sender = str(msg.get("sender") or msg.get("role") or "human")
            role = "user" if sender in ("human", "user") else "assistant"
            text = _claude_message_text(msg)
            if text:
                turns.append({"role": role, "content": text})
        if not turns:
            continue
        doc.sessions.append(turns)
        created = convo.get("created_at") or convo.get("updated_at")
        if isinstance(created, str):
            m = _ISO_DATE_RE.search(created)
            doc.dates.append(m.group(0) if m else None)
        else:
            doc.dates.append(_find_date(convo))
    return doc if doc.sessions else None


def _parse_json(p: Path) -> ParsedDoc:
    if p.suffix.lower() == ".jsonl":
        data = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    else:
        data = json.loads(p.read_text())

    chatgpt = _parse_chatgpt_export(data, p.name, str(p))
    if chatgpt is not None:
        return chatgpt
    claude = _parse_claude_export(data, p.name, str(p))
    if claude is not None:
        return claude

    doc = ParsedDoc(name=p.name, doc_type="chat", origin=str(p))

    # {"sessions": [...], "dates": [...]} — MemBukkit's native shape
    if isinstance(data, dict) and isinstance(data.get("sessions"), list):
        doc.sessions = [[_clean_turn(t) for t in s if _is_turn(t)] for s in data["sessions"]]
        raw_dates = data.get("dates") or [None] * len(doc.sessions)
        doc.dates = [d if isinstance(d, str) else None for d in raw_dates]
        return doc

    # {"messages": [...]} — OpenAI-style single conversation
    if isinstance(data, dict) and isinstance(data.get("messages"), list):
        turns = [_clean_turn(t) for t in data["messages"] if _is_turn(t)]
        doc.sessions = [turns]
        doc.dates = [_find_date(data)]
        return doc

    if isinstance(data, list) and data:
        # list of turns — one session
        if all(_is_turn(x) for x in data):
            doc.sessions = [[_clean_turn(t) for t in data]]
            doc.dates = [None]
            return doc
        # list of sessions (lists of turns)
        if all(isinstance(x, list) for x in data) and all(
            _is_turn(t) for x in data for t in x
        ):
            doc.sessions = [[_clean_turn(t) for t in s] for s in data]
            doc.dates = [None] * len(doc.sessions)
            return doc
        # list of session objects: {"turns"/"messages": [...], "date": ...}
        if all(isinstance(x, dict) for x in data):
            turn_key = None
            for key in ("turns", "messages", "conversation"):
                if all(isinstance(x.get(key), list) for x in data):
                    turn_key = key
                    break
            if turn_key:
                for x in data:
                    doc.sessions.append([_clean_turn(t) for t in x[turn_key] if _is_turn(t)])
                    doc.dates.append(_find_date(x))
                return doc
            # generic records (tickets, events, rows-as-objects)
            doc.doc_type = "records"
            for i in range(0, len(data), _CSV_ROWS_PER_SESSION):
                batch = data[i : i + _CSV_ROWS_PER_SESSION]
                turns = [{"role": "user", "content": _record_to_text(r)} for r in batch]
                doc.sessions.append(turns)
                doc.dates.append(_find_date(batch[0]))
            return doc

    # anything else: stringify as one document chunk
    doc.doc_type = "document"
    text = json.dumps(data, ensure_ascii=False, indent=1)
    doc.sessions, doc.dates = _chunk_document(text)
    return doc


def _record_to_text(record: Dict) -> str:
    parts = []
    for k, v in record.items():
        if v is None or v == "":
            continue
        if isinstance(v, (dict, list)):
            v = json.dumps(v, ensure_ascii=False)
        parts.append(f"{k}: {v}")
    return "; ".join(parts)


# ----------------------------------------------------------------------- CSV


def _parse_csv(p: Path) -> ParsedDoc:
    doc = ParsedDoc(name=p.name, doc_type="records", origin=str(p))
    with open(p, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for i in range(0, len(rows), _CSV_ROWS_PER_SESSION):
        batch = rows[i : i + _CSV_ROWS_PER_SESSION]
        turns = [{"role": "user", "content": _record_to_text(r)} for r in batch]
        doc.sessions.append(turns)
        doc.dates.append(_find_date(batch[0]) if batch else None)
    return doc


# ---------------------------------------------------------------- TXT/MD/PDF


def _split_paragraphs(text: str) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    # Split oversized paragraphs on sentence-ish boundaries.
    out: List[str] = []
    for para in paras:
        while len(para) > _MAX_TURN_CHARS:
            cut = para.rfind(". ", 0, _MAX_TURN_CHARS)
            cut = cut + 1 if cut > _MAX_TURN_CHARS // 2 else _MAX_TURN_CHARS
            out.append(para[:cut].strip())
            para = para[cut:].strip()
        if para:
            out.append(para)
    return out


def _chunk_document(text: str):
    paras = _split_paragraphs(text)
    sessions, dates = [], []
    for i in range(0, len(paras), _TURNS_PER_SESSION):
        turns = [{"role": "user", "content": t} for t in paras[i : i + _TURNS_PER_SESSION]]
        sessions.append(turns)
        dates.append(None)
    return sessions, dates


def _parse_text(p: Path) -> ParsedDoc:
    text = p.read_text(errors="replace")
    wa = _parse_whatsapp(text)
    if wa is not None:
        sessions, dates = wa
        return ParsedDoc(
            name=p.name, doc_type="chat", sessions=sessions, dates=dates, origin=str(p)
        )
    doc = ParsedDoc(name=p.name, doc_type="document", origin=str(p))
    doc.sessions, doc.dates = _chunk_document(text)
    return doc


# ------------------------------------------------------------------ WhatsApp

# Invisible directional marks WhatsApp sprinkles through exports (U+200E/U+200F
# and friends) — stripped before any matching.
_WA_MARKS_RE = re.compile(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]")

# "[18/04/2026, 12:44:37] Name: text"  (iOS classic; seconds optional)
_WA_BRACKET_RE = re.compile(
    r"^\[(\d{1,2})/(\d{1,2})/(\d{2,4}),?\s+(\d{1,2}):(\d{2})(?::(\d{2}))?"
    r"(?:\s*[AaPp][Mm]\.?)?\]\s*(.*)$"
)
# "18/04/2026, 12:44 - Name: text"  (Android variant, no brackets)
_WA_DASH_RE = re.compile(
    r"^(\d{1,2})/(\d{1,2})/(\d{2,4}),?\s+(\d{1,2}):(\d{2})(?::(\d{2}))?"
    r"(?:\s*[AaPp][Mm]\.?)?\s+-\s+(.*)$"
)

_WA_MEDIA_RE = re.compile(
    r"^(?:<media omitted>|<attached:[^>]*>|(?:image|video|audio|sticker|gif|document"
    r"|contact card) omitted|this message was deleted\.?|you deleted this message\.?|null)$",
    re.IGNORECASE,
)

_WA_SYSTEM_RE = re.compile(
    r"(messages and calls are end-to-end encrypted"
    r"|created this group"
    r"|created group "
    r"|you were added"
    r"|added you\b"
    r"|joined using this group's invite link"
    r"|changed the subject"
    r"|changed this group's icon"
    r"|changed the group description"
    r"|pinned a message"
    r"|turned on disappearing messages"
    r"|turned off disappearing messages"
    r"|your security code with)",
    re.IGNORECASE,
)


def _wa_match(line: str):
    return _WA_BRACKET_RE.match(line) or _WA_DASH_RE.match(line)


def _wa_date(m: re.Match) -> str:
    a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if y < 100:
        y += 2000
    # Day-first (the common export locale); swap only when impossible.
    day, month = (a, b) if a > 12 or b <= 12 else (b, a)
    if month > 12:
        day, month = month, day
    return f"{y:04d}-{month:02d}-{day:02d}"


def _parse_whatsapp(text: str):
    """Detect and parse a WhatsApp chat export.

    Returns (sessions, dates) — one session per calendar day, one turn per
    message with the speaker kept in the content as "Name: text" — or None
    when the text does not look like a WhatsApp export. System notices and
    media placeholders ("image omitted") are dropped; continuation lines of
    multi-line messages attach to the preceding message.
    """
    lines = _WA_MARKS_RE.sub("", text).splitlines()
    nonempty = [ln for ln in lines if ln.strip()]
    n_headers = sum(1 for ln in nonempty if _wa_match(ln.strip()))
    if n_headers < 3 or n_headers < 0.3 * len(nonempty):
        return None

    # messages: [iso_date, speaker, [body lines]]
    messages: List[List] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        m = _wa_match(line)
        if not m:
            if messages:
                messages[-1][2].append(line)
            continue
        rest = m.group(7).strip()
        speaker, sep, body = rest.partition(": ")
        if not sep and rest.endswith(":"):
            speaker, sep, body = rest[:-1], ":", ""
        if not sep or not speaker.strip():
            continue  # system notice without a sender, or a bare timestamp
        messages.append([_wa_date(m), speaker.strip(), [body.strip()]])

    sessions: List[List[Dict[str, str]]] = []
    dates: List[Optional[str]] = []
    for date, speaker, body_lines in messages:
        body = "\n".join(ln for ln in body_lines if ln).strip()
        if not body or _WA_MEDIA_RE.match(body) or _WA_SYSTEM_RE.search(body):
            continue
        if not sessions or dates[-1] != date:
            sessions.append([])
            dates.append(date)
        sessions[-1].append({"role": "user", "content": f"{speaker}: {body}"})

    # Days that held only system/media messages leave no session behind.
    keep = [i for i, s in enumerate(sessions) if s]
    return [sessions[i] for i in keep], [dates[i] for i in keep]


def _parse_pdf(p: Path) -> ParsedDoc:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError(
            "PDF support needs the 'pdf' extra: pip install 'membukkit[pdf]'"
        ) from e
    reader = PdfReader(str(p))
    text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
    doc = ParsedDoc(name=p.name, doc_type="document", origin=str(p))
    doc.sessions, doc.dates = _chunk_document(text)
    return doc
