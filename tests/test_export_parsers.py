"""ChatGPT / Claude / zip export parsers."""

from __future__ import annotations

import zipfile
from pathlib import Path

from membukkit.ingest.parsers import parse_file, parse_path

FIXTURES = Path(__file__).parent / "fixtures" / "exports"


def test_chatgpt_mapping_walker():
    doc = parse_file(FIXTURES / "chatgpt_conversations.json")
    assert doc.doc_type == "chat"
    assert len(doc.sessions) == 1
    roles = [t["role"] for t in doc.sessions[0]]
    assert "system" not in roles
    texts = " ".join(t["content"] for t in doc.sessions[0])
    assert "800" in texts
    assert "950" in texts
    assert doc.dates[0] is not None


def test_claude_chat_messages():
    doc = parse_file(FIXTURES / "claude_conversations.json")
    assert len(doc.sessions) == 1
    assert doc.sessions[0][0]["role"] == "user"
    assert "vegetarian" in doc.sessions[0][0]["content"].lower()
    assert any("fish" in t["content"].lower() for t in doc.sessions[0])
    assert doc.dates[0] == "2024-03-15"


def test_zip_export(tmp_path):
    zpath = tmp_path / "export.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.write(FIXTURES / "chatgpt_conversations.json", "conversations.json")
    docs = parse_path(zpath)
    assert len(docs) == 1
    assert docs[0].sessions
