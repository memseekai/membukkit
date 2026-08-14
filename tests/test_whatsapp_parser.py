"""WhatsApp chat-export parsing: sessions per day, speakers kept, noise dropped."""

from __future__ import annotations

from membukkit.ingest.parsers import parse_file

IOS_EXPORT = (
    "[18/04/2026, 12:44:37] Girls takeover: \u200eMessages and calls are end-to-end "
    "encrypted. Only people in this chat can read, listen to, or share them.\n"
    "[18/04/2026, 12:44:37] Gaia Bianciotto: \u200eGaia Bianciotto created this group\n"
    "[18/04/2026, 12:44:37] Girls takeover: \u200eYou were added\n"
    "[18/04/2026, 12:45:02] Gaia Bianciotto: Ciao ragazze! Beach day on Saturday?\n"
    "[18/04/2026, 12:46:11] Zak: I'm in! I'll bring the frisbee\n"
    "and maybe some sandwiches\n"
    "[18/04/2026, 12:47:00] Marta: \u200eimage omitted\n"
    "\u200e[26/04/2026, 19:31:36]\n"
    "[19/04/2026, 09:15:30] Marta: I booked the van for 10am\n"
    "[19/04/2026, 09:16:00] Zak: \u200esticker omitted\n"
    "[19/04/2026, 09:17:45] Gaia Bianciotto: Perfect, see you at Piazza Duomo\n"
)


def test_ios_export_parses_to_daily_sessions(tmp_path):
    f = tmp_path / "_chat.txt"
    f.write_text(IOS_EXPORT, encoding="utf-8")
    doc = parse_file(f)

    assert doc.doc_type == "chat"
    assert doc.dates == ["2026-04-18", "2026-04-19"]
    assert len(doc.sessions) == 2

    day1, day2 = doc.sessions
    # System notices ("created this group", "You were added", e2e banner) and
    # the media placeholder are dropped; real messages keep "Name: text".
    assert [t["content"] for t in day1] == [
        "Gaia Bianciotto: Ciao ragazze! Beach day on Saturday?",
        "Zak: I'm in! I'll bring the frisbee\nand maybe some sandwiches",
    ]
    assert [t["content"] for t in day2] == [
        "Marta: I booked the van for 10am",
        "Gaia Bianciotto: Perfect, see you at Piazza Duomo",
    ]
    assert all(t["role"] == "user" for s in doc.sessions for t in s)
    # invisible direction marks are stripped everywhere
    assert all("\u200e" not in t["content"] for s in doc.sessions for t in s)


def test_android_export_without_brackets(tmp_path):
    text = (
        "18/04/2026, 12:45 - Gaia: Beach day on Saturday?\n"
        "18/04/2026, 12:46 - Zak: I'm in\n"
        "18/04/2026, 12:47 - Marta: <Media omitted>\n"
        "19/04/2026, 09:15 - Marta: I booked the van\n"
    )
    f = tmp_path / "chat.txt"
    f.write_text(text, encoding="utf-8")
    doc = parse_file(f)

    assert doc.doc_type == "chat"
    assert doc.dates == ["2026-04-18", "2026-04-19"]
    assert [t["content"] for t in doc.sessions[0]] == [
        "Gaia: Beach day on Saturday?",
        "Zak: I'm in",
    ]
    assert [t["content"] for t in doc.sessions[1]] == ["Marta: I booked the van"]


def test_plain_text_still_chunks_as_document(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text(
        "Meeting notes from April.\n\nWe decided to ship the parser on 18/04.\n",
        encoding="utf-8",
    )
    doc = parse_file(f)
    assert doc.doc_type == "document"
    assert doc.dates == [None]
    assert doc.sessions and doc.sessions[0][0]["role"] == "user"
