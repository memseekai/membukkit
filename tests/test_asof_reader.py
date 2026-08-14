"""As-of / effective-date framing and day-granularity guidance for readers."""

from membukkit.prompts.reading import DATED_READER_PROMPT, REASONING_READER_PROMPT
from membukkit.reading.readers import _today_line, make_dated_reader, make_reasoning_reader


def test_today_line_frames_as_of():
    line = _today_line("2024-04-15")
    assert "2024-04-15" in line
    assert "as of" in line.lower() or "Answer strictly as of" in line
    assert "not yet current" in line


def test_dated_prompt_effective_date_and_day_granularity():
    assert "takes effect AFTER today" in DATED_READER_PROMPT
    assert "not yet current" in DATED_READER_PROMPT
    assert "day when known" in DATED_READER_PROMPT
    assert "do not coarsen to month-only" in DATED_READER_PROMPT
    # Changelog overlay is opt-in; stock current-state prompts stay clean.
    assert "memory timeline" not in DATED_READER_PROMPT


def test_reasoning_prompt_in_effect_not_latest_announcement():
    assert "in effect as of today" in REASONING_READER_PROMPT
    assert "most recently stated announcement" in REASONING_READER_PROMPT
    assert "day granularity" in REASONING_READER_PROMPT


def test_dated_reader_injects_effective_framing():
    captured = {}

    def llm(prompt: str) -> str:
        captured["p"] = prompt
        return "800€, changing to 950€ from June"

    reader = make_dated_reader(llm)
    facts = [
        "[2024-01-08] rent is 800€",
        "[2024-04-02] rent is going up to 950€ from June",
    ]
    assert reader(facts, "How much is my rent?", "2024-04-15") == (
        "800€, changing to 950€ from June"
    )
    assert "2024-04-15" in captured["p"]
    assert "not yet current" in captured["p"]
    assert "day when known" in captured["p"]


def test_reasoning_reader_injects_effective_framing():
    captured = {}

    def llm(prompt: str) -> str:
        captured["p"] = prompt
        return "Answer: 800€ until June; rises to 950€ then"

    reader = make_reasoning_reader(llm)
    out = reader(
        ["[2024-01-08] rent 800", "[2024-04-02] rent 950 from June"],
        "How much is rent and when did it change?",
        "2024-04-15",
    )
    assert "800" in out
    assert "in effect as of today" in captured["p"] or "not yet current" in captured["p"]
