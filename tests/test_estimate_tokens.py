"""Reader token estimate for ask receipts."""

from membukkit.pipeline import estimate_tokens


def test_estimate_tokens_chars_div_4():
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 400) == 100
    assert estimate_tokens("hello", "world") == len("helloworld") // 4
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0  # type: ignore[arg-type]
