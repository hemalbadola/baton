"""Incremental detokenization (PRD 7.4)."""

from typing import ClassVar

from baton.head.detok import IncrementalDetokenizer, single_token_stop_ids


class Tok:
    """Ids 1 and 2 are the two bytes of one character. Other ids are words."""

    WORDS: ClassVar[dict[int, str]] = {3: "he", 4: "llo", 5: "#", 6: "##", 7: " end"}

    def decode(self, ids):
        raw = b"".join(
            {1: b"\xe2\x82", 2: b"\xac"}.get(i, self.WORDS.get(i, "?").encode()) for i in ids
        )
        return raw.decode("utf-8", errors="replace")

    def encode(self, text, add_special_tokens=True):
        return [i for i, w in self.WORDS.items() if w == text]


def run(ids, stops=()):
    d = IncrementalDetokenizer(Tok(), tuple(stops))
    pieces = []
    for i in ids:
        e = d.push(i)
        pieces.append(e.text)
        if e.finished:
            return d, pieces, e.stop
    pieces.append(d.flush())
    return d, pieces, None


def test_a_character_split_across_tokens_waits_for_its_second_half() -> None:
    _, pieces, _ = run([3, 1, 2, 4])
    assert pieces == ["he", "", "€", "llo", ""]


def test_a_stop_string_spanning_tokens_never_leaks_its_first_half() -> None:
    d, pieces, stop = run([3, 4, 5, 6, 7], stops=["###"])
    assert stop == "###"
    assert "".join(pieces) == "hello"
    assert d.text() == "hello"


def test_held_text_that_is_not_a_stop_string_is_released() -> None:
    _, pieces, stop = run([3, 5, 4], stops=["###"])
    assert stop is None
    assert "".join(pieces) == "he#llo"


def test_held_text_is_flushed_at_the_end() -> None:
    _, pieces, _ = run([3, 5], stops=["###"])
    assert "".join(pieces) == "he#"


def test_the_prefix_advances_without_changing_the_text() -> None:
    ids = [3, 4] * 70
    d, pieces, _ = run(ids)
    assert "".join(pieces) == "hello" * 70
    assert d.prefix_start > 0


def test_only_single_token_stop_strings_become_ids() -> None:
    assert single_token_stop_ids(Tok(), ["#", "##", "nope"]) == {5, 6}
