"""Incremental detokenization on the head. PRD 7.4.

M1 interface only. Every body raises NotImplementedError.

The head holds the tokenizer, the workers do not. Nk sends token ids, the head
turns them into the text of an SSE chunk. Two problems make this more than one
`decode` call per token:

1. A multi-byte character can span several tokens. A partial decode ends in
   U+FFFD, and the head must hold that text back until the next token arrives.
2. Decoding the whole id list on every token is quadratic over a response. The
   head decodes from a moving `prefix_start` instead, and advances it to the
   last safe boundary every 64 tokens.

Multi-token stop strings are detected here, not on Nk. Text after a stop string
is never emitted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

PREFIX_ADVANCE_EVERY = 64
REPLACEMENT_CHAR = "�"


@dataclass
class Emission:
    """What one token id produced.

    `text` is the new text for the SSE chunk. It is empty when the decode still
    waits for the rest of a character. `stop` is the stop string that ended the
    response, or None.
    """

    text: str
    stop: str | None = None

    @property
    def finished(self) -> bool:
        """True when a stop string ended this response."""
        raise NotImplementedError


@dataclass
class IncrementalDetokenizer:
    """Turns a stream of token ids into a stream of text pieces (PRD 7.4).

    `tokenizer` is a Hugging Face tokenizer. The annotation is Any because the
    model lane owns the loading of it.

    One instance per request. It is not thread safe: the driver drives it from
    the single task that reads that request's token frames.
    """

    tokenizer: Any
    stop_strings: Sequence[str] = ()
    ids: list[int] = field(default_factory=list)
    prefix_start: int = 0
    prev_text: str = ""
    emitted: str = ""

    def push(self, token_id: int) -> Emission:
        """Append one id and return the text that it completes.

        Returns an empty `text` when the decode ends in U+FFFD: the character
        is incomplete, so the head waits for the next token. Sets `stop` when
        the emitted text plus this piece contains a stop string, and truncates
        the piece so that no text after the stop string is ever returned.
        """
        raise NotImplementedError

    def advance_prefix(self) -> None:
        """Move `prefix_start` to the last safe boundary (PRD 7.4).

        The driver calls this every `PREFIX_ADVANCE_EVERY` tokens to keep each
        decode constant in the length of the response. A safe boundary is a
        position whose decode does not end in U+FFFD.
        """
        raise NotImplementedError

    def find_stop(self, candidate: str) -> tuple[str, str] | None:
        """Match the stop strings against the tail of the emitted text.

        Only the last `max(len(s) for s in stop_strings)` characters of the
        emitted text plus `candidate` need a check. Returns
        `(stop_string, kept_text)`, where `kept_text` is the part of `candidate`
        before the stop string. Returns None when no stop string matched.
        """
        raise NotImplementedError

    def text(self) -> str:
        """Every character emitted so far for this request."""
        raise NotImplementedError


def single_token_stop_ids(tokenizer: Any, stop_strings: Sequence[str]) -> set[int]:
    """Ids of the stop strings that tokenize to one token (PRD 7.3 step 5).

    Those ids go to Nk inside the `prompt` frame, so Nk stops the ring itself
    and computes no token after the stop. Every other stop string is matched
    here on the head, which costs one race window of dropped tokens.
    """
    raise NotImplementedError
