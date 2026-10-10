"""Incremental detokenization on the head. PRD 7.4.

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
        return self.stop is not None


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
    pending: str = ""

    def push(self, token_id: int) -> Emission:
        """Append one id and return the text that it completes.

        Returns an empty `text` when the decode ends in U+FFFD: the character
        is incomplete, so the head waits for the next token. Text that could be
        the start of a stop string is held back in `pending` until the next
        token shows what it is, so a stop string that spans tokens never leaks
        its first half. Sets `stop` when a stop string completes, and returns
        only the text before it.
        """
        self.ids.append(token_id)
        full = self.tokenizer.decode(self.ids[self.prefix_start :])
        if full.endswith(REPLACEMENT_CHAR):
            return Emission("")
        piece = full[len(self.prev_text) :]
        self.prev_text = full
        if len(self.ids) - self.prefix_start >= PREFIX_ADVANCE_EVERY:
            self.advance_prefix()
        found = self.find_stop(piece)
        if found is not None:
            stop, kept = found
            self.pending = ""
            self.emitted += kept
            return Emission(kept, stop)
        window = self.pending + piece
        hold = max(
            (
                k
                for stop in self.stop_strings
                for k in range(1, len(stop))
                if window.endswith(stop[:k])
            ),
            default=0,
        )
        out = window[: len(window) - hold]
        self.pending = window[len(window) - hold :]
        self.emitted += out
        return Emission(out)

    def flush(self) -> str:
        """The held-back text, once the response ended without that stop string."""
        out, self.pending = self.pending, ""
        self.emitted += out
        return out

    def advance_prefix(self) -> None:
        """Move `prefix_start` to the last safe boundary (PRD 7.4).

        The driver calls this every `PREFIX_ADVANCE_EVERY` tokens to keep each
        decode constant in the length of the response. A safe boundary is a
        position whose decode does not end in U+FFFD.
        """
        if not self.prev_text.endswith(REPLACEMENT_CHAR):
            self.prefix_start = len(self.ids)
            self.prev_text = ""

    def find_stop(self, candidate: str) -> tuple[str, str] | None:
        """Match the stop strings against the held text plus `candidate`.

        Returns `(stop_string, kept_text)`, where `kept_text` is the part of
        the window before the stop string. Returns None when none matched.
        """
        window = self.pending + candidate
        hits = [(i, s) for s in self.stop_strings if (i := window.find(s)) >= 0]
        if not hits:
            return None
        at, stop = min(hits)
        return stop, window[:at]

    def text(self) -> str:
        """Every character emitted so far for this request."""
        return self.emitted


def single_token_stop_ids(tokenizer: Any, stop_strings: Sequence[str]) -> set[int]:
    """Ids of the stop strings that tokenize to one token (PRD 7.3 step 5).

    Those ids go to Nk inside the `prompt` frame, so Nk stops the ring itself
    and computes no token after the stop. Every other stop string is matched
    here on the head, which costs one race window of dropped tokens.
    """
    ids: set[int] = set()
    for stop in stop_strings:
        encoded = tokenizer.encode(stop, add_special_tokens=False)
        if len(encoded) == 1:
            ids.add(int(encoded[0]))
    return ids
