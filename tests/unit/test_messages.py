"""Unit tests for the message catalogue (PRD 8.3)."""

from __future__ import annotations

import pytest

from baton.common import messages
from baton.common.framing import decode, encode_bytes
from baton.common.messages import (
    CONTROL_DOWN_TYPES,
    CONTROL_UP_TYPES,
    DATA_TYPES,
    MESSAGE_TYPES,
    Act,
    Health,
    Hello,
    Load,
    Prompt,
    Token,
    message_type,
)

# The catalogue exactly as PRD 8.3 tabulates it.
PRD_CONTROL_UP = {
    "hello",
    "health",
    "bench_result",
    "load_progress",
    "loaded",
    "link_down",
    "error",
    "token",
    "release_ack",
    "pong_peer",
}
PRD_CONTROL_DOWN = {"welcome", "bench", "load", "unload", "ping_peer", "abort"}
PRD_DATA = {"prompt", "next", "act", "release", "abort"}


def test_catalogue_matches_the_prd():
    assert CONTROL_UP_TYPES == PRD_CONTROL_UP
    assert CONTROL_DOWN_TYPES == PRD_CONTROL_DOWN
    assert DATA_TYPES == PRD_DATA
    assert MESSAGE_TYPES == PRD_CONTROL_UP | PRD_CONTROL_DOWN | PRD_DATA


def test_every_typed_dict_declares_a_known_type():
    """Each TypedDict's `t` Literal must name a type in the catalogue."""
    seen = set()
    for name in messages.__all__:
        cls = getattr(messages, name)
        hints = getattr(cls, "__annotations__", None)
        if not isinstance(hints, dict) or "t" not in hints:
            continue
        (literal,) = hints["t"].__args__
        assert literal in MESSAGE_TYPES, f"{name} declares unknown t={literal!r}"
        seen.add(literal)
    # `abort` has one class shared by the control and data planes, so the set of
    # declared literals is the whole catalogue.
    assert seen == MESSAGE_TYPES


def test_required_and_optional_keys_of_hello():
    assert Hello.__required_keys__ == frozenset(
        {"t", "name", "token", "version", "caps", "data_addr"}
    )
    assert Hello.__optional_keys__ == frozenset()


def test_health_makes_loaded_rev_optional():
    assert "loaded_rev" in Health.__optional_keys__


def test_token_reason_is_optional():
    assert Token.__required_keys__ == frozenset({"t", "req", "id", "pos", "final"})
    assert Token.__optional_keys__ == frozenset({"reason"})


def test_load_carries_the_6_4_fields_plus_index_and_headers():
    assert Load.__required_keys__ == frozenset(
        {
            "t",
            "plan_rev",
            "model",
            "range",
            "quant",
            "roles",
            "ctx_max",
            "kv_budget_bytes",
            "next_node",
            "head_data_addr",
            "index",
            "headers",
        }
    )
    assert Load.__optional_keys__ == frozenset({"hf_token"})


def test_a_typed_dict_is_a_plain_dict_and_survives_a_frame():
    msg: Act = {"t": "act", "req": "r1", "pos": 3, "n": 2, "dtype": "bf16", "trace": []}
    assert isinstance(msg, dict)
    payload = b"\x00\x01" * 8
    got_meta, got_payload = decode(encode_bytes(msg, payload))
    assert got_meta == msg
    assert got_payload == payload


def test_message_type_accepts_every_catalogue_entry():
    for t in MESSAGE_TYPES:
        assert message_type({"t": t}) == t


def test_message_type_rejects_an_unknown_type():
    with pytest.raises(ValueError, match="unknown message type"):
        message_type({"t": "teleport"})


@pytest.mark.parametrize("meta", [{}, {"t": 7}, {"t": None}])
def test_message_type_rejects_a_missing_or_non_string_t(meta):
    with pytest.raises(TypeError, match="no string 't'"):
        message_type(meta)


def test_prompt_is_control_free_of_payload_fields():
    msg: Prompt = {
        "t": "prompt",
        "req": "r1",
        "ids": [128000, 9906],
        "max_len": 128,
        "sampling": {"temperature": 0.7, "top_p": 0.95},
        "stop_ids": [128009],
        "trace": [],
    }
    assert message_type(msg) == "prompt"
    assert decode(encode_bytes(msg))[1] == b""
