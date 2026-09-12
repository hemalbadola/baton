"""The head's HTTP surface is a contract (PRD 13, 14.3). These tests pin it."""

import pytest

pytest.importorskip("fastapi", reason="head extra not installed")

from pydantic import ValidationError

from baton.head.api import (
    ERROR_STATUS,
    ChatCompletionRequest,
    ClusterSnapshot,
    error_response,
    router,
)

PATHS = {"/v1/models", "/v1/chat/completions", "/v1/completions", "/healthz", "/cluster", "/ws"}

SNAPSHOT = {
    "state": "READY",
    "model": {
        "id": "meta-llama/Llama-3.1-70B-Instruct",
        "quant": "int4",
        "ctx": 8192,
        "n_layers": 80,
    },
    "plan_rev": 3,
    "nodes": [
        {
            "name": "mac",
            "role": "N1",
            "backend": "mps",
            "link": "wifi",
            "layers": [0, 25],
            "mem": {"total": 13.0e9, "weights": 11.4e9, "kv_used": 4.0e8, "kv_budget": 1.3e9},
            "stage_ms": {"p50": 74.1, "p95": 91.0},
            "queue_depth": 0,
            "state": "loaded",
        }
    ],
    "hops_ms": [2.9, 0.4],
    "live": {"tok_s": 1.68, "ttft_ms": 6120, "active": 1, "queued": 0},
    "events": [{"t": 1725700000.1, "kind": "loss", "msg": "worker 'old' lost"}],
}


def test_every_documented_path_is_routed():
    assert PATHS <= {r.path for r in router.routes}


def test_snapshot_matches_the_prd_example():
    snap = ClusterSnapshot.model_validate(SNAPSHOT)
    assert snap.nodes[0].layers == (0, 25)
    assert snap.live.tok_s == pytest.approx(1.68)


def _chat(**over):
    body = {"model": "x", "messages": [{"role": "user", "content": "hi"}]}
    body.update(over)
    return ChatCompletionRequest.model_validate(body)


def test_defaults_follow_13_2():
    req = _chat()
    assert (req.max_tokens, req.temperature, req.top_p, req.top_k) == (1024, 0.7, 1.0, 0)
    assert req.repetition_penalty == 1.0 and req.stream is False


@pytest.mark.parametrize(
    "over",
    [{"n": 2}, {"stop": ["a", "b", "c", "d", "e"]}, {"tools": [{"type": "function"}]}],
)
def test_unsupported_requests_are_rejected(over):
    with pytest.raises(ValidationError):
        _chat(**over)


def test_empty_tools_list_is_accepted():
    assert _chat(tools=[]).tools == []


def test_error_body_and_status_map():
    body = error_response("worker 'mac' lost", "worker_lost")
    assert body["error"]["type"] == "server_error"
    assert body["error"]["code"] == "worker_lost"
    assert ERROR_STATUS["timeout"] == 504
    assert all(ERROR_STATUS[c] == 503 for c in ("not_ready", "worker_lost", "oom", "queue_timeout"))
