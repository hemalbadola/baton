"""Forward engine tests (PRD 6.6). Two engines on one process play a two-node
ring: the test carries each `Out` where the ring would carry it."""

import torch

from baton.model.layers import DecoderStack
from baton.worker.engine import ForwardEngine, KVPool

PROMPT = [1, 5, 9, 2, 7, 3]


def split(whole: DecoderStack, cut: int = 2) -> tuple[ForwardEngine, ForwardEngine]:
    spec = whole.spec
    state = whole.state_dict()
    names = whole.checkpoint_keys()

    def engine(start: int, end: int, *, first: bool, last: bool) -> ForwardEngine:
        stack = DecoderStack(
            spec, start, end, embed=first, head=last, max_ctx=whole.max_ctx, dtype=torch.float32
        )
        # Only the tensors this shard owns are needed, by checkpoint name.
        own = set(stack.checkpoint_keys().values())
        stack.load_hf_weights({n: state[k] for k, n in names.items() if n in own})
        per_token = 2 * spec.n_kv_heads * spec.head_dim * 4
        kv = KVPool(10**9, end - start, per_token)
        return ForwardEngine(
            stack.eval(), kv, device="cpu", dtype=torch.float32, wire="fp32", ctx_max=64
        )

    return engine(0, cut, first=True, last=False), engine(
        cut, spec.n_layers, first=False, last=True
    )


def run_ring(n1: ForwardEngine, nk: ForwardEngine, first: dict, steps: int) -> list[int]:
    """Carry frames until `steps` tokens came out. Returns the token ids."""
    tokens: list[int] = []
    queue = [("n1", first, b"")]
    while queue and len(tokens) < steps:
        who, meta, payload = queue.pop(0)
        for out in (n1 if who == "n1" else nk).handle(meta, payload):
            if out.to == "head":
                if out.meta["t"] == "token":
                    tokens.append(out.meta["id"])
            elif who == "n1":  # N1's next node is Nk
                queue.append(("nk", out.meta, out.payload))
            else:  # Nk's next node is N1
                queue.append(("n1", out.meta, out.payload))
    return tokens


def prompt_frame(n: int) -> dict:
    return {
        "t": "prompt",
        "req": "r",
        "ids": PROMPT,
        "max_len": len(PROMPT) + n,
        "sampling": {"temperature": 0.0},
        "stop_ids": [],
        "trace": [],
    }


def test_two_engines_match_the_whole_model(tiny_checkpoint) -> None:
    _, _, whole = tiny_checkpoint
    n1, nk = split(whole)
    got = run_ring(n1, nk, prompt_frame(6), 6)

    # Reference: the same stack, one cache, argmax.
    ref = ForwardEngine(
        whole, KVPool(10**9, whole.n_local_layers, 2 * whole.spec.n_kv_heads * whole.spec.head_dim * 4),
        device="cpu", dtype=torch.float32, wire="fp32", ctx_max=64,
    )  # fmt: skip
    queue = [prompt_frame(6)]
    want: list[int] = []
    while queue and len(want) < 6:
        for out in ref.handle(queue.pop(0)):
            if out.meta["t"] == "token":
                want.append(out.meta["id"])
            elif out.to == "next":
                queue.append(out.meta)
    assert got == want and len(got) == 6


def test_a_frame_for_an_unknown_request_is_dropped(tiny_checkpoint) -> None:
    _, _, whole = tiny_checkpoint
    n1, nk = split(whole)
    assert n1.handle({"t": "next", "req": "gone", "id": 3, "pos": 4}) == []
    assert (
        nk.handle({"t": "act", "req": "gone", "pos": 0, "n": 1, "dtype": "fp32", "last": True}, b"")
        == []
    )


def test_release_frees_kv_and_only_the_last_node_acks(tiny_checkpoint) -> None:
    _, _, whole = tiny_checkpoint
    n1, nk = split(whole)
    n1.handle(prompt_frame(2))
    nk.handle(prompt_frame(2))
    assert n1.active_reqs == nk.active_reqs == 1
    forwarded = n1.handle({"t": "release", "req": "r"})
    assert [o.to for o in forwarded] == ["next"] and n1.active_reqs == 0
    acked = nk.handle({"t": "release", "req": "r"})
    assert [(o.to, o.meta["t"]) for o in acked] == [("head", "release_ack")]
    assert nk.active_reqs == 0
    assert n1.kv.used_bytes == nk.kv.used_bytes == 0


def test_a_stop_id_ends_the_request_on_the_token_frame(tiny_checkpoint) -> None:
    _, _, whole = tiny_checkpoint
    n1, nk = split(whole)
    first = run_ring(n1, nk, prompt_frame(1), 1)[0]
    n1, nk = split(whole)
    frame = prompt_frame(5) | {"stop_ids": [first]}
    queue = [("n1", frame, b"")]
    final = None
    while queue and final is None:
        who, meta, payload = queue.pop(0)
        for out in (n1 if who == "n1" else nk).handle(meta, payload):
            if out.to == "head" and out.meta["t"] == "token":
                final = out.meta
            elif out.to == "next":
                queue.append(("nk" if who == "n1" else "n1", out.meta, out.payload))
    assert final["final"] and final["reason"] == "stop"
