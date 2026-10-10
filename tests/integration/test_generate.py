"""A prompt goes around a real ring and tokens come back (BAT-38, BAT-39).

One head and two workers on one machine over loopback, the tiny random model,
a fake tokenizer. The ring must produce the same greedy tokens as the whole
model in one process, over the driver and over HTTP.

Sockets need the sandbox off. A `PermissionError` on bind is the sandbox.
"""

import asyncio
import contextlib
import copy
import json
import re
import socket

import httpx
import pytest
import torch

from baton.head import serve
from baton.head.driver import Sampling
from baton.worker import probe
from baton.worker.daemon import WorkerConfig, WorkerDaemon
from baton.worker.engine import LayerKV

GB = 1024**3


class FakeTokenizer:
    """`<7><3>` is the ids [7, 3]. Enough to test the head without a model repo."""

    eos_token_id = None
    unk_token_id = None

    def encode(self, text, add_special_tokens=True):
        return [int(x) for x in re.findall(r"<(\d+)>", text)]

    def decode(self, ids):
        return "".join(f"<{i}>" for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "".join(f"<{len(m['content'])}>" for m in messages)

    def convert_tokens_to_ids(self, token):
        return None


async def until(done, timeout=60.0) -> None:
    async with asyncio.timeout(timeout):
        while not done():
            await asyncio.sleep(0.05)


# This tiny tied model repeats its last token under argmax, which cannot show a
# wrong position or a lost row. A seeded sample gives a varied reference.
SAMPLING = Sampling(temperature=40.0, seed=3)


def greedy(whole, prompt, n, temperature=40.0, device="cpu"):
    """Reference: the whole model, one process, one KV cache, the same seeded sampler."""
    from baton.worker.sampler import Sampler, SamplingParams

    sampler = Sampler(SamplingParams(temperature=temperature, seed=3), device=device)
    spec = whole.spec
    dtype = next(whole.parameters()).dtype
    kv = [
        LayerKV(len(prompt) + n, spec.n_kv_heads, spec.head_dim, device, dtype)
        for _ in range(whole.n_local_layers)
    ]
    out: list[int] = []
    ids = list(prompt)
    pos = 0
    with torch.inference_mode():
        for _ in range(n):
            x = whole.embed(torch.tensor(ids))
            logits = whole(x, pos_start=pos, cache=kv, last_only=True)
            pos += len(ids)
            tid = sampler.sample(logits[-1])
            out.append(tid)
            ids = [tid]
    return out


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def ring(tiny_checkpoint, tmp_path, monkeypatch):
    """`async with ring() as (head, workers, reference, port)`. The suite has no
    async fixtures (see `tests/conftest.py`), so the fixture hands out the manager."""
    return lambda: _ring(tiny_checkpoint, tmp_path, monkeypatch)


@contextlib.asynccontextmanager
async def _ring(tiny_checkpoint, tmp_path, monkeypatch):
    directory, _, whole = tiny_checkpoint
    monkeypatch.setattr(serve, "JOIN_QUIET_S", 0.3)
    monkeypatch.setattr(serve, "load_tokenizer", lambda *a, **k: FakeTokenizer())
    monkeypatch.setattr(probe, "memory_total_free", lambda backend: (64 * GB, 60 * GB))
    dtype = probe.probe_capabilities("cpu", tmp_path).compute_dtype
    layer_bytes = serve.read_metadata(str(directory)).planner_model("bf16").weight_bytes_per_layer
    budget = int(3.5 * layer_bytes / 0.8) * serve.DTYPE_BYTES[dtype] // 2

    port = free_port()
    options = serve.ServeOptions(
        model=str(directory),
        quant="bf16",
        ctx=64,
        control_port=0,
        no_local_worker=True,
        min_workers=2,
        port=port,
        http=True,
        dashboard=False,
    )
    head = serve.Head(options, echo=lambda line: None)
    tasks = [asyncio.create_task(head.run())]
    await until(lambda: head.control_port != 0, 10)
    workers = [
        WorkerDaemon(
            WorkerConfig(
                head=f"127.0.0.1:{head.control_port}",
                name=name,
                device="cpu",
                max_mem_bytes=budget,
                cache_dir=tmp_path,
                wire_dtype=dtype,  # exact: a bf16 wire would round fp32 activations
            )
        )
        for name in ("a", "b")
    ]
    tasks += [asyncio.create_task(w.run()) for w in workers]
    await until(lambda: head.state == "ready")
    reference = copy.deepcopy(whole).to(dtype={"fp32": torch.float32}.get(dtype, torch.bfloat16))
    try:
        yield head, workers, reference, port
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def collect(driver, request) -> tuple[list[int], str, str | None]:
    await driver.submit(request)
    text, finish = "", None
    async for chunk in driver.stream(request):
        text += chunk.text
        finish = chunk.finish_reason
    return request.generated, text, finish


async def test_the_ring_matches_the_whole_model(ring) -> None:
    async with ring() as (head, workers, reference, _):
        driver = head.driver
        assert driver is not None
        for prompt in ([1, 5, 9, 2], [3]):
            request = driver.build_request(None, "".join(f"<{i}>" for i in prompt), 12, SAMPLING)
            ids, text, finish = await collect(driver, request)
            assert ids == greedy(reference, prompt, 12), f"prompt {prompt}"
            assert text == "".join(f"<{i}>" for i in ids)
            assert finish == "length"

        # `release` went around the ring: no node holds KV, the head holds no charge.
        await until(lambda: not driver.active and all(w.engine.active_reqs == 0 for w in workers))
        assert all(used == 0 for used, _ in driver.admission.usage().values())


async def test_a_long_prompt_is_prefilled_in_chunks(ring, monkeypatch) -> None:
    from baton.worker import engine

    async with ring() as (head, _workers, reference, _port):
        monkeypatch.setattr(engine, "CHUNK_TOKENS", 5)  # 13 ids become three chunks
        prompt = [(7 * i + 3) % 500 for i in range(13)]
        request = head.driver.build_request(None, "".join(f"<{i}>" for i in prompt), 6, SAMPLING)
        ids, _, _ = await collect(head.driver, request)
        assert ids == greedy(reference, prompt, 6)


async def test_a_stop_id_ends_the_reply_and_is_not_text(ring) -> None:
    async with ring() as (head, _, reference, _):
        prompt = [1, 5, 9, 2]
        want = greedy(reference, prompt, 8)
        request = head.driver.build_request(
            None, "<1><5><9><2>", 8, SAMPLING, stop=[f"<{want[3]}>"]
        )
        request.stop_ids.add(want[3])
        ids, text, finish = await collect(head.driver, request)
        assert finish == "stop"
        assert ids == want[:4]
        assert text == "".join(f"<{i}>" for i in want[:3])


async def test_a_stop_string_aborts_the_ring(ring) -> None:
    async with ring() as (head, workers, reference, _):
        want = greedy(reference, [1, 5, 9, 2], 8)
        stop = f"<{want[2]}><{want[3]}>"  # two tokens: matched on the head
        request = head.driver.build_request(None, "<1><5><9><2>", 8, SAMPLING, [stop])
        _, text, finish = await collect(head.driver, request)
        assert finish == "stop"
        assert text == "".join(f"<{i}>" for i in want[:2])
        await until(
            lambda: not head.driver.active and all(w.engine.active_reqs == 0 for w in workers)
        )


async def test_http_chat_and_stream(ring) -> None:
    async with ring() as (head, _, reference, port):
        base = f"http://127.0.0.1:{port}"
        prompt = [len("hello"), len("hi")]  # the fake chat template
        want = "".join(f"<{i}>" for i in greedy(reference, prompt, 5, temperature=1.5))
        body = {
            "model": "x",
            "messages": [{"role": "user", "content": "hello"}, {"role": "user", "content": "hi"}],
            "max_tokens": 5,
            "temperature": 1.5,
            "seed": 3,
        }
        async with httpx.AsyncClient(timeout=30) as client:
            await until(lambda: head.state == "ready")
            health = await client.get(f"{base}/healthz")
            assert health.status_code == 200
            models = (await client.get(f"{base}/v1/models")).json()
            assert [m["id"] for m in models["data"]] == [head.options.model]

            whole = await client.post(f"{base}/v1/chat/completions", json=body)
            assert whole.status_code == 200, whole.text
            reply = whole.json()
            assert reply["choices"][0]["message"]["content"] == want
            assert reply["usage"] == {"prompt_tokens": 2, "completion_tokens": 5, "total_tokens": 7}

            text, finish, usage = "", None, None
            async with client.stream(
                "POST", f"{base}/v1/chat/completions", json=body | {"stream": True}
            ) as r:
                assert r.headers["content-type"].startswith("text/event-stream")
                lines = [line async for line in r.aiter_lines() if line]
            assert lines[-1] == "data: [DONE]"
            for line in lines[:-1]:
                chunk = json.loads(line.removeprefix("data: "))
                choice = chunk["choices"][0]
                text += choice["delta"].get("content", "")
                finish = choice["finish_reason"] or finish
                usage = chunk.get("usage", usage)
            assert text == want and finish == "length"
            assert usage["completion_tokens"] == 5

            await until(lambda: not head.driver.active)  # the last `release_ack`
            snap = (await client.get(f"{base}/cluster")).json()
            assert snap["state"] == "READY"
            assert [n["role"] for n in snap["nodes"]] == ["N1", "N2"]
            assert snap["plan"]["rows"][0]["layers"] == [0, 1]


async def test_a_lost_worker_fails_the_request_and_the_next_one_says_not_ready(ring) -> None:
    async with ring() as (head, workers, _, port):
        driver = head.driver
        request = driver.build_request(None, "<1><5>", 40, SAMPLING)
        await driver.submit(request)
        stream = driver.stream(request)
        await stream.__anext__()
        # Kill the last node mid-reply.
        nk = next(w for w in workers if w.engine is not None and w.engine.is_last)
        await nk.stop()
        with pytest.raises(Exception) as err:
            async with asyncio.timeout(15):
                async for _ in stream:
                    pass
        assert getattr(err.value, "code", None) == "worker_lost"
        await until(lambda: head.state != "ready", 15)
        async with httpx.AsyncClient() as client:
            r = await client.post(
                f"http://127.0.0.1:{port}/v1/completions",
                json={"model": "x", "prompt": "<1>", "max_tokens": 2},
            )
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "not_ready"


async def test_a_client_that_leaves_aborts_the_ring(ring) -> None:
    async with ring() as (head, workers, _, port):
        body = {
            "model": "x",
            "prompt": "<1><5>",
            "max_tokens": 50,
            "temperature": 40,
            "stream": True,
        }
        async with (
            httpx.AsyncClient(timeout=30) as client,
            client.stream("POST", f"http://127.0.0.1:{port}/v1/completions", json=body) as r,
        ):
            async for line in r.aiter_lines():
                if line.startswith("data: {"):
                    break  # the first chunk, then the client goes away
        await until(
            lambda: not head.driver.active and all(w.engine.active_reqs == 0 for w in workers), 15
        )
        assert head.state == "ready"
