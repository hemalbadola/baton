# Baton tickets

This file is the ticket system for the project. Every change starts as a ticket here.

**Rules**

1. No code change without a ticket ID.
2. A ticket is `DONE` only when its "Done when" check passes.
3. Nothing is committed yet. The "Files" line of each ticket shows what to stage, so the
   work can be split by teammate later.

**Status values:** `TODO`, `DOING`, `DONE`, `BACKLOG` (known, not in this phase).

## Board

| ID | Type | Title | Status |
|---|---|---|---|
| BAT-1 | Story | Detect device specs and memory budget | DONE |
| BAT-2 | Story | Find the head on the network with mDNS | DONE |
| BAT-3 | Story | Worker joins the head and stays alive | DONE |
| BAT-4 | Story | Benchmark each worker for the planner | DONE |
| BAT-5 | Story | Plan the split and load the model shards | DONE |
| BAT-6 | Story | `baton serve` and `baton worker` commands | DONE |
| BAT-7 | Task | End-to-end loopback test: find, join, load, READY | DONE |
| BAT-8 | Bug | PRD 6.3 and PRD 18.1 disagree on `--max-mem` | DONE |
| BAT-9 | Bug | `load` message cannot carry the model spec or the tensor index | DONE |
| BAT-14 | Bug | A remote worker gets `127.0.0.1` as its next ring node | DONE |
| BAT-15 | Bug | A worker that joins during bench or load is missed, head stays idle | DONE |
| BAT-16 | Bug | An error from outside the ring fails a load that was good | DONE |
| BAT-17 | Bug | A standby worker with an old layer range can take READY down | DONE |
| BAT-18 | Bug | Two workers with one name share one registry entry | DONE |
| BAT-19 | Bug | The head waits for all nodes after one node is lost in a load | DONE |
| BAT-20 | Bug | SIGTERM to the head leaves the local worker running | DONE |
| BAT-21 | Bug | `NaN` or `Infinity` in a worker frame can stop the head | DONE |
| BAT-22 | Bug | A shard load reads the full layer range in one fetch | DONE |
| BAT-23 | Bug | The memory budget is 0 on a real 8 GB Mac | DONE |
| BAT-24 | Bug | A bad handshake stops the worker | DONE |
| BAT-25 | Bug | A load in flight answers on the next connection | DONE |
| BAT-26 | Bug | `unload` can run before the `load` that came first | DONE |
| BAT-27 | Bug | A load cannot be stopped | DONE |
| BAT-28 | Bug | On Linux a killed worker leaves the sleep lock | DONE |
| BAT-29 | Bug | The worker cannot detect a silent head | BACKLOG |
| BAT-30 | Story | Web page: pick OS, role and model, copy one command | DONE |
| BAT-31 | Story | `install.sh`: one line installs and starts Baton on macOS and Linux | DONE |
| BAT-32 | Bug | The plan gives N1 a shard that does not fit with the embedding table | DONE |
| BAT-33 | Task | Run `install.ps1` on a Windows laptop | TODO |
| BAT-34 | Task | Head prints its address and says when it waits | DONE |
| BAT-35 | Task | Check the model fetch against the real Hugging Face | DONE |
| BAT-36 | Task | Publish: commit the work, push, make the repository public | DONE |
| BAT-37 | Task | Host the web page | DONE |
| BAT-38 | Story | Generate tokens around the ring | DONE |
| BAT-39 | Story | OpenAI API, streaming, and the live dashboard | DONE |
| BAT-40 | Bug | Qwen2 attention biases are dropped, the text is garbage | DONE |
| BAT-41 | Story | `baton app`: one page per laptop, find nearby laptops, host or join | DONE |
| BAT-42 | Task | Publish `baton-cluster` on GitHub, PyPI and Homebrew | DOING |
| BAT-43 | Story | The head serves the model files, workers keep them on disk | DONE |
| BAT-44 | Story | Update Baton from the page, show what every laptop does | DONE |
| BAT-10 | Story | Run int8 and int4 weights in the decoder | BACKLOG |
| BAT-11 | Story | Use the shard cache on load, memmap the embedding table | BACKLOG |
| BAT-12 | Story | Measure round-trip time between workers (`ping_peer`) | BACKLOG |
| BAT-13 | Story | Re-plan when a worker is lost: reuse and resume | BACKLOG |

## Tickets

### BAT-1 — Detect device specs and memory budget
- **Why:** The head cannot plan without the memory and the backend of each device.
- **Scope:** `pick_device`, `memory_total_free`, `int4_fast_path`, `detect_link`,
  `probe_capabilities`, `memory_budget` in `baton/worker/probe.py`.
- **Done when:** `pytest tests/unit/test_probe.py` passes with no skipped test.
- **Files:** `baton/worker/probe.py`, `tests/unit/test_probe.py`
- **Result:** DONE. 15 tests pass, none skipped. On the development Mac the probe reports
  `mps`, `bf16`, link `wifi`. `detect_link` on Linux and Windows is written from the
  documentation and is not run on those systems yet.

### BAT-2 — Find the head on the network with mDNS
- **Why:** A worker must find the head with no IP address typed by the user. Wi-Fi and
  wired Ethernet must both work.
- **Scope:** `Head.advertise` in `baton/head/serve.py`, `discover_head` in
  `baton/worker/daemon.py`.
- **Done when:** a test starts a head, and `discover_head("auto")` returns its address.
- **Files:** `baton/head/serve.py`, `baton/worker/daemon.py`
- **Result:** DONE. `discover_head("auto")` returned the LAN address of a running head
  (`172.20.10.4`) in a live run. The worker dials each advertised address and keeps the
  first that answers, so a head with Wi-Fi, Ethernet and a VPN is found on the right one.

### BAT-3 — Worker joins the head and stays alive
- **Why:** Discovery gives an address only. The worker must connect, say `hello`, and send
  `health` every 2 s. The head must drop a worker that is silent for 6 s.
- **Scope:** control server and frame loop in `serve.py`, `connect_control`, `heartbeat`,
  reconnect rule (PRD 6.1) and sleep prevention in `daemon.py`.
- **Done when:** a joined worker shows in `Registry.roster()`, and a stopped worker goes to
  `lost`.
- **Files:** `baton/head/serve.py`, `baton/worker/daemon.py`, `baton/head/registry.py`
- **Result:** DONE. `test_cluster.py` covers join, loss on a closed connection, and a wrong
  token. `baton/worker/daemon.py` also keeps the laptop awake (PRD 6.1 step 7).

### BAT-4 — Benchmark each worker for the planner
- **Why:** The planner needs milliseconds per layer for each device (PRD 6.3, 9.2).
- **Scope:** `run_bench` in `probe.py`, `handle_bench` in `daemon.py`, `Head.benchmark`.
- **Done when:** after `bench`, each worker has `t_dec_ms > 0` in the registry.
- **Files:** `baton/worker/probe.py`, `baton/worker/daemon.py`, `baton/head/serve.py`
- **Result:** DONE. `TestBench` (3 tests) plus the end-to-end test. The bench runs a dense
  layer for each tier until BAT-10.

### BAT-5 — Plan the split and load the model shards
- **Why:** This is the product: each device loads only its own layers.
- **Scope:** `Head.fetch_metadata`, `make_plan`, `load_plan`; `handle_load`,
  `handle_unload` in `daemon.py`; `load_shard` in `baton/worker/engine.py`;
  `open_source`, `fetch_header` in `baton/model/safetensors_io.py`.
- **Done when:** every worker in the plan replies `loaded`, and the head state is `ready`.
- **Files:** `baton/head/serve.py`, `baton/worker/daemon.py`, `baton/worker/engine.py`,
  `baton/model/safetensors_io.py`, `baton/model/layers.py`
- **Result:** DONE. `test_load_shard.py` (4 tests): two shards give the same logits as the
  whole model, bit for bit. A shard larger than the budget is refused before allocation.

### BAT-6 — `baton serve` and `baton worker` commands
- **Why:** The two-laptop test uses these two commands.
- **Scope:** bodies of `serve` and `worker` in `baton/cli.py`.
- **Done when:** `baton worker --help` and `baton serve --help` work, and the commands call
  the daemon and the head.
- **Files:** `baton/cli.py`, `tests/unit/test_cli.py`
- **Result:** DONE. `tests/unit/test_cli.py`, 21 tests. New flag `--no-local-worker`.
  `--quant none` is necessary until BAT-10.

### BAT-7 — End-to-end loopback test
- **Why:** Proof by command, not by eye. PRD 18.1 says one machine is a valid test fleet.
- **Done when:** `pytest tests/integration/test_cluster.py` passes: one head and two
  workers on one machine, found by mDNS, joined, benchmarked, planned across both
  workers, loaded, head `ready`.
- **Files:** `tests/integration/test_cluster.py`
- **Result:** DONE. 2 tests, about 3 s. The test also stops one worker, checks that the head
  leaves `ready`, then adds a new worker by address and checks that the head plans again.

### BAT-8 — PRD 6.3 and PRD 18.1 disagree on `--max-mem`
- **Problem:** PRD 6.3 says `usable = min(free, max_mem) - os_reserve`. PRD 18.1 runs
  `--max-mem 1G` workers and expects about 1 GB of usable memory. With the PRD 6.3
  formula, 1 GB minus a 1.5 GB reserve is zero. The one-machine test fleet cannot work.
- **Decision:** `usable = min(free - os_reserve, max_mem)`. `--max-mem` is the amount the
  worker can use. The reserve protects the OS from the free figure only.
- **Files:** `baton/worker/probe.py`
- **Result:** DONE in `memory_budget`. BAT-23 changed the formula again: see that ticket.

### BAT-9 — `load` message cannot carry the model spec or the tensor index
- **Problem:** PRD 5.3 step 1 says the worker receives the `ModelSpec` and the index JSON
  from the head. In `messages.py`, `Load` has no `spec`, and `index` is typed as an
  integer ring position.
- **Decision:** add `spec`. Make `index` the tensor-name-to-file map. The worker reads each
  shard header itself with two small ranged reads, because all headers of a 70B model do
  not fit in the 1 MB frame meta limit.
- **Files:** `baton/common/messages.py`
- **Result:** DONE. `tests/unit/test_messages.py` updated.

### BAT-14 — A remote worker gets `127.0.0.1` as its next ring node
- **Source:** review of the head, severity critical.
- **Problem:** The local worker joins through `127.0.0.1` and reports that address. The head sent it to a
  worker on a different machine, which then dialled itself. The two-laptop ring did not form.
- **Fix:** `Head._ring_addr` gives the remote worker the head address that its control socket
  already uses.
- **Check:** `test_a_remote_worker_is_never_told_to_dial_loopback` in `tests/integration/test_head_faults.py`
- **Files:** `baton/head/serve.py`

### BAT-15 — A worker that joins during bench or load is missed, head stays idle
- **Source:** review of the head, severity high.
- **Problem:** `Head.run` read the join counter after the attempt. A join during the bench or the load was
  not seen, and the head waited for a join that was already in the past.
- **Fix:** Read the counter before the attempt.
- **Check:** `test_a_worker_that_joins_during_the_bench_is_not_missed` in `tests/integration/test_head_faults.py`
- **Files:** `baton/head/serve.py`

### BAT-16 — An error from outside the ring fails a load that was good
- **Source:** review of the head, severity medium.
- **Problem:** `load_plan` failed on each `error` frame, also from a worker that was not in the plan, and
  from a load of an old plan revision.
- **Fix:** Count only the errors of ring members. The worker puts `rev` in a load error and the
  head drops an old one.
- **Check:** `test_an_error_from_outside_the_ring_does_not_fail_the_load` in `tests/integration/test_head_faults.py`
- **Files:** `baton/head/serve.py`, `baton/worker/daemon.py`

### BAT-17 — A standby worker with an old layer range can take READY down
- **Source:** review of the head, severity medium.
- **Problem:** A worker that came back kept its old layer range. When it left again, the head thought that
  a ring node was lost and reloaded the full ring.
- **Fix:** `_lose` uses the worker state, not the range. `load_plan` releases each node that the
  new plan does not use.
- **Check:** `test_a_standby_worker_with_a_stale_range_cannot_take_ready_down` in `tests/integration/test_head_faults.py`
- **Files:** `baton/head/serve.py`

### BAT-18 — Two workers with one name share one registry entry
- **Source:** review of the head, severity medium.
- **Problem:** A second `hello` with the same name replaced a `standby` worker. The first socket stayed
  open and its frames changed the entry of the second worker.
- **Fix:** The head refuses a live name from a different address. For the same address it closes
  the old socket. A socket that does not own the entry stops.
- **Check:** `test_a_second_machine_with_the_same_name_is_refused` in `tests/integration/test_head_faults.py`
- **Files:** `baton/head/serve.py`

### BAT-19 — The head waits for all nodes after one node is lost in a load
- **Source:** review of the head, severity low.
- **Problem:** After one node was lost in a load, the head waited for the other nodes to download their
  full shards, then discarded them.
- **Fix:** The wait ends when a ring member has an error or is not `loading` or `loaded`.
- **Check:** `test_a_lost_ring_member_ends_the_load_at_once` in `tests/integration/test_head_faults.py`
- **Files:** `baton/head/serve.py`

### BAT-20 — SIGTERM to the head leaves the local worker running
- **Source:** review of the head, severity low.
- **Problem:** Only Ctrl-C ran `Head.close`. After `kill`, the local worker and its `caffeinate` process
  stayed alive.
- **Fix:** `serve()` cancels on SIGTERM. `close()` waits for the local worker.
- **Check:** `manual: start `baton serve`, send SIGTERM, list processes`
- **Files:** `baton/head/serve.py`

### BAT-21 — `NaN` or `Infinity` in a worker frame can stop the head
- **Source:** review of the head, severity low.
- **Problem:** The frame decoder accepts `NaN` and `Infinity`. A `bench_result` with `NaN` stopped the
  planner. A `health` with `Infinity` raised in the connection handler.
- **Fix:** `Registry.on_bench` accepts finite positive numbers only. The integer coercion and
  `_on_frame` handle `OverflowError`.
- **Check:** `test_non_finite_numbers_from_the_wire_are_dropped` in `tests/integration/test_head_faults.py`
- **Files:** `baton/head/serve.py`, `baton/head/registry.py`

### BAT-22 — A shard load reads the full layer range in one fetch
- **Source:** review of the worker, severity high.
- **Problem:** Tensors are back to back in a shard file, so the 1 MB gap rule merged all of them into one
  read. The worker held that read and the full stack together: about 2 times the shard.
- **Fix:** `load_shard` reads one tensor for each fetch (`max_gap=-1`).
- **Check:** `test_no_read_is_larger_than_one_tensor` in `tests/unit/test_load_shard.py`
- **Files:** `baton/worker/engine.py`

### BAT-23 — The memory budget is 0 on a real 8 GB Mac
- **Source:** review of the worker, severity high.
- **Problem:** The budget took the OS reserve off the free memory. Free memory already excludes the OS and
  the open apps. With 1.3 GB free the worker reported 0 bytes, and `--mem-budget` could not
  increase it. This changes the decision of BAT-8.
- **Fix:** The reserve is taken from the total. Usable is the free memory, or the `--mem-budget`
  value, up to `total - reserve`. The worker measures again at each `load`. Measured on the
  development Mac after the fix: 1.43 GB usable.
- **Check:** `test_an_8gb_mac_with_little_free_memory_still_has_a_budget` in `tests/unit/test_probe.py`
- **Files:** `baton/worker/probe.py`, `baton/worker/daemon.py`

### BAT-24 — A bad handshake stops the worker
- **Source:** review of the worker, severity high.
- **Problem:** A reset in `hello`, a late `welcome`, or a port that is not Baton raised out of `run`, and
  the worker process ended. PRD 8.6 says retry forever. A late `welcome` also dropped the
  shard 25 s early.
- **Fix:** `run` logs the error and tries again. It unloads only when the 30 s hold is over. A
  `name_in_use` refusal is tried again too, because the head clears a stale name in 6 s.
- **Check:** `test_a_bad_handshake_is_retried_not_fatal` in `tests/integration/test_worker_faults.py`
- **Files:** `baton/worker/daemon.py`

### BAT-25 — A load in flight answers on the next connection
- **Source:** review of the worker, severity medium.
- **Problem:** After a control drop, the old load task continued and sent `loaded` or `error` to the new
  session, which did not ask for it.
- **Fix:** Each load carries an epoch. A drop, a new `load`, an `unload` or a stop changes the
  epoch. A load with an old epoch frees its work and sends nothing. A load `error` has `rev`.
- **Check:** `test_unload_after_load_leaves_nothing_resident` (same mechanism)
- **Files:** `baton/worker/daemon.py`

### BAT-26 — `unload` can run before the `load` that came first
- **Source:** review of the worker, severity medium.
- **Problem:** `load` was a task and `unload` ran inline. When the two frames came in one read, `unload`
  ran first and the worker kept the shard.
- **Fix:** `unload` is a task too. Tasks start in order and the device lock is first-in first-out.
- **Check:** `test_unload_after_load_leaves_nothing_resident` in `tests/integration/test_worker_faults.py`
- **Files:** `baton/worker/daemon.py`

### BAT-27 — A load cannot be stopped
- **Source:** review of the worker, severity medium.
- **Problem:** Ctrl-C during a download did not end the process until the download was complete. An
  `unload` waited for the full load.
- **Fix:** The loader thread checks the epoch after each tensor and stops. With BAT-22 that is at
  most one tensor of work.
- **Check:** `test_unload_after_load_leaves_nothing_resident`
- **Files:** `baton/worker/daemon.py`

### BAT-28 — On Linux a killed worker leaves the sleep lock
- **Source:** review of the worker, severity low.
- **Problem:** `systemd-inhibit ... sleep infinity` had no link to the worker process. The worker had no
  SIGTERM handler, and the head stops its local worker with SIGTERM.
- **Fix:** The lock process is `tail --pid=<worker>`, which ends with the worker. `run_worker`
  cancels on SIGTERM. The Linux part is not run on Linux yet.
- **Check:** manual on macOS: `baton serve`, SIGTERM, no worker process stays
- **Files:** `baton/worker/daemon.py`

### BAT-29 — The worker cannot detect a silent head (BACKLOG)
- **Source:** review of the worker, severity low.
- **Problem:** When the head laptop sleeps or loses power, no FIN arrives. The worker sees it only after
  the kernel stops retransmission, which can be 15 minutes. PRD 6.1 says 30 s.
- **Fix:** Not done. It needs a reply to `health` from the head, which is a new message that PRD 8.3
  does not have, or TCP keepalive, which has different options on each OS. Decide first.
- **Check:** none
- **Files:** `baton/worker/daemon.py`, `baton/head/serve.py`

### BAT-30 — Web page: pick OS, role and model, copy one command
- **Why:** A person with no Baton must get it with one paste.
- **Scope:** the approved wireframe, shipped as a static page with no build step. Fonts are
  self-hosted. Hero additions: entry motion, a relay animation, a flash when the command
  changes. All motion stops under `prefers-reduced-motion`.
- **Result:** DONE. Rendered in a desktop and a phone size. The 4 command forms (macOS or
  Windows, start or join) are checked in a browser. Not hosted yet.
- **Files:** `web/index.html`, `web/fonts/`

### BAT-31 — `install.sh`: one line installs and starts Baton on macOS and Linux
- **Why:** `baton` is not a command until something installs it.
- **Scope:** install `uv` if it is absent, install Baton as a `uv` tool, run `baton` with the
  arguments that follow. `BATON_SOURCE` selects a different source.
- **Result:** DONE on macOS. `sh install.sh serve --model <dir> --quant none`, with the source
  set to this working tree and a temporary tool directory, reached READY in 58 s. The shell
  profile was not changed. Not run on Linux.
- **Files:** `install.sh`

### BAT-32 — The plan gives N1 a shard that does not fit with the embedding table
- **Problem:** N1 holds the embedding table (501 MB for Llama 3.2 1B) and its layers. The
  planner counts the layers only. The worker then refused the shard and the head stayed idle.
- **Fix:** `Head.make_plan` charges the table to the device that the planner puts first, and
  plans again until the first device is one that was charged.
- **Check:** `test_the_plan_leaves_room_for_the_embedding_table_on_n1`. The test also shows
  that the planner alone goes over the budget.
- **Files:** `baton/head/serve.py`

### BAT-33 — Run `install.ps1` on a Windows laptop
- **Why:** The script is written from the `uv` documentation and was never run. Three points
  are not proven: the `scriptblock` form that passes arguments, the CUDA build of torch
  (`uv pip install --torch-backend auto`), and the firewall prompt.
- **Done when:** the Windows command from the web page reaches `joined` on a head.
- **Files:** `install.ps1`

### BAT-34 — Head prints its address and says when it waits
- **Why:** mDNS does not cross each Wi-Fi. A phone hotspot can drop it. The user then needs
  the address, and must know that the head waits.
- **Result:** DONE. `baton serve` prints `baton worker --head <ip>:<port>` at start, and
  `not ready. Waiting for another worker to join.` when a plan is not possible.
- **Files:** `baton/head/serve.py`

### BAT-35 — Check the model fetch against the real Hugging Face
- **Result:** DONE for metadata and one tensor. `read_metadata("unsloth/Llama-3.2-1B")` read
  16 808 bytes in 3.6 s: 16 layers, 146 tensors, all names of a two-node split are in the
  index. One ranged fetch returned `model.layers.0.input_layernorm.weight`, 4096 bytes.
  A full load of the 2.3 GB model from Hugging Face is NOT run yet.

### BAT-36 — Publish: commit the work, push, make the repository public
- **Why:** The command on the web page reads `install.sh` and the code from GitHub. GitHub
  has the code from before this work, and the repository is private. Until this ticket is
  done, the command from the page cannot work on a different laptop.
- **Result:** DONE. Ten commits, grouped by area. The repository is public at
  `https://github.com/hemalbadola/baton`. Each commit has one author, the owner. The command
  from the live page, run against the public repository in a temporary tool directory,
  reached READY in 24 s.

### BAT-37 — Host the web page
- **Result:** DONE. `web/` is on Vercel (project `baton`) at `https://baton-plum.vercel.app`.
  The page and the two fonts return 200. To publish a change: `cd web && vercel deploy --prod`.
- **Files:** `web/.gitignore`

### BAT-38 — Generate tokens around the ring
- **Why:** Until now the cluster formed and loaded, but no text came out.
- **Scope:** `ForwardEngine`, `KVPool` in `baton/worker/engine.py`; the frame pump in
  `baton/worker/daemon.py`; `Driver`, `Admission` in `baton/head/driver.py`;
  `IncrementalDetokenizer` in `baton/head/detok.py`; ring closing in `Head.load_plan`.
- **Decisions:** Nk dials N1 so `next` and `release` ride the ring (PRD 10.2). `prompt`,
  `abort`, `token`, `release_ack` and request `error` use the control socket, not a second
  data listener: it exists and is authenticated. Text that could start a stop string is held
  back until the next token (PRD 7.4 leaks the first half).
- **Done when:** `pytest tests/integration/test_generate.py` passes: a head and two workers
  produce the same seeded tokens as the whole model in one process, with a 13-token prompt in
  three chunks, a stop id, a stop string, a client that leaves, and a worker lost mid-reply.
- **Files:** `baton/worker/engine.py`, `baton/worker/daemon.py`, `baton/head/driver.py`,
  `baton/head/detok.py`, `baton/head/serve.py`, `tests/integration/test_generate.py`,
  `tests/unit/test_engine.py`, `tests/unit/test_kv_pool.py`, `tests/unit/test_detok.py`,
  `tests/unit/test_driver.py`
- **Result:** DONE. Real run: Qwen2.5-0.5B-Instruct on two workers (`mps`, bf16), 26 tok/s,
  TTFT 0.6 to 0.8 s. A worker killed mid-reply gives `worker_lost` in the stream, then a new
  worker makes plan 2 and the next reply works.

### BAT-39 — OpenAI API, streaming, and the live dashboard
- **Scope:** `baton/head/api.py` (`/v1/chat/completions`, `/v1/completions`, `/v1/models`,
  `/healthz`, `/cluster`, `/ws`, SSE), `Head.start_http`, `dashboard/src/App.tsx`,
  `dashboard/src/views/Chat.tsx`.
- **Decisions:** `websockets` joins the `head` extra, because uvicorn has no WebSocket server
  without it. The CLI default `--quant` is `none` until BAT-10. The Live view draws the last
  decode steps that each worker reports in `health.compute_ms`.
- **Done when:** `curl` and the dashboard chat get a streamed answer from a ready cluster;
  `/cluster` shows the plan; `/healthz` is 503 when a worker is lost.
- **Result:** DONE. `dashboard/dist` is served at `/` by `baton serve` when it is built
  (`cd dashboard && npm install && npm run build`).

### BAT-40 — Qwen2 attention biases are dropped, the text is garbage
- **Problem:** A real Qwen2.5-0.5B gave `HandlerContextYM fontStyle ...` while the ring and
  every synthetic test passed.
- **Cause:** Qwen2 configs have no `attention_bias` key, yet q, k and v have biases.
  `ModelSpec.from_config` read the key with default `False`. Layer 0 differed from Hugging Face
  by 1.2, layer 2 by 480.
- **Fix:** `attn_bias` defaults to `model_type == "qwen2"`. Test:
  `test_qwen2_config_without_attention_bias_key_has_biases`.
- **Result:** DONE. All 24 layers now equal Hugging Face, and 12 greedy tokens match on `cpu`
  fp32 and `mps` bf16.
- **Files:** `baton/model/spec.py`, `tests/numerics/test_spec.py`

### BAT-41 — `baton app`: find nearby laptops, host or join, with no other command
- **Why:** The demo must not need a terminal on every laptop. Two commands (`serve`, `worker`)
  and flags were too much.
- **Scope:** `baton/agent.py` (agent, mDNS `_baton-node._tcp`, invite flow, subprocess control,
  proxy to the head), `baton/agent_ui.html` (one file, no build step), `baton app` in
  `baton/cli.py`, install page and `install.sh` hand out `app`.
- **Flow:** open the page on every laptop. One person picks a model and creates a room, then
  invites nearby laptops. Each invited person clicks Accept. The host clicks Start. The agents
  run `baton serve` and `baton worker` themselves. Chat works on every laptop.
- **Safety:** an agent never runs a received command. A peer can send an invite (shown to the
  user), an accept (only for an invite this laptop sent), and a join (only for an invite the user
  accepted, and only from the laptop that sent it).
- **Decisions:** all head dependencies (fastapi, uvicorn, transformers) are now base
  dependencies, because any laptop can host. The page is plain HTML so a pip install carries it.
- **Done when:** `pytest tests/integration/test_agent.py` passes, and a real run of two agents
  reaches READY and answers a chat through the guest's page.
- **Result:** DONE. Real run: agents `alpha` and `beta`, invite, accept, start, READY, answer in
  190 ms to first token. Not run on two physical laptops, or on Windows.
- **Files:** `baton/agent.py`, `baton/agent_ui.html`, `baton/cli.py`, `pyproject.toml`,
  `install.sh`, `web/index.html`, `tests/integration/test_agent.py`

### BAT-42 — Publish `baton-cluster` on GitHub, PyPI and Homebrew
- **Done:** package renamed `baton-cluster` 0.1.0 (the name `baton` is taken on PyPI, the
  command stays `baton`). Pushed to `main`, tag `v0.1.0`. The one-line install from GitHub
  was run into an isolated tool directory: it installs in 3.5 minutes and `baton app` exists.
  Tap `hemalbadola/homebrew-baton` is public, `brew style` passes.
- **Open:** PyPI upload needs the owner's token: `uv publish /tmp/dist/*` after `uv build`.
  The formula was not installed on the development Mac: its Command Line Tools are out of date.
  Run `brew install hemalbadola/baton/baton` on a Mac with current tools. No license file exists.
- **Files:** `pyproject.toml`, `README.md`, `install.sh`, `install.ps1`

### BAT-43 — The head serves the model files, workers keep them on disk (and BAT-11)
- **Problem:** a guest laptop failed to download from Hugging Face (`WinError 10054`, five
  retries), and one request per tensor made the first load take five minutes.
- **Fix:** only the head downloads, once, in a resumable stream (988 MB at about 9 MB/s). Workers
  fetch their layers from `GET /weights/<file>` on the head. Each worker keeps every tensor it
  fetched under the cache root, so the second start reads the disk.
- **Files:** `baton/head/weights.py`, `baton/head/api.py`, `baton/head/serve.py`,
  `baton/model/cache.py`, `baton/worker/engine.py`, `baton/worker/daemon.py`

### BAT-44 — Update Baton from the page, show what every laptop does
- **Page:** a phase line and steps (join, measure, download, load, ready), a download bar, and one
  card per laptop: device, layers on a strip, what it does now, load bar, memory, last step.
- **Update:** `baton app` installs a newer version at start (`--no-update` skips), and the page
  offers the same button. Only the `baton-cluster` package is replaced, so a CUDA torch stays.
  The truth is `version` in `pyproject.toml` on `main`: raise it for every change users must get.
- **Public page:** `https://baton-plum.vercel.app` asks `127.0.0.1:7800/api/hello` (one route, one
  origin) and shows "Baton is running here" and "a new version is out".
- **Files:** `baton/updater.py`, `baton/agent.py`, `baton/agent_ui.html`, `web/index.html`

### BAT-10 — Run int8 and int4 weights in the decoder (BACKLOG)
- `quant.py` can quantize, but `layers.py` runs dense weights only. Until this is done,
  `baton serve` accepts `--quant none` only and refuses the other tiers with a clear error.

### BAT-11 — Use the shard cache on load, memmap the embedding table (BACKLOG)
- `cache.py` is ready. The load path does not use it yet, so each start fetches again, and
  N1 holds the embedding table in memory. The plan now counts that table (BAT-32).

### BAT-12 — Measure round-trip time between workers (BACKLOG)
- The planner uses a default of 3 ms for each hop until `ping_peer` is built.

### BAT-13 — Re-plan when a worker is lost (BACKLOG)
- The basic form works: the head marks the worker `lost`, goes to `idle`, and plans again
  from zero when enough workers are present. Not done: reuse of layers that a node already
  holds (PRD 9.7), resume of a worker that reconnects with the same plan (PRD 11.3), and
  the request errors of PRD 11.2.
