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
| BAT-36 | Task | Publish: commit the work, push, make the repository public | TODO |
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
- **Owner:** Hemal. Commits are on hold for the split between teammates.

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
