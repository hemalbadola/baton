# Baton

LAN-distributed LLM inference. Cut one large model into shards, put one shard on
each laptop, pass activations device to device like a relay baton.

The full build spec is `../PRD.md`. Read section 0 first. The module layout in
this repository follows PRD section 15.5, file for file.

Current milestone: **M0 Spike** (PRD 19). Exit test:

```
pytest tests/numerics
```

plus 64 greedy tokens of Llama-3.2-1B identical across one process and two
processes, and a measured per-frame overhead under 0.1 ms.
