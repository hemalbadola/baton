// Hardcoded sample data for the static shell. M3 replaces this with the live
// /cluster snapshot and the /ws feed. Shape is PRD 14.3, verbatim.
import type { ClusterSnapshot } from "./types";

export const SAMPLE: ClusterSnapshot = {
  state: "SERVING",
  model: {
    id: "meta-llama/Llama-3.1-70B-Instruct",
    quant: "int4",
    ctx: 8192,
    n_layers: 80,
  },
  plan_rev: 3,
  nodes: [
    {
      name: "mac",
      role: "N1",
      backend: "mps",
      link: "wifi",
      layers: [0, 25],
      mem: { total: 13.0e9, weights: 11.4e9, kv_used: 4.0e8, kv_budget: 1.3e9 },
      stage_ms: { p50: 74.1, p95: 91.0 },
      queue_depth: 0,
      state: "loaded",
    },
    {
      name: "rig",
      role: "N2",
      backend: "cuda",
      link: "wired",
      layers: [26, 51],
      mem: { total: 24.0e9, weights: 11.6e9, kv_used: 5.2e8, kv_budget: 2.4e9 },
      stage_ms: { p50: 41.7, p95: 55.3 },
      queue_depth: 0,
      state: "loaded",
    },
    {
      name: "thinkpad",
      role: "N3",
      backend: "cpu",
      link: "wifi",
      layers: [52, 67],
      mem: { total: 16.0e9, weights: 7.1e9, kv_used: 2.4e8, kv_budget: 1.1e9 },
      stage_ms: { p50: 188.4, p95: 233.9 },
      queue_depth: 1,
      state: "loaded",
    },
    {
      name: "spare",
      role: "N4",
      backend: "cpu",
      link: "wifi",
      layers: [68, 79],
      mem: { total: 8.0e9, weights: 5.3e9, kv_used: 1.1e8, kv_budget: 0.7e9 },
      stage_ms: { p50: 142.2, p95: 190.6 },
      queue_depth: 0,
      state: "loaded",
    },
  ],
  hops_ms: [2.9, 0.4, 3.1, 0.5, 2.7],
  live: { tok_s: 1.68, ttft_ms: 6120, active: 1, queued: 0 },
  plan: {
    objective: "latency",
    kv_fraction: 0.15,
    guaranteed_ctx: 8192,
    predicted_ttft_ms: 5880,
    predicted_tok_s: 1.74,
    rows: [
      { role: "N1", name: "mac", layers: [0, 25], weights: 11.4e9, kv_budget: 1.3e9,
        predicted_ms: 70.0, measured_ms: 74.1 },
      { role: "N2", name: "rig", layers: [26, 51], weights: 11.6e9, kv_budget: 2.4e9,
        predicted_ms: 44.0, measured_ms: 41.7 },
      { role: "N3", name: "thinkpad", layers: [52, 67], weights: 7.1e9, kv_budget: 1.1e9,
        predicted_ms: 176.0, measured_ms: 188.4 },
      { role: "N4", name: "spare", layers: [68, 79], weights: 5.3e9, kv_budget: 0.7e9,
        predicted_ms: 150.0, measured_ms: 142.2 },
    ],
  },
  events: [
    { t: 1725700000.1, kind: "join", msg: "worker 'mac' joined (mps, 13.0 GB)" },
    { t: 1725700004.4, kind: "join", msg: "worker 'rig' joined (cuda, 24.0 GB)" },
    { t: 1725700011.7, kind: "join", msg: "worker 'thinkpad' joined (cpu, 16.0 GB)" },
    { t: 1725700019.2, kind: "replan", msg: "plan_rev 2 applied, 3 nodes" },
    { t: 1725700188.0, kind: "loss", msg: "worker 'old' lost" },
    { t: 1725700188.9, kind: "replan", msg: "plan_rev 3 applied, 4 nodes" },
    { t: 1725700301.3, kind: "error", msg: "request r-0f21 failed: queue_timeout" },
  ],
};

// Placeholder rows for the Live view sparkline. Per-node compute ms, newest last.
export const SAMPLE_COMPUTE_MS: Record<string, number[]> = {
  mac: [72, 75, 71, 78, 74, 73, 76, 74, 72, 75, 79, 74],
  rig: [40, 43, 41, 44, 42, 41, 40, 43, 45, 41, 42, 42],
  thinkpad: [180, 195, 188, 201, 186, 190, 188, 193, 205, 187, 189, 188],
  spare: [140, 149, 138, 145, 142, 151, 140, 144, 139, 147, 142, 143],
};

// Placeholder chat transcript. The Chat view posts to /v1 in M3.
export const SAMPLE_CHAT = [
  { role: "user" as const, content: "Explain pipeline parallelism in one sentence." },
  {
    role: "assistant" as const,
    content:
      "Each device holds a contiguous slice of the layers and passes activations " +
      "to the next device, so one model runs across several machines.",
    timings: { ttft_ms: 6120, decode_tok_s: 1.68, tokens: 31 },
  },
];
