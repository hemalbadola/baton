// The one schema GET /cluster and every /ws message share (PRD 14.3).
// The dashboard does no math beyond formatting.

export type ClusterState =
  | "INIT"
  | "PLANNING"
  | "LOADING"
  | "READY"
  | "SERVING"
  | "DEGRADED";

export type NodeState = "joining" | "loading" | "loaded" | "standby" | "lost";

export interface ModelInfo {
  id: string;
  quant: "none" | "int8" | "int4";
  ctx: number;
  n_layers: number;
}

export interface NodeMemory {
  total: number;
  weights: number;
  kv_used: number;
  kv_budget: number;
}

export interface NodeSnapshot {
  name: string;
  role: string;
  backend: "cuda" | "mps" | "cpu";
  link: "wired" | "wifi";
  layers: [number, number];
  mem: NodeMemory;
  stage_ms: { p50: number; p95: number };
  queue_depth: number;
  state: NodeState;
  /** Extension: the last decode steps on this node, oldest first. */
  compute_ms?: number[];
}

export interface LiveStats {
  tok_s: number;
  ttft_ms: number;
  active: number;
  queued: number;
}

export interface ClusterEvent {
  t: number;
  kind: "join" | "loss" | "replan" | "error";
  msg: string;
}

/**
 * The Plan view needs the planner table (PRD 9.9) and its inputs. PRD 14.3 does
 * not list these fields, so they are an agreed extension of the same snapshot.
 * See the note in the surface lane's memory.md.
 */
export interface PlanRow {
  role: string;
  name: string;
  layers: [number, number];
  weights: number;
  kv_budget: number;
  predicted_ms: number;
  measured_ms: number | null;
}

export interface PlanInfo {
  objective: "latency" | "balance";
  kv_fraction: number;
  guaranteed_ctx: number;
  predicted_ttft_ms: number;
  predicted_tok_s: number;
  rows: PlanRow[];
}

export interface ClusterSnapshot {
  state: ClusterState;
  model: ModelInfo | null;
  plan_rev: number;
  nodes: NodeSnapshot[];
  /** len(nodes) + 1: head to N1, each hop, Nk back to head. */
  hops_ms: number[];
  live: LiveStats;
  events: ClusterEvent[];
  /** Extension, see PlanInfo. Absent until the head sends it. */
  plan?: PlanInfo | null;
}
