import type { ClusterSnapshot } from "../types";
import { gb, ms, pct } from "../format";

/** The ring as a horizontal chain of device cards in layer order (PRD 14.2). */
export function Topology({ snap }: { snap: ClusterSnapshot }) {
  return (
    <section className="view">
      <div className="chain">
        <div className="card head">
          <div className="card-name">head</div>
          <div className="muted">:7700 http</div>
          <div className="muted">:7711 control</div>
          <div className="muted">plan rev {snap.plan_rev}</div>
        </div>

        {snap.nodes.map((n, i) => (
          <div className="chain-cell" key={n.name}>
            <Hop ms={snap.hops_ms[i]} />
            <div className={`card node ${n.state}`}>
              <div className="card-head">
                <span className="card-name">{n.name}</span>
                <span className="tag">{n.role}</span>
              </div>
              <div className="row">
                <span className="tag">{n.backend}</span>
                <span className="tag">{n.link === "wired" ? "wired" : "Wi-Fi"}</span>
              </div>
              <div className="muted">
                layers {n.layers[0]}&ndash;{n.layers[1]}
              </div>
              <MemoryBar node={n} />
              <div className="muted">
                stage {ms(n.stage_ms.p50)} p50 &middot; {ms(n.stage_ms.p95)} p95
              </div>
              <div className="muted">queue {n.queue_depth}</div>
            </div>
          </div>
        ))}

        <Hop ms={snap.hops_ms[snap.nodes.length]} />
      </div>
    </section>
  );
}

function Hop({ ms: hop }: { ms: number | undefined }) {
  return (
    <div className="hop">
      <span className="arrow">&rarr;</span>
      <span className="hop-ms">{hop === undefined ? "—" : `${hop.toFixed(1)} ms`}</span>
    </div>
  );
}

function MemoryBar({ node }: { node: ClusterSnapshot["nodes"][number] }) {
  const weights = pct(node.mem.weights, node.mem.total);
  const kv = pct(node.mem.kv_used, node.mem.total);
  return (
    <div className="mem">
      <div className="mem-bar" title={`${gb(node.mem.total)} total`}>
        <span className="seg weights" style={{ width: `${weights}%` }} />
        <span className="seg kv" style={{ width: `${kv}%` }} />
      </div>
      <div className="muted">
        {gb(node.mem.weights)} weights &middot; {gb(node.mem.kv_used)} of{" "}
        {gb(node.mem.kv_budget)} KV &middot; {gb(node.mem.total)} total
      </div>
    </div>
  );
}
