import type { ClusterSnapshot } from "../types";
import { ms } from "../format";

/** tok/s, TTFT, active requests, queue depth, per-node sparkline (PRD 14.2). */
export function Live({ snap }: { snap: ClusterSnapshot }) {
  return (
    <section className="view">
      <div className="stats">
        <Stat label="tok/s (last 5 s)" value={snap.live.tok_s.toFixed(2)} />
        <Stat label="TTFT, last request" value={ms(snap.live.ttft_ms)} />
        <Stat label="active requests" value={String(snap.live.active)} />
        <Stat label="queue depth" value={String(snap.live.queued)} />
      </div>

      <h2>Per-node compute</h2>
      <table>
        <thead>
          <tr>
            <th>node</th>
            <th>backend</th>
            <th>compute ms</th>
            <th>p50</th>
          </tr>
        </thead>
        <tbody>
          {snap.nodes.map((n) => (
            <tr key={n.name}>
              <td>{n.name}</td>
              <td className="muted">{n.backend}</td>
              <td>
                <Sparkline values={n.compute_ms ?? []} />
              </td>
              <td className="num">{ms(n.stage_ms.p50)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat">
      <div className="stat-value">{value}</div>
      <div className="muted">{label}</div>
    </div>
  );
}

function Sparkline({ values }: { values: number[] }) {
  if (values.length === 0) return <span className="muted">—</span>;
  const max = Math.max(...values);
  const w = 120;
  const h = 22;
  const step = w / Math.max(1, values.length - 1);
  const d = values
    .map((v, i) => `${i === 0 ? "M" : "L"}${(i * step).toFixed(1)},${(h - (v / max) * h).toFixed(1)}`)
    .join(" ");
  return (
    <svg className="spark" width={w} height={h} viewBox={`0 0 ${w} ${h}`} aria-hidden="true">
      <path d={d} fill="none" />
    </svg>
  );
}
