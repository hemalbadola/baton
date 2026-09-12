import type { ClusterSnapshot } from "../types";
import { clock } from "../format";

/** Last 200 events: joins, losses, re-plans, errors (PRD 14.2). */
export function Log({ snap }: { snap: ClusterSnapshot }) {
  const events = [...snap.events].reverse().slice(0, 200);
  return (
    <section className="view">
      <table className="log">
        <thead>
          <tr>
            <th>time</th>
            <th>kind</th>
            <th>message</th>
          </tr>
        </thead>
        <tbody>
          {events.map((e, i) => (
            <tr key={`${e.t}-${i}`}>
              <td className="muted">{clock(e.t)}</td>
              <td>
                <span className={`tag kind-${e.kind}`}>{e.kind}</span>
              </td>
              <td>{e.msg}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {events.length === 0 && <p className="muted">No events yet.</p>}
    </section>
  );
}
