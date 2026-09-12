import type { ClusterSnapshot } from "../types";
import { gb, ms } from "../format";

/** The planner table, predicted against measured, kv_fraction, context (14.2). */
export function Plan({ snap }: { snap: ClusterSnapshot }) {
  const plan = snap.plan;
  if (!plan) {
    return (
      <section className="view">
        <p className="muted">The head has not published a plan yet.</p>
      </section>
    );
  }
  const predictedSum = plan.rows.reduce((a, r) => a + r.predicted_ms, 0);
  const measuredSum = plan.rows.reduce((a, r) => a + (r.measured_ms ?? 0), 0);
  return (
    <section className="view">
      <div className="stats">
        <Field label="objective" value={plan.objective} />
        <Field label="kv_fraction in effect" value={plan.kv_fraction.toFixed(2)} />
        <Field label="guaranteed context" value={`${plan.guaranteed_ctx} tok`} />
        <Field label="plan rev" value={String(snap.plan_rev)} />
      </div>

      <table>
        <thead>
          <tr>
            <th>role</th>
            <th>node</th>
            <th>layers</th>
            <th className="num">weights</th>
            <th className="num">KV budget</th>
            <th className="num">predicted</th>
            <th className="num">measured</th>
            <th className="num">delta</th>
          </tr>
        </thead>
        <tbody>
          {plan.rows.map((r) => {
            const delta = r.measured_ms === null ? null : r.measured_ms - r.predicted_ms;
            return (
              <tr key={r.name}>
                <td>{r.role}</td>
                <td>{r.name}</td>
                <td className="muted">
                  {r.layers[0]}&ndash;{r.layers[1]}
                </td>
                <td className="num">{gb(r.weights)}</td>
                <td className="num">{gb(r.kv_budget)}</td>
                <td className="num">{ms(r.predicted_ms)}</td>
                <td className="num">{r.measured_ms === null ? "—" : ms(r.measured_ms)}</td>
                <td className={`num ${delta !== null && delta > 0 ? "over" : "under"}`}>
                  {delta === null ? "—" : `${delta > 0 ? "+" : ""}${delta.toFixed(1)} ms`}
                </td>
              </tr>
            );
          })}
        </tbody>
        <tfoot>
          <tr>
            <td colSpan={5}>sum of stage time</td>
            <td className="num">{ms(predictedSum)}</td>
            <td className="num">{ms(measuredSum)}</td>
            <td />
          </tr>
          <tr>
            <td colSpan={5}>predicted TTFT / tok·s⁻¹</td>
            <td className="num">{ms(plan.predicted_ttft_ms)}</td>
            <td className="num">{plan.predicted_tok_s.toFixed(2)}</td>
            <td />
          </tr>
        </tfoot>
      </table>
    </section>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat">
      <div className="stat-value">{value}</div>
      <div className="muted">{label}</div>
    </div>
  );
}
