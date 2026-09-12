import { ms } from "../format";
import { SAMPLE_CHAT } from "../sample";

/**
 * Minimal chat against the local /v1/chat/completions, with a per-token timing
 * strip under each reply (PRD 14.2). The shell renders a sample transcript; M3
 * wires the streaming request.
 */
export function Chat() {
  return (
    <section className="view chat">
      <div className="transcript">
        {SAMPLE_CHAT.map((m, i) => (
          <div className={`msg ${m.role}`} key={i}>
            <div className="msg-role muted">{m.role}</div>
            <div className="msg-body">{m.content}</div>
            {"timings" in m && m.timings && (
              <div className="timing muted">
                TTFT {ms(m.timings.ttft_ms)} &middot; {m.timings.decode_tok_s.toFixed(2)} tok/s
                &middot; {m.timings.tokens} tokens
              </div>
            )}
          </div>
        ))}
      </div>

      <form className="composer" onSubmit={(e) => e.preventDefault()}>
        <input type="text" placeholder="Not wired yet — M3" disabled />
        <button type="submit" disabled>
          Send
        </button>
      </form>
    </section>
  );
}
