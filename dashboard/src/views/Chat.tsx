import { useRef, useState } from "react";
import { ms } from "../format";

interface Msg {
  role: "user" | "assistant";
  content: string;
  timings?: { ttft_ms: number; decode_tok_s: number; tokens: number };
  error?: string;
}

/**
 * Chat against the head's own /v1/chat/completions with `stream: true`, and a
 * timing strip under each reply (PRD 14.2). The last SSE chunk carries
 * `usage` and `x_baton`.
 */
export function Chat() {
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const history = useRef<Msg[]>([]);

  const patchLast = (f: (m: Msg) => Msg) =>
    setMsgs((all) => [...all.slice(0, -1), f(all[all.length - 1])]);

  async function send(e: React.FormEvent) {
    e.preventDefault();
    const content = text.trim();
    if (!content || busy) return;
    setText("");
    setBusy(true);
    history.current = [...history.current, { role: "user", content }];
    setMsgs([...history.current, { role: "assistant", content: "" }]);
    try {
      const res = await fetch("/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "baton",
          stream: true,
          messages: history.current.map(({ role, content }) => ({ role, content })),
        }),
      });
      if (!res.ok || !res.body) {
        const body = await res.json().catch(() => null);
        throw new Error(body?.error?.message ?? `HTTP ${res.status}`);
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let reply = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop() ?? "";
        for (const event of events) {
          const data = event.replace(/^data: /, "");
          if (!data || data === "[DONE]") continue;
          const chunk = JSON.parse(data);
          if (chunk.error) throw new Error(chunk.error.message);
          reply += chunk.choices[0].delta?.content ?? "";
          const timings = chunk.x_baton && {
            ttft_ms: chunk.x_baton.ttft_ms,
            decode_tok_s: chunk.x_baton.decode_tok_s,
            tokens: chunk.usage.completion_tokens,
          };
          patchLast((m) => ({ ...m, content: reply, timings }));
        }
      }
      history.current = [...history.current, { role: "assistant", content: reply }];
    } catch (err) {
      history.current = history.current.slice(0, -1); // the failed question is not history
      patchLast((m) => ({ ...m, error: String(err instanceof Error ? err.message : err) }));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="view chat">
      <div className="transcript">
        {msgs.length === 0 && <p className="muted">Ask the cluster something.</p>}
        {msgs.map((m, i) => (
          <div className={`msg ${m.role}`} key={i}>
            <div className="msg-role muted">{m.role}</div>
            <div className="msg-body">{m.content}</div>
            {m.error && <div className="msg-body">Error: {m.error}</div>}
            {m.timings && (
              <div className="timing muted">
                TTFT {ms(m.timings.ttft_ms)} &middot; {m.timings.decode_tok_s.toFixed(2)} tok/s
                &middot; {m.timings.tokens} tokens
              </div>
            )}
          </div>
        ))}
      </div>

      <form className="composer" onSubmit={send}>
        <input
          type="text"
          placeholder="Message"
          value={text}
          onChange={(e) => setText(e.target.value)}
          disabled={busy}
        />
        <button type="submit" disabled={busy || !text.trim()}>
          Send
        </button>
      </form>
    </section>
  );
}
