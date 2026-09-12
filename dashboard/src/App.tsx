import { useState } from "react";
import { SAMPLE } from "./sample";
import { Topology } from "./views/Topology";
import { Live } from "./views/Live";
import { Plan } from "./views/Plan";
import { Chat } from "./views/Chat";
import { Log } from "./views/Log";

const VIEWS = ["Topology", "Live", "Plan", "Chat", "Log"] as const;
type View = (typeof VIEWS)[number];

export default function App() {
  const [view, setView] = useState<View>("Topology");
  // Static shell: one hardcoded snapshot. M3 replaces this with /cluster + /ws.
  const snap = SAMPLE;

  return (
    <div className="app">
      <header>
        <div className="brand">baton</div>
        <nav>
          {VIEWS.map((v) => (
            <button
              key={v}
              className={v === view ? "tab active" : "tab"}
              onClick={() => setView(v)}
            >
              {v}
            </button>
          ))}
        </nav>
        <div className="status">
          <span className={`tag state-${snap.state}`}>{snap.state}</span>
          <span className="muted">{snap.model?.id ?? "no model"}</span>
          <span className="muted">rev {snap.plan_rev}</span>
        </div>
      </header>

      <main>
        {view === "Topology" && <Topology snap={snap} />}
        {view === "Live" && <Live snap={snap} />}
        {view === "Plan" && <Plan snap={snap} />}
        {view === "Chat" && <Chat />}
        {view === "Log" && <Log snap={snap} />}
      </main>

      <footer className="muted">Sample data. Live wiring lands in M3.</footer>
    </div>
  );
}
