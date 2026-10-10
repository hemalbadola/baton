import { useEffect, useState } from "react";
import type { ClusterSnapshot } from "./types";
import { Topology } from "./views/Topology";
import { Live } from "./views/Live";
import { Plan } from "./views/Plan";
import { Chat } from "./views/Chat";
import { Log } from "./views/Log";

const VIEWS = ["Topology", "Live", "Plan", "Chat", "Log"] as const;
type View = (typeof VIEWS)[number];

/** The head pushes one snapshot at 2 Hz on /ws (PRD 14.1). Reconnects when it drops. */
function useCluster(): ClusterSnapshot | null {
  const [snap, setSnap] = useState<ClusterSnapshot | null>(null);
  useEffect(() => {
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    let socket: WebSocket;
    let closed = false;
    const open = () => {
      socket = new WebSocket(`${scheme}://${location.host}/ws`);
      socket.onmessage = (e) => setSnap(JSON.parse(e.data));
      socket.onclose = () => {
        if (!closed) setTimeout(open, 1000);
      };
    };
    open();
    return () => {
      closed = true;
      socket.close();
    };
  }, []);
  return snap;
}

export default function App() {
  const [view, setView] = useState<View>("Topology");
  const snap = useCluster();
  if (!snap) {
    return <p className="muted" style={{ padding: 24 }}>Connecting to the head...</p>;
  }

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

      <footer className="muted">
        {snap.nodes.length} node(s) &middot; {snap.live.active} active &middot; {snap.live.queued} queued
      </footer>
    </div>
  );
}
