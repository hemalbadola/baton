import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The head serves this build from dashboard/dist at `/` (PRD 13.1).
export default defineConfig({
  plugins: [react()],
  base: "./",
  server: {
    proxy: {
      "/cluster": "http://127.0.0.1:7700",
      "/v1": "http://127.0.0.1:7700",
      "/ws": { target: "ws://127.0.0.1:7700", ws: true },
    },
  },
});
