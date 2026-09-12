export const gb = (bytes: number): string => `${(bytes / 1e9).toFixed(1)} GB`;

export const ms = (v: number): string =>
  v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${v.toFixed(1)} ms`;

export const pct = (part: number, whole: number): number =>
  whole > 0 ? Math.min(100, (part / whole) * 100) : 0;

export const clock = (t: number): string =>
  new Date(t * 1000).toLocaleTimeString([], { hour12: false });
