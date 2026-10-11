import { create } from "zustand";

/** Long-running work that a page reload would lose. The update prompt waits until this is empty. */
interface JobsState {
  running: ReadonlySet<string>;
  start: (id: string) => void;
  end: (id: string) => void;
}

export const useJobs = create<JobsState>((set) => ({
  running: new Set(),
  start: (id) => set((s) => ({ running: new Set([...s.running, id]) })),
  end: (id) => set((s) => ({ running: new Set([...s.running].filter((x) => x !== id)) })),
}));
