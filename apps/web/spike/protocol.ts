import type { ModelKey } from "./models";

export interface RunRequest {
  model: ModelKey;
  ep: "webgpu" | "wasm";
  seconds: number;
  /** Overrides the download, for models patched or converted locally. */
  modelFile?: File;
}

export interface Report {
  model: ModelKey;
  epRequested: string;
  epUsed: string;
  epFallbackReason?: string;
  modelBytes: number;
  downloadMs: number;
  fromCache: boolean;
  sessionCreateMs: number;
  clipSeconds: number;
  chunks: number;
  inferenceMs: number;
  perChunkMs: number[];
  realTimeFactor: number;
  outputPeak: number;
}

export type WorkerMessage =
  | { type: "progress"; text: string }
  | { type: "done"; report: Report }
  | { type: "error"; message: string };
