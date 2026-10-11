import type { ModelKey } from "./models";

export interface RunRequest {
  model: ModelKey;
  ep: "webgpu" | "wasm";
  seconds: number;
  /** Overrides the download, for models patched or converted locally. */
  modelFile?: File;
  /** Feeds random tensors shaped from the model's input metadata and times this many runs, for subgraphs. */
  graphRuns?: number;
  webgpu?: {
    preferredLayout?: "NCHW" | "NHWC";
    enableGraphCapture?: boolean;
    profile?: boolean;
    /** Hides the adapter's `subgroups` feature from ORT before it creates its device. */
    noSubgroups?: boolean;
  };
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
