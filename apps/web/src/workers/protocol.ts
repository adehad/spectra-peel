import type { Layer, Op, Project } from "../project/schema";

export interface ImportRequest {
  file: File;
  projectId: string;
  name: string;
}

export type ImportMessage =
  | { type: "progress"; fraction: number }
  | { type: "done"; project: Project }
  | { type: "error"; message: string };

export type ReaderRequest =
  | {
      type: "open";
      project: Project;
      ring: SharedArrayBuffer;
      /** Ring channels: every layer's channels, in project layer order. */
      channels: number;
      capacity: number;
      position: number;
    }
  | { type: "seek"; frame: number }
  | { type: "close" };

export type ReaderMessage = { type: "ready" } | { type: "closed" } | { type: "error"; message: string };

export interface DenoiseRequest {
  project: Project;
  op: Op;
  /** The FLAC layer to write, then the residual layer whose tiles to write. */
  output: Layer;
  removed: Layer;
}

export type DenoiseMessage =
  | { type: "progress"; fraction: number; stage: string }
  | { type: "done"; realTimeFactor: number }
  | { type: "error"; message: string };

export type ExportFormat = "wav" | "flac" | "m4a" | "opus";

export interface ExportRequest {
  project: Project;
  format: ExportFormat;
  /** Source samples. */
  range: { start: number; end: number };
  fileName: string;
  /** Bits per second for M4A and Opus; ignored by the lossless formats. */
  bitrate: number;
  /** Loudness target in LUFS and true-peak ceiling in dBTP, or null to export at the mix's own level. */
  normalize: { target: number; ceiling: number } | null;
}

export interface LoudnessReport {
  measured: number;
  truePeak: number;
  gainDb: number;
}

export type ExportMessage =
  | { type: "formats"; formats: ExportFormat[]; kbps: Partial<Record<ExportFormat, number[]>> }
  | { type: "progress"; fraction: number }
  | { type: "done"; path: string; bytes: number; loudness: LoudnessReport | null }
  | { type: "error"; message: string };

export interface SeparateRequest {
  project: Project;
  op: Op;
  /** FLAC layers in the model's source order, then the residual whose tiles to write. */
  stems: Layer[];
  rest: Layer;
  /** Stems, by index, to take from their fine-tuned specialists instead of the fast model. */
  best: number[];
  useGpu: boolean;
}

export type SeparateMessage =
  | { type: "progress"; fraction: number; stage: string; etaSeconds: number | null }
  | { type: "done"; device: string; realTimeFactor: number }
  | { type: "error"; message: string };
