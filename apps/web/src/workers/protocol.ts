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
}

export type ExportMessage =
  | { type: "formats"; formats: ExportFormat[] }
  | { type: "progress"; fraction: number }
  | { type: "done"; path: string; bytes: number }
  | { type: "error"; message: string };
