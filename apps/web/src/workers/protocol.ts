import type { Project } from "../project/schema";

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
  | { type: "open"; projectId: string; ring: SharedArrayBuffer; channels: number; capacity: number }
  | { type: "seek"; frame: number }
  | { type: "close" };

export type ReaderMessage = { type: "ready" } | { type: "closed" } | { type: "error"; message: string };
