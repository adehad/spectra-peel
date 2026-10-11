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
