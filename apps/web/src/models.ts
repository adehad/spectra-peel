/** Models the app ships or downloads, pinned by content hash. */
export const MODELS = {
  "dfn3-stream": {
    url: "/models/deepfilter-stream.onnx",
    sha256: "d8e183c7b80b1c8a71230b1350a8d495bad68699974a8e289e5d003a131ea39a",
    bytes: 8_609_015,
  },
} as const;

export type ModelId = keyof typeof MODELS;
