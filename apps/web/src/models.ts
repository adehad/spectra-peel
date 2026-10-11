export interface ModelSpec {
  label: string;
  purpose: string;
  sha256: string;
  bytes: number;
  /** Where to fetch it. Bundled models ship with the app; others are stored in OPFS by hash. */
  url: string | null;
  bundled: boolean;
  license: string;
}

export const MODEL_IDS = ["dfn3-stream", "htdemucs-stems"] as const;
export type ModelId = (typeof MODEL_IDS)[number];

/** Models the app ships or downloads, pinned by content hash. */
export const MODELS: Record<ModelId, ModelSpec> = {
  "dfn3-stream": {
    label: "DeepFilterNet3",
    purpose: "Speech denoise",
    url: "/models/deepfilter-stream.onnx",
    sha256: "d8e183c7b80b1c8a71230b1350a8d495bad68699974a8e289e5d003a131ea39a",
    bytes: 8_609_015,
    bundled: true,
    license: "MIT",
  },
  "htdemucs-stems": {
    label: "HT-Demucs",
    purpose: "Four stems: drums, bass, other, vocals",
    // reference/patch_float64.py, simplify.py and rewrite_stft.py applied to StemSplitio/htdemucs-onnx
    // htdemucs_fp16weights.onnx at d54ed9eb60e258ea82131c6ee14578628816456a. Not hosted yet: add it from a file.
    url: null,
    sha256: "",
    bytes: 0,
    bundled: false,
    license: "MIT",
  },
};
