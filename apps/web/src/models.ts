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

export const MODEL_IDS = [
  "dfn3-stream",
  "htdemucs-stems",
  "htdemucs-ft-drums",
  "htdemucs-ft-bass",
  "htdemucs-ft-other",
  "htdemucs-ft-vocals",
] as const;
export type ModelId = (typeof MODEL_IDS)[number];

/** One model for all four stems. */
export const FAST_STEMS: ModelId = "htdemucs-stems";

/**
 * htdemucs_ft, in HT-Demucs' source order. Its bag weights take one stem from each of four fine-tuned
 * models, so each can run alone and keep only the stem it was tuned for.
 */
export const SPECIALISTS: readonly ModelId[] = [
  "htdemucs-ft-drums",
  "htdemucs-ft-bass",
  "htdemucs-ft-other",
  "htdemucs-ft-vocals",
];

/** The models a separation runs in turn: the fast model for any stems not in `best`, then a specialist per stem in it. */
export function stemModels(best: readonly number[]): ModelId[] {
  const fast = best.length < SPECIALISTS.length ? [FAST_STEMS] : [];
  return [...fast, ...best.flatMap((i) => SPECIALISTS[i] ?? [])];
}

// reference/patch_float64.py, simplify.py and rewrite_stft.py applied to StemSplitio's
// htdemucs-ft-<stem>-onnx htdemucs_ft_<stem>_fp16weights.onnx, at revisions drums 55f929d3, bass 410457f1,
// other db6d606b and vocals 2ef0d757. Not hosted yet: add them from files.
function specialist(stem: string, sha256: string, bytes: number): ModelSpec {
  return {
    label: `HT-Demucs FT ${stem}`,
    purpose: `Best-quality ${stem.toLowerCase()}, one of four`,
    url: null,
    sha256,
    bytes,
    bundled: false,
    license: "MIT",
  };
}

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
    sha256: "04823dc557b0e458fe49ec84da2b6af991159216d6753c9a203315745032aa55",
    bytes: 345_402_157,
    bundled: false,
    license: "MIT",
  },
  "htdemucs-ft-drums": specialist(
    "Drums",
    "10a4b0d5fc20b228ada60109d789ebda1a5d7cfa6de6e950f42a79c1a97ad09c",
    345_402_157,
  ),
  "htdemucs-ft-bass": specialist(
    "Bass",
    "91d43d0d12318974ecf02aafdd29bcf51120e03df21197ee4801f449cbc84b41",
    345_402_157,
  ),
  "htdemucs-ft-other": specialist(
    "Other",
    "8f83d100d30c03f483d24b8ebf4070e02e6204f94a4a4ce888ef7b98bee2039f",
    345_402_157,
  ),
  "htdemucs-ft-vocals": specialist(
    "Vocals",
    "526c7cb51f6ae443b1a923e307e6ca36f96340e10d558fdc2b6c17fab6284a40",
    345_402_157,
  ),
};
