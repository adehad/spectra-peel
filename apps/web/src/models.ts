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
  "speaker-segmentation",
  "speaker-embedding",
  "speaker-separation",
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

/** Who speaks when, voice fingerprints, then the crosstalk separator. */
export const SPEAKER_MODELS: readonly ModelId[] = [
  "speaker-segmentation",
  "speaker-embedding",
  "speaker-separation",
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
  "speaker-segmentation": {
    label: "pyannote segmentation-3.0",
    purpose: "Who speaks when, for separating people",
    // sherpa-onnx's export of pyannote/segmentation-3.0.
    url: "https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/9403a6902bb58e3d5ae8c7e77c3422de279db2e0/model.onnx",
    sha256: "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079",
    bytes: 5_992_913,
    bundled: false,
    license: "MIT",
  },
  "speaker-embedding": {
    label: "WeSpeaker ECAPA-TDNN",
    purpose: "Voice fingerprints, for separating people",
    url: "https://huggingface.co/Wespeaker/wespeaker-voxceleb-ecapa-tdnn512/resolve/89426e111abc9ffcc40279baf808e087693c6e9b/voxceleb_ECAPA512.onnx",
    sha256: "f901ae3571912bad4e11e157744cb8b7d77a0b46923b83c410bdbb6c77d69736",
    bytes: 24_861_931,
    bundled: false,
    license: "Apache-2.0",
  },
  "speaker-separation": {
    label: "TIGER",
    purpose: "Splits two people talking at once",
    // reference/tiger/export_tiger.py on JusperLee/TIGER-speech at f0340340b2d9bbf72074edf8c076dcab59a10ba2.
    // Not hosted yet: add it from a file.
    url: null,
    sha256: "83c027f2164d79b32f02f2f0e9e9a867ce3bf97544fe88f8b33e224d85533320",
    bytes: 59_080_388,
    bundled: false,
    license: "Apache-2.0",
  },
};
