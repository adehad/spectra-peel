export type ModelKey = "demucs-vocals-fp32" | "demucs-vocals-fp16" | "dfn3";

export interface ModelSpec {
  url: string;
  bytes: number;
}

const HF = "https://huggingface.co";
const VOCALS = `${HF}/StemSplitio/htdemucs-ft-vocals-onnx/resolve/2ef0d757d3e226d0da85fb8c71514f464fcabdd0`;
const DFN3 = `${HF}/soniqo/DeepFilterNet3-ONNX/resolve/63d8ba442ba900143c468b798e94a04009b2f0c9`;

export const MODELS: Record<ModelKey, ModelSpec> = {
  "demucs-vocals-fp32": { url: `${VOCALS}/htdemucs_ft_vocals.onnx`, bytes: 316_446_953 },
  "demucs-vocals-fp16": { url: `${VOCALS}/htdemucs_ft_vocals_fp16weights.onnx`, bytes: 165_612_636 },
  dfn3: { url: `${DFN3}/deepfilter.onnx`, bytes: 8_608_859 },
};
