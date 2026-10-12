import { AUDIOSEP, type SeparationModel } from "@spectra-peel/engine";
import * as ort from "onnxruntime-web/jspi";
import prompts from "../extract-prompts.json";
import { modelBytes } from "../storage/models";
import { projectDir } from "../storage/opfs";
import { TileWriter } from "./layer-writer";
import { createSession } from "./ort-session";
import type { ExtractMessage, ExtractRequest } from "./protocol";
import { passCheckpoints, progressPoster, separatePass } from "./separate-pass";

const post = (m: ExtractMessage) => postMessage(m);

/**
 * Runs AudioSep on the mid channel and returns its target as both channels: the model is mono, and
 * running each channel would double the cost. Whatever differs between the channels goes to Rest.
 */
class OrtAudioSep implements SeparationModel {
  private readonly session: ort.InferenceSession;
  private readonly condition: ort.Tensor;

  constructor(session: ort.InferenceSession, embedding: Float32Array) {
    this.session = session;
    this.condition = new ort.Tensor("float32", embedding, [1, AUDIOSEP.embedding]);
  }

  async run(mix: Float32Array): Promise<Float32Array> {
    const n = AUDIOSEP.segment;
    const mid = new Float32Array(n);
    for (let i = 0; i < n; i++) mid[i] = ((mix[i] ?? 0) + (mix[n + i] ?? 0)) / 2;
    const out = await this.session.run({
      mixture: new ort.Tensor("float32", mid, [1, 1, n]),
      condition: this.condition,
    });
    const target = out["target"];
    const data = await target?.getData();
    target?.dispose();
    if (!(data instanceof Float32Array)) throw new Error("model has no float32 'target' output");
    const both = new Float32Array(2 * n);
    both.set(data.subarray(0, n));
    both.set(data.subarray(0, n), n);
    return both;
  }
}

async function extract(req: ExtractRequest): Promise<{ device: string; realTimeFactor: number }> {
  const { project, op, target, rest } = req;
  const embedding = (prompts as Record<string, number[] | undefined>)[req.prompt];
  if (!embedding) throw new Error(`no embedding for "${req.prompt}"`);
  const total = project.source.frames;
  const dir = await projectDir(project.id);
  const t0 = performance.now();
  const checkpoints = await passCheckpoints(dir, op.id, "pass");
  const from = (checkpoints.resume?.written ?? 0) / total;
  post({ type: "progress", fraction: from, stage: `${req.prompt}: loading model`, etaSeconds: null });
  const created = await createSession(await modelBytes("audiosep-base"), req.useGpu);
  const stage = `${req.prompt} on the ${created.device}`;
  post({ type: "progress", fraction: from, stage, etaSeconds: null });
  const restTiles = new TileWriter(dir, rest.id, project.source.sampleRate);
  const model = {
    model: new OrtAudioSep(created.session, Float32Array.from(embedding)),
    sampleRate: AUDIOSEP.sampleRate,
    segment: AUDIOSEP.segment,
    overlap: AUDIOSEP.overlap,
  };
  const progress = progressPoster(post, t0);
  try {
    const report = (written: number) => progress(written / total, stage);
    await separatePass(dir, project, op.input, model, [target], restTiles, report, checkpoints);
  } finally {
    await created.session.release();
  }
  await restTiles.finish();
  return {
    device: created.device,
    realTimeFactor: (performance.now() - t0) / 1000 / (total / project.source.sampleRate),
  };
}

addEventListener("message", (e: MessageEvent<ExtractRequest>) => {
  extract(e.data).then(
    (r) => post({ type: "done", ...r }),
    (err: unknown) => post({ type: "error", message: err instanceof Error ? err.message : String(err) }),
  );
});
