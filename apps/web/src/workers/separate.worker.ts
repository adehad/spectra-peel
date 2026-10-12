import { DEMUCS, type SeparationModel } from "@spectra-peel/engine";
import * as ort from "onnxruntime-web/jspi";
import { z } from "zod";
import { LayerBlocks } from "../audio/layer-blocks";
import { FAST_STEMS, SPECIALISTS, type ModelId } from "../models";
import { dependencies } from "../project/layers";
import { modelBytes } from "../storage/models";
import { projectDir } from "../storage/opfs";
import { loadJson, saveJson } from "./checkpoint";
import { TileWriter, mono } from "./layer-writer";
import { passCheckpoints, progressPoster, separatePass, type PassModel } from "./separate-pass";
import { createSession } from "./ort-session";
import type { SeparateMessage, SeparateRequest } from "./protocol";

const post = (m: SeparateMessage) => postMessage(m);

/** Runs an HT-Demucs graph and keeps the `keep` sources of its four, in that order. */
class OrtDemucs implements SeparationModel {
  private readonly session: ort.InferenceSession;
  private readonly keep: readonly number[];

  constructor(session: ort.InferenceSession, keep: readonly number[]) {
    this.session = session;
    this.keep = keep;
  }

  async run(mix: Float32Array): Promise<Float32Array> {
    const out = await this.session.run({ mix: new ort.Tensor("float32", mix, [1, 2, DEMUCS.segment]) });
    const stems = out["stems"];
    const data = await stems?.getData();
    stems?.dispose();
    if (!(data instanceof Float32Array)) throw new Error("model has no float32 'stems' output");
    if (this.keep.length === DEMUCS.sources.length) return data;
    const n = 2 * DEMUCS.segment;
    const picked = new Float32Array(this.keep.length * n);
    for (const [j, s] of this.keep.entries()) picked.set(data.subarray(s * n, (s + 1) * n), j * n);
    return picked;
  }
}

interface Pass {
  model: ModelId;
  /** Indexes into the model's sources, and into the request's stems, that this pass writes. */
  keep: number[];
}

/** The fast model for every stem not in `best`, then each chosen stem's specialist. */
function passesFor(best: readonly number[]): Pass[] {
  const fast = DEMUCS.sources.map((_, i) => i).filter((i) => !best.includes(i));
  return [
    ...(fast.length > 0 ? [{ model: FAST_STEMS, keep: fast }] : []),
    ...best.flatMap((i) => {
      const model = SPECIALISTS[i];
      return model ? [{ model, keep: [i] }] : [];
    }),
  ];
}

/** Spectrogram of the input minus the stems, read back from the stems once they are all written. */
async function writeRestTiles(dir: FileSystemDirectoryHandle, req: SeparateRequest): Promise<void> {
  const { project, stems, rest } = req;
  const withStems = { ...project, layers: [...project.layers, ...stems, rest] };
  const blocks = await LayerBlocks.open(dir, withStems, dependencies(withStems, rest.id));
  const tiles = new TileWriter(dir, rest.id, project.source.sampleRate);
  try {
    for (let k = 0; k < blocks.totalBlocks; k++) await tiles.push(mono(blocks.block(rest.id, k)));
  } finally {
    blocks.close();
  }
  await tiles.finish();
}

/** How many passes have written all their stems; a resumed run starts after them. */
const PassesDone = z.object({ done: z.number().int().nonnegative() });

async function separate(req: SeparateRequest): Promise<{ device: string; realTimeFactor: number }> {
  const { project, op, stems, rest } = req;
  const total = project.source.frames;
  const dir = await projectDir(project.id);
  const passes = passesFor(req.best);
  const t0 = performance.now();
  let device = "";
  const resumed = PassesDone.safeParse(await loadJson(dir, op.id, "passes.json"));
  const skip = resumed.success ? resumed.data.done : 0;
  const progress = progressPoster(post, t0);

  for (const [p, pass] of passes.entries()) {
    if (p < skip) continue;
    const what =
      pass.keep.length > 1 ? "Separating" : `${stems[pass.keep[0] ?? 0]?.label ?? "Stem"} at best quality`;
    const label = passes.length === 1 ? what : `${what} (${p + 1} of ${passes.length})`;
    const checkpoints = await passCheckpoints(dir, op.id, `pass-${p}`);
    const from = (p + (checkpoints.resume?.written ?? 0) / total) / passes.length;
    post({
      type: "progress",
      fraction: from,
      stage: `${label}: loading model`,
      etaSeconds: null,
    });
    const created = await createSession(await modelBytes(pass.model), req.useGpu);
    device = created.device;
    const stage = `${label} on the ${device}`;
    post({ type: "progress", fraction: from, stage, etaSeconds: null });
    const everyStem = pass.keep.length === DEMUCS.sources.length;
    const restTiles = everyStem ? new TileWriter(dir, rest.id, project.source.sampleRate) : null;
    try {
      const model: PassModel = {
        model: new OrtDemucs(created.session, pass.keep),
        sampleRate: DEMUCS.sampleRate,
        segment: DEMUCS.segment,
        overlap: DEMUCS.overlap,
      };
      const layers = pass.keep.flatMap((i) => stems[i] ?? []);
      const report = (written: number) => progress((p + written / total) / passes.length, stage);
      await separatePass(dir, project, op.input, model, layers, restTiles, report, checkpoints);
    } finally {
      await created.session.release();
    }
    await restTiles?.finish();
    await saveJson(dir, op.id, "passes.json", { done: p + 1 });
  }
  if (passes.length > 1) {
    post({ type: "progress", fraction: 1, stage: "Finding what the stems missed", etaSeconds: null });
    await writeRestTiles(dir, req);
  }
  return { device, realTimeFactor: (performance.now() - t0) / 1000 / (total / project.source.sampleRate) };
}

addEventListener("message", (e: MessageEvent<SeparateRequest>) => {
  separate(e.data).then(
    (r) => post({ type: "done", ...r }),
    (err: unknown) => post({ type: "error", message: err instanceof Error ? err.message : String(err) }),
  );
});
