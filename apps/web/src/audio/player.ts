import { AudioRing } from "@spectra-peel/engine";
import { layerGains, type Project } from "../project/schema";
import type { ReaderMessage, ReaderRequest } from "../workers/protocol";
import type { MixerMessage, MixerOptions } from "./mixer.worklet";
// oxlint-disable-next-line import/default -- Vite's ?worker&url query has a default export the resolver cannot see
import workletUrl from "./mixer.worklet.ts?worker&url";

/** About 0.7 s at 44.1 kHz: enough to ride out a main-thread stall without making seeks feel late. */
const RING_FRAMES = 32_768;

/**
 * Plays every layer of a project at once through its own gain. Each layer also feeds an analyser before
 * its gain, so the live view shows a layer even while it is muted. A new layer needs a new Player.
 */
export class Player {
  readonly analysers: ReadonlyMap<string, AnalyserNode>;
  readonly sampleRate: number;
  readonly frames: number;
  private readonly project: Project;
  private readonly ctx: AudioContext;
  private readonly gains = new Map<string, GainNode>();
  private readonly ring: AudioRing;
  private readonly reader: Worker;
  private readonly ringBuffer: SharedArrayBuffer;
  private readonly ringChannels: number;
  private node: AudioWorkletNode | undefined;
  private playing = false;
  private pausedAt: number;
  private onEnded: () => void = () => undefined;

  constructor(project: Project, position = 0) {
    this.project = project;
    this.sampleRate = project.source.sampleRate;
    this.frames = project.source.frames;
    this.pausedAt = position;
    this.ctx = new AudioContext({ sampleRate: this.sampleRate, latencyHint: "interactive" });
    const analysers = new Map<string, AnalyserNode>();
    for (const layer of project.layers) {
      const gain = this.ctx.createGain();
      gain.connect(this.ctx.destination);
      this.gains.set(layer.id, gain);
      analysers.set(layer.id, new AnalyserNode(this.ctx, { fftSize: 4096, smoothingTimeConstant: 0.5 }));
    }
    this.analysers = analysers;
    this.setMix(project);

    this.ringChannels = project.layers.length * project.source.channels;
    this.ringBuffer = new SharedArrayBuffer(AudioRing.bytes(this.ringChannels, RING_FRAMES));
    this.ring = new AudioRing(this.ringBuffer, this.ringChannels, RING_FRAMES);
    this.reader = new Worker(new URL("../workers/reader.worker.ts", import.meta.url), { type: "module" });
    this.reader.addEventListener("message", (e: MessageEvent<ReaderMessage>) => {
      if (e.data.type === "error") console.error("reader worker:", e.data.message);
    });
    this.send({
      type: "open",
      project,
      ring: this.ringBuffer,
      channels: this.ringChannels,
      capacity: RING_FRAMES,
      position,
    });
  }

  get isPlaying(): boolean {
    return this.playing;
  }

  /** Source frame being played, or where playback will resume. */
  position(): number {
    if (!this.playing) return this.pausedAt;
    const at = this.ring.playedFrame();
    if (at >= this.frames) {
      this.pause();
      this.pausedAt = this.frames;
      this.onEnded();
    }
    return Math.min(at, this.frames);
  }

  setOnEnded(fn: () => void): void {
    this.onEnded = fn;
  }

  /** Applies the project's mix (gains, mutes, solo, A/B) with a 10 ms ramp. */
  setMix(project: Project): void {
    for (const [id, value] of layerGains(project)) {
      this.gains.get(id)?.gain.setTargetAtTime(value, this.ctx.currentTime, 0.01);
    }
  }

  async play(): Promise<void> {
    if (this.playing) return;
    if (this.pausedAt >= this.frames) this.seek(0);
    await this.ensureNode();
    await this.ctx.resume();
    this.playing = true;
    this.mixer({ playing: true });
  }

  pause(): void {
    if (!this.playing) return;
    this.pausedAt = Math.min(this.ring.playedFrame(), this.frames);
    this.playing = false;
    this.mixer({ playing: false });
    this.seek(this.pausedAt);
  }

  seek(frame: number): void {
    const target = Math.max(0, Math.min(this.frames, Math.round(frame)));
    if (!this.playing) this.pausedAt = target;
    this.send({ type: "seek", frame: target });
  }

  /** Closes the reader's sync handles before ending the worker, so the files can be reopened at once. */
  async close(): Promise<void> {
    const closed = new Promise<void>((resolve) => {
      this.reader.addEventListener("message", (e: MessageEvent<ReaderMessage>) => {
        if (e.data.type === "closed") resolve();
      });
      setTimeout(resolve, 1000);
    });
    this.send({ type: "close" });
    await closed;
    this.reader.terminate();
    await this.ctx.close();
  }

  private async ensureNode(): Promise<AudioWorkletNode> {
    if (this.node) return this.node;
    await this.ctx.audioWorklet.addModule(workletUrl);
    const processorOptions: MixerOptions = {
      ring: this.ringBuffer,
      channels: this.ringChannels,
      capacity: RING_FRAMES,
      layerChannels: this.project.source.channels,
    };
    const node = new AudioWorkletNode(this.ctx, "spectra-mixer", {
      numberOfInputs: 0,
      numberOfOutputs: this.project.layers.length,
      outputChannelCount: this.project.layers.map(() => 2),
      processorOptions,
    });
    for (const [i, layer] of this.project.layers.entries()) {
      const gain = this.gains.get(layer.id);
      const analyser = this.analysers.get(layer.id);
      if (gain) node.connect(gain, i);
      if (analyser) node.connect(analyser, i);
    }
    this.node = node;
    return node;
  }

  private mixer(m: MixerMessage): void {
    this.node?.port.postMessage(m);
  }

  private send(m: ReaderRequest): void {
    this.reader.postMessage(m);
  }
}
