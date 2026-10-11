import { AudioRing } from "@spectra-peel/engine";
import type { Project } from "../project/schema";
import type { ReaderMessage, ReaderRequest } from "../workers/protocol";
import type { MixerMessage, MixerOptions } from "./mixer.worklet";
// oxlint-disable-next-line import/default -- Vite's ?worker&url query has a default export the resolver cannot see
import workletUrl from "./mixer.worklet.ts?worker&url";

/** About 0.7 s at 44.1 kHz: enough to ride out a main-thread stall without making seeks feel late. */
const RING_FRAMES = 32_768;

export class Player {
  readonly analyser: AnalyserNode;
  readonly sampleRate: number;
  readonly frames: number;
  private readonly ctx: AudioContext;
  private readonly gain: GainNode;
  private readonly ring: AudioRing;
  private readonly reader: Worker;
  private readonly ringBuffer: SharedArrayBuffer;
  private readonly channels: number;
  private node: AudioWorkletNode | undefined;
  private playing = false;
  private pausedAt = 0;
  private onEnded: () => void = () => undefined;

  constructor(project: Project) {
    this.sampleRate = project.source.sampleRate;
    this.frames = project.source.frames;
    this.channels = project.source.channels;
    this.ctx = new AudioContext({ sampleRate: this.sampleRate, latencyHint: "interactive" });
    this.gain = this.ctx.createGain();
    this.analyser = new AnalyserNode(this.ctx, { fftSize: 4096, smoothingTimeConstant: 0.5 });
    this.gain.connect(this.analyser).connect(this.ctx.destination);

    this.ringBuffer = new SharedArrayBuffer(AudioRing.bytes(this.channels, RING_FRAMES));
    this.ring = new AudioRing(this.ringBuffer, this.channels, RING_FRAMES);
    this.reader = new Worker(new URL("../workers/reader.worker.ts", import.meta.url), { type: "module" });
    this.reader.addEventListener("message", (e: MessageEvent<ReaderMessage>) => {
      if (e.data.type === "error") console.error("reader worker:", e.data.message);
    });
    this.send({
      type: "open",
      projectId: project.id,
      ring: this.ringBuffer,
      channels: this.channels,
      capacity: RING_FRAMES,
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

  setGain(value: number): void {
    this.gain.gain.setTargetAtTime(value, this.ctx.currentTime, 0.01);
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

  /** Closes the reader's sync handle before ending the worker, so the project's files can be reopened at once. */
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
      channels: this.channels,
      capacity: RING_FRAMES,
    };
    this.node = new AudioWorkletNode(this.ctx, "spectra-mixer", {
      numberOfInputs: 0,
      numberOfOutputs: 1,
      outputChannelCount: [2],
      processorOptions,
    });
    this.node.connect(this.gain);
    return this.node;
  }

  private mixer(m: MixerMessage): void {
    this.node?.port.postMessage(m);
  }

  private send(m: ReaderRequest): void {
    this.reader.postMessage(m);
  }
}
