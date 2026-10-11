import { AudioRing } from "@spectra-peel/engine";

// lib.dom has AudioWorkletNode but not the processor-side globals.
declare global {
  abstract class AudioWorkletProcessor {
    readonly port: MessagePort;
    constructor(options?: AudioWorkletNodeOptions);
  }
  function registerProcessor(
    name: string,
    ctor: new (options: AudioWorkletNodeOptions) => AudioWorkletProcessor,
  ): void;
}

export interface MixerOptions {
  ring: SharedArrayBuffer;
  channels: number;
  capacity: number;
}

export type MixerMessage = { playing: boolean };

function isMixerOptions(o: unknown): o is MixerOptions {
  return (
    typeof o === "object" &&
    o !== null &&
    "ring" in o &&
    o.ring instanceof SharedArrayBuffer &&
    "channels" in o &&
    typeof o.channels === "number" &&
    "capacity" in o &&
    typeof o.capacity === "number"
  );
}

class Mixer extends AudioWorkletProcessor {
  private readonly ring: AudioRing;
  private readonly scratch: Float32Array[];
  private playing = false;

  constructor(options: AudioWorkletNodeOptions) {
    super(options);
    const o: unknown = options.processorOptions;
    if (!isMixerOptions(o)) throw new TypeError("spectra-mixer needs ring, channels and capacity options");
    this.ring = new AudioRing(o.ring, o.channels, o.capacity);
    this.scratch = Array.from({ length: o.channels }, () => new Float32Array(128));
    this.port.addEventListener("message", (e: MessageEvent<MixerMessage>) => {
      this.playing = e.data.playing;
    });
    this.port.start();
  }

  process(_inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
    const out = outputs[0];
    if (!out || !this.playing) return true;
    const frames = out[0]?.length ?? 128;
    this.ring.read(this.scratch, frames);
    for (const [c, channel] of out.entries()) {
      const src = this.scratch[c] ?? this.scratch[0];
      if (src) channel.set(src.subarray(0, frames));
    }
    return true;
  }
}

registerProcessor("spectra-mixer", Mixer);
