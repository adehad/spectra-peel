import { closeSync, openSync, writeSync } from "node:fs";

/** 16-bit stereo WAV: a tone gliding 100-500 Hz in the left channel, the same tone plus hiss in the right. */
export function syntheticWav(seconds: number, sampleRate = 44_100): Buffer {
  const frames = Math.round(seconds * sampleRate);
  return Buffer.concat([wavHeader(frames, sampleRate), wavData(0, frames, sampleRate)]);
}

/** Writes the same signal to `path` a minute at a time, for files too big to hold in memory. */
export function writeSyntheticWav(path: string, seconds: number, sampleRate = 44_100): void {
  const frames = Math.round(seconds * sampleRate);
  const fd = openSync(path, "w");
  writeSync(fd, wavHeader(frames, sampleRate));
  for (let at = 0; at < frames; at += 60 * sampleRate) {
    writeSync(fd, wavData(at, Math.min(at + 60 * sampleRate, frames), sampleRate));
  }
  closeSync(fd);
}

function wavData(from: number, to: number, sampleRate: number): Buffer {
  const data = Buffer.alloc((to - from) * 4);
  let seed = from + 1;
  for (let i = from; i < to; i++) {
    seed = (seed * 16807) % 2147483647;
    const t = i / sampleRate;
    const tone = 0.4 * Math.sin(2 * Math.PI * (300 * t - 200 * Math.cos(t)));
    const hiss = 0.05 * (seed / 2147483647 - 0.5);
    data.writeInt16LE(Math.round(tone * 32767), (i - from) * 4);
    data.writeInt16LE(Math.round((tone + hiss) * 32767), (i - from) * 4 + 2);
  }
  return data;
}

function wavHeader(frames: number, sampleRate: number): Buffer {
  const dataBytes = frames * 4;
  const header = Buffer.alloc(44);
  header.write("RIFF", 0);
  header.writeUInt32LE(36 + dataBytes, 4);
  header.write("WAVEfmt ", 8);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(2, 22);
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(sampleRate * 4, 28);
  header.writeUInt16LE(4, 32);
  header.writeUInt16LE(16, 34);
  header.write("data", 36);
  header.writeUInt32LE(dataBytes, 40);
  return header;
}
