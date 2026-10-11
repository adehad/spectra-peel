import { describe, expect, it } from "vitest";
import { AudioRing } from "./ring";

function pair(channels = 2, capacity = 8) {
  const buffer = new SharedArrayBuffer(AudioRing.bytes(channels, capacity));
  return {
    producer: new AudioRing(buffer, channels, capacity),
    consumer: new AudioRing(buffer, channels, capacity),
  };
}

const seq = (from: number, n: number) => Float32Array.from({ length: n }, (_, i) => from + i);

describe("AudioRing", () => {
  it("passes frames through in order across the wrap point", () => {
    const { producer, consumer } = pair(2, 8);
    const out = [new Float32Array(5), new Float32Array(5)];
    let next = 0;
    for (let round = 0; round < 6; round++) {
      expect(producer.write([seq(next, 5), seq(100 + next, 5)])).toBe(5);
      expect(consumer.read(out, 5)).toBe(5);
      expect(Array.from(out[0] ?? [])).toEqual(Array.from(seq(next, 5)));
      expect(Array.from(out[1] ?? [])).toEqual(Array.from(seq(100 + next, 5)));
      next += 5;
    }
    expect(consumer.playedFrame()).toBe(30);
  });

  it("refuses to overwrite unread frames", () => {
    const { producer } = pair(1, 8);
    expect(producer.write([seq(0, 6)])).toBe(6);
    expect(producer.space()).toBe(2);
    expect(producer.write([seq(6, 6)])).toBe(2);
  });

  it("zero-fills an underrun and reports the short read", () => {
    const { producer, consumer } = pair(1, 8);
    producer.write([seq(1, 3)]);
    const out = [new Float32Array(5).fill(9)];
    expect(consumer.read(out, 5)).toBe(3);
    expect(Array.from(out[0] ?? [])).toEqual([1, 2, 3, 0, 0]);
  });

  it("drops queued audio on seek and resumes from the new source frame", () => {
    const { producer, consumer } = pair(1, 16);
    producer.write([seq(0, 6)]);
    const out = [new Float32Array(2)];
    consumer.read(out, 2);
    producer.seek(44_100);
    producer.write([seq(500, 4)]);
    expect(consumer.available()).toBe(4);
    expect(consumer.read(out, 2)).toBe(2);
    expect(Array.from(out[0] ?? [])).toEqual([500, 501]);
    expect(consumer.playedFrame()).toBe(44_102);
    expect(producer.space()).toBe(14);
  });

  it("keeps the fill level right after the indices wrap past 2^31", () => {
    const capacity = 4;
    const buffer = new SharedArrayBuffer(AudioRing.bytes(1, capacity));
    new Int32Array(buffer, 0, 2).set([0x7ffffffe, 0x7ffffffe]);
    const producer = new AudioRing(buffer, 1, capacity);
    const consumer = new AudioRing(buffer, 1, capacity);
    expect(producer.write([seq(1, 4)])).toBe(4);
    expect(producer.space()).toBe(0);
    const out = [new Float32Array(4)];
    expect(consumer.read(out, 4)).toBe(4);
    expect(Array.from(out[0] ?? [])).toEqual([1, 2, 3, 4]);
  });

  it("requires a power-of-two capacity", () => {
    expect(() => new AudioRing(new SharedArrayBuffer(AudioRing.bytes(1, 6)), 1, 6)).toThrow(RangeError);
  });
});
