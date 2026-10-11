/** MSB-first bit writer over a fixed buffer; the caller sizes it for the worst case. */
export class BitWriter {
  readonly bytes: Uint8Array<ArrayBuffer>;
  private pos = 0;
  private acc = 0;
  private accBits = 0;

  constructor(capacity: number) {
    this.bytes = new Uint8Array(capacity);
  }

  get bitLength(): number {
    return this.pos * 8 + this.accBits;
  }

  get byteLength(): number {
    return this.pos;
  }

  /** Writes the low `n` bits of `value`, n <= 24. */
  bits(value: number, n: number): void {
    this.acc = (this.acc << n) | (value & ((1 << n) - 1));
    this.accBits += n;
    while (this.accBits >= 8) {
      this.accBits -= 8;
      this.bytes[this.pos++] = (this.acc >>> this.accBits) & 0xff;
    }
    this.acc &= (1 << this.accBits) - 1;
  }

  /** Writes `n` bits of `value`, n <= 48. */
  long(value: number, n: number): void {
    if (n > 24) {
      this.bits(Math.floor(value / 2 ** 24), n - 24);
      this.bits(value % 2 ** 24, 24);
    } else this.bits(value, n);
  }

  /** `q` zero bits then a one. */
  unary(q: number): void {
    while (q >= 24) {
      this.bits(0, 24);
      q -= 24;
    }
    this.bits(1, q + 1);
  }

  rice(v: number, k: number): void {
    const u = v >= 0 ? v * 2 : -v * 2 - 1;
    this.unary(u >>> k);
    if (k > 0) this.bits(u, k);
  }

  alignToByte(): void {
    if (this.accBits > 0) this.bits(0, 8 - this.accBits);
  }

  byte(value: number): void {
    this.bits(value, 8);
  }
}

export class BitReader {
  private readonly bytes: Uint8Array;
  private pos: number;
  private acc = 0;
  private accBits = 0;

  constructor(bytes: Uint8Array, offset = 0) {
    this.bytes = bytes;
    this.pos = offset;
  }

  /** Byte position, valid only when aligned. */
  get bytePos(): number {
    return this.pos - (this.accBits >> 3);
  }

  /** n <= 24. */
  bits(n: number): number {
    while (this.accBits < n) {
      const b = this.bytes[this.pos++];
      if (b === undefined) throw new RangeError("read past end of FLAC data");
      this.acc = ((this.acc << 8) | b) >>> 0;
      this.accBits += 8;
    }
    this.accBits -= n;
    const v = (this.acc >>> this.accBits) & ((1 << n) - 1);
    this.acc &= this.accBits === 0 ? 0 : (1 << this.accBits) - 1;
    return v;
  }

  long(n: number): number {
    return n > 24 ? this.bits(n - 24) * 2 ** 24 + this.bits(24) : this.bits(n);
  }

  signed(n: number): number {
    const v = this.bits(n);
    return v >= 2 ** (n - 1) ? v - 2 ** n : v;
  }

  unary(): number {
    let q = 0;
    while (this.bits(1) === 0) q++;
    return q;
  }

  rice(k: number): number {
    const u = this.unary() * 2 ** k + (k > 0 ? this.bits(k) : 0);
    return u % 2 === 0 ? u / 2 : -(u + 1) / 2;
  }

  alignToByte(): void {
    this.accBits -= this.accBits % 8;
    this.acc &= this.accBits === 0 ? 0 : (1 << this.accBits) - 1;
  }
}

const CRC8 = new Uint8Array(256);
const CRC16 = new Uint16Array(256);
for (let i = 0; i < 256; i++) {
  let c8 = i;
  let c16 = i << 8;
  for (let b = 0; b < 8; b++) {
    c8 = c8 & 0x80 ? ((c8 << 1) ^ 0x07) & 0xff : (c8 << 1) & 0xff;
    c16 = c16 & 0x8000 ? ((c16 << 1) ^ 0x8005) & 0xffff : (c16 << 1) & 0xffff;
  }
  CRC8[i] = c8;
  CRC16[i] = c16;
}

export function crc8(bytes: Uint8Array, start: number, end: number): number {
  let c = 0;
  for (let i = start; i < end; i++) c = CRC8[c ^ (bytes[i] ?? 0)] ?? 0;
  return c;
}

export function crc16(bytes: Uint8Array, start: number, end: number): number {
  let c = 0;
  for (let i = start; i < end; i++) c = ((c << 8) & 0xffff) ^ (CRC16[(c >>> 8) ^ (bytes[i] ?? 0)] ?? 0);
  return c;
}
