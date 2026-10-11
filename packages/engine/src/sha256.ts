const K = Uint32Array.from([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
  0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
  0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
  0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
  0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
  0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
  0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
  0xc67178f2,
]);

/**
 * Incremental SHA-256, for checking a model while it streams to disk. WebCrypto only hashes a whole
 * buffer, which for a 345 MB model is more memory than a phone tab should hold at once.
 */
export class Sha256 {
  private readonly h = Uint32Array.from([
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
  ]);
  private readonly w = new Uint32Array(64);
  private readonly block = new Uint8Array(64);
  private fill = 0;
  private length = 0;

  update(data: Uint8Array): this {
    let at = 0;
    this.length += data.length;
    while (at < data.length) {
      const n = Math.min(64 - this.fill, data.length - at);
      this.block.set(data.subarray(at, at + n), this.fill);
      this.fill += n;
      at += n;
      if (this.fill === 64) {
        this.compress(this.block);
        this.fill = 0;
      }
    }
    return this;
  }

  hex(): string {
    const bits = this.length * 8;
    this.update(Uint8Array.of(0x80));
    if (this.fill > 56) this.update(new Uint8Array(64 - this.fill));
    this.update(new Uint8Array(56 - this.fill));
    const tail = new DataView(new ArrayBuffer(8));
    tail.setUint32(0, Math.floor(bits / 2 ** 32));
    tail.setUint32(4, bits >>> 0);
    this.update(new Uint8Array(tail.buffer));
    return Array.from(this.h, (v) => v.toString(16).padStart(8, "0")).join("");
  }

  private compress(b: Uint8Array): void {
    const w = this.w;
    for (let i = 0; i < 16; i++) {
      w[i] =
        (((b[4 * i] ?? 0) << 24) |
          ((b[4 * i + 1] ?? 0) << 16) |
          ((b[4 * i + 2] ?? 0) << 8) |
          (b[4 * i + 3] ?? 0)) >>>
        0;
    }
    for (let i = 16; i < 64; i++) {
      const x = w[i - 15] ?? 0;
      const y = w[i - 2] ?? 0;
      const s0 = ((x >>> 7) | (x << 25)) ^ ((x >>> 18) | (x << 14)) ^ (x >>> 3);
      const s1 = ((y >>> 17) | (y << 15)) ^ ((y >>> 19) | (y << 13)) ^ (y >>> 10);
      w[i] = ((w[i - 16] ?? 0) + s0 + (w[i - 7] ?? 0) + s1) >>> 0;
    }
    let a = this.h[0] ?? 0;
    let bb = this.h[1] ?? 0;
    let c = this.h[2] ?? 0;
    let d = this.h[3] ?? 0;
    let e = this.h[4] ?? 0;
    let f = this.h[5] ?? 0;
    let g = this.h[6] ?? 0;
    let h = this.h[7] ?? 0;
    for (let i = 0; i < 64; i++) {
      const S1 = ((e >>> 6) | (e << 26)) ^ ((e >>> 11) | (e << 21)) ^ ((e >>> 25) | (e << 7));
      const ch = (e & f) ^ (~e & g);
      const t1: number = (h + S1 + ch + (K[i] ?? 0) + (w[i] ?? 0)) >>> 0;
      const S0: number = ((a >>> 2) | (a << 30)) ^ ((a >>> 13) | (a << 19)) ^ ((a >>> 22) | (a << 10));
      const maj = (a & bb) ^ (a & c) ^ (bb & c);
      const t2: number = (S0 + maj) >>> 0;
      h = g;
      g = f;
      f = e;
      e = (d + t1) >>> 0;
      d = c;
      c = bb;
      bb = a;
      a = (t1 + t2) >>> 0;
    }
    const out = [a, bb, c, d, e, f, g, h];
    for (let i = 0; i < 8; i++) this.h[i] = ((this.h[i] ?? 0) + (out[i] ?? 0)) >>> 0;
  }
}
