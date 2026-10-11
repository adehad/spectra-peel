/** pyannote segmentation-3.0 and the windows the diarization runs it on, at 16 kHz. */
export const DIARIZATION = Object.freeze({
  sampleRate: 16_000,
  window: 160_000,
  step: 32_000,
  /** segmentation-3.0's receptive-field shift: the grid speaker activity is kept on. */
  frame: 270,
  /** Fewest single-speaker samples in a window worth an embedding. */
  minEmbedding: 16_000,
  localSpeakers: 3,
  /** AHC is quadratic in memory, so it clusters an even sample of at most this many embeddings. */
  maxLinkage: 1000,
});

/** Speakers active in each of segmentation-3.0's seven powerset classes, as bit masks. */
const POWERSET = [0b000, 0b001, 0b010, 0b100, 0b011, 0b101, 0b110];

export function windowStarts(total: number): number[] {
  const { window, step } = DIARIZATION;
  const starts: number[] = [];
  for (let s = 0; s < Math.max(1, total - window + step); s += step) starts.push(s);
  return starts;
}

/** Per-frame bit masks of a window's local speakers, from the argmax class of each frame. */
export function decodePowerset(classes: Uint8Array): Uint8Array {
  return classes.map((c) => POWERSET[c] ?? 0);
}

/** Sample ranges inside a window where local speaker `j` talks alone. */
export function soloSpans(masks: Uint8Array, j: number): [number, number][] {
  const frames = masks.length;
  const { window } = DIARIZATION;
  const spans: [number, number][] = [];
  for (let f = 0; f < frames; f++) {
    if (masks[f] !== 1 << j) continue;
    const a = Math.floor((f * window) / frames);
    const b = Math.floor(((f + 1) * window) / frames);
    const last = spans.at(-1);
    if (last && last[1] === a) last[1] = b;
    else spans.push([a, b]);
  }
  return spans;
}

function normalize(v: Float64Array): Float64Array {
  let n = 0;
  for (const x of v) n += x * x;
  n = Math.sqrt(n) || 1;
  return v.map((x) => x / n);
}

function dot(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] ?? 0) * (b[i] ?? 0);
  return s;
}

function meanOf(rows: readonly Float64Array[], dim: number): Float64Array {
  const m = new Float64Array(dim);
  for (const r of rows) for (let i = 0; i < dim; i++) m[i] = (m[i] ?? 0) + (r[i] ?? 0);
  return m.map((x) => x / Math.max(rows.length, 1));
}

/** Average-linkage agglomerative clustering on cosine distance, stopped at `k` clusters. */
function averageLinkage(x: readonly Float64Array[], k: number): number[] {
  const n = x.length;
  const d = new Float64Array(n * n);
  const norms = x.map((v) => Math.sqrt(dot(v, v)) || 1);
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      const v =
        1 - dot(x[i] ?? new Float64Array(), x[j] ?? new Float64Array()) / ((norms[i] ?? 1) * (norms[j] ?? 1));
      d[i * n + j] = v;
      d[j * n + i] = v;
    }
  const size = Array.from({ length: n }, () => 1);
  const alive = Array.from({ length: n }, () => true);
  const label = Array.from({ length: n }, (_, i) => i);
  const nearest = new Int32Array(n);
  const nearestD = new Float64Array(n);
  const refresh = (i: number) => {
    let best = -1;
    let bd = Infinity;
    for (let j = 0; j < n; j++)
      if (j !== i && alive[j] && (d[i * n + j] ?? Infinity) < bd) {
        bd = d[i * n + j] ?? Infinity;
        best = j;
      }
    nearest[i] = best;
    nearestD[i] = bd;
  };
  for (let i = 0; i < n; i++) refresh(i);
  for (let clusters = n; clusters > k; clusters--) {
    let a = -1;
    for (let i = 0; i < n; i++) if (alive[i] && (a < 0 || (nearestD[i] ?? 0) < (nearestD[a] ?? 0))) a = i;
    const b = nearest[a] ?? -1;
    if (a < 0 || b < 0) break;
    const na = size[a] ?? 1;
    const nb = size[b] ?? 1;
    alive[b] = false;
    size[a] = na + nb;
    for (let i = 0; i < n; i++) if (label[i] === b) label[i] = a;
    for (let j = 0; j < n; j++) {
      if (!alive[j] || j === a) continue;
      const v = (na * (d[a * n + j] ?? 0) + nb * (d[b * n + j] ?? 0)) / (na + nb);
      d[a * n + j] = v;
      d[j * n + a] = v;
    }
    for (let i = 0; i < n; i++) {
      if (!alive[i]) continue;
      if (i === a || nearest[i] === a || nearest[i] === b) refresh(i);
      else if ((d[i * n + a] ?? Infinity) < (nearestD[i] ?? Infinity)) {
        nearest[i] = a;
        nearestD[i] = d[i * n + a] ?? Infinity;
      }
    }
  }
  const ids = [...new Set(label)].toSorted((p, q) => p - q);
  return label.map((l) => ids.indexOf(l));
}

export interface Voices {
  /** Unit centroid per voice. */
  centroids: Float64Array[];
  /** Mean of the unit embeddings: in noise it is a shared channel component, so it is subtracted. */
  mean: Float64Array;
}

/** An embedding at unit length, less `mean`, normalised again: how voices are compared. */
export function centred(e: Float32Array | Float64Array, mean: Float64Array): Float64Array {
  const u = normalize(Float64Array.from(e));
  return normalize(u.map((v, i) => v - (mean[i] ?? 0)));
}

/**
 * Groups embeddings into `k` voices: average-linkage AHC on an even sample, then spherical k-means
 * over all of them. AHC alone tends to leave one cluster holding a few outliers while two real voices
 * share another, so tiny clusters are reseeded with farthest points first. Voices come in order of the
 * first embedding nearest each.
 */
export function clusterVoices(embeddings: readonly Float32Array[], k: number, iterations = 20): Voices {
  const dim = embeddings[0]?.length ?? 0;
  const mean = meanOf(
    embeddings.map((e) => normalize(Float64Array.from(e))),
    dim,
  );
  const x = embeddings.map((e) => centred(e, mean));
  const m = Math.min(x.length, DIARIZATION.maxLinkage);
  const pick = Array.from({ length: m }, (_, i) =>
    m === 1 ? 0 : Math.round((i * (x.length - 1)) / (m - 1)),
  );
  const sample = pick.map((i) => x[i] ?? new Float64Array(dim));
  const labels = averageLinkage(sample, k);
  const sizes = Array.from({ length: k }, () => 0);
  for (const l of labels) sizes[l] = (sizes[l] ?? 0) + 1;
  let centroids: Float64Array[];
  if (Math.min(...sizes) < 0.05 * sample.length) {
    const biggest = sizes.indexOf(Math.max(...sizes));
    centroids = [
      normalize(
        meanOf(
          sample.filter((_, i) => labels[i] === biggest),
          dim,
        ),
      ),
    ];
    while (centroids.length < k) {
      let far = 0;
      let farSim = Infinity;
      for (const [i, v] of x.entries()) {
        const s = Math.max(...centroids.map((c) => dot(v, c)));
        if (s < farSim) {
          farSim = s;
          far = i;
        }
      }
      centroids.push(x[far] ?? new Float64Array(dim));
    }
  } else {
    centroids = Array.from({ length: k }, (_, c) =>
      meanOf(
        sample.filter((__, i) => labels[i] === c),
        dim,
      ),
    );
  }
  for (let it = 0; it < iterations; it++) {
    centroids = centroids.map(normalize);
    const members: Float64Array[][] = centroids.map(() => []);
    for (const v of x) {
      let best = 0;
      let bs = -Infinity;
      for (const [c, cen] of centroids.entries()) {
        const s = dot(v, cen);
        if (s > bs) {
          bs = s;
          best = c;
        }
      }
      members[best]?.push(v);
    }
    centroids = centroids.map((c, i) => {
      const group = members[i] ?? [];
      return group.length > 0 ? meanOf(group, dim) : c;
    });
  }
  const unit = centroids.map(normalize);
  const nearest = x.map((v) => {
    const sims = unit.map((c) => dot(v, c));
    return sims.indexOf(Math.max(...sims));
  });
  const first = unit.map((_, c) => {
    const i = nearest.indexOf(c);
    return i < 0 ? x.length : i;
  });
  const order = unit.map((_, c) => c).toSorted((a, b) => (first[a] ?? 0) - (first[b] ?? 0));
  return { centroids: order.map((c) => unit[c] ?? new Float64Array(dim)), mean };
}

function permutations(n: number, r: number): number[][] {
  if (r === 0) return [[]];
  const out: number[][] = [];
  for (let i = 0; i < n; i++)
    for (const rest of permutations(n, r - 1)) if (!rest.includes(i)) out.push([i, ...rest]);
  return out;
}

/** Pairs rows with distinct columns for the highest total; exhaustive, as both sides are at most 3. */
export function bestAssignment(sim: readonly number[][], cols: number): [number, number][] {
  const rows = sim.length;
  let best = -Infinity;
  let pairs: [number, number][] = [];
  if (rows <= cols) {
    for (const p of permutations(cols, rows)) {
      const s = p.reduce((acc, c, r) => acc + (sim[r]?.[c] ?? 0), 0);
      if (s > best) {
        best = s;
        pairs = p.map((c, r) => [r, c]);
      }
    }
  } else {
    for (const p of permutations(rows, cols)) {
      const s = p.reduce((acc, r, c) => acc + (sim[r]?.[c] ?? 0), 0);
      if (s > best) {
        best = s;
        pairs = p.map((r, c): [number, number] => [r, c]).toSorted((a, b) => a[0] - b[0]);
      }
    }
  }
  return pairs;
}

export interface Window {
  start: number;
  /** Per-frame bit masks of the window's local speakers. */
  masks: Uint8Array;
  /** Embedding of each local speaker that talked alone for long enough. */
  embeddings: Map<number, Float32Array>;
}

/**
 * Who speaks on the global frame grid: one 0/1 mask per voice. Each window's local speakers are
 * matched to voices by embedding, the windows' votes averaged per frame, and the frame keeps as many
 * of the top-voted voices as the windows say people were talking.
 */
export function speakerActivity(windows: readonly Window[], voices: Voices, total: number): Uint8Array[] {
  const k = voices.centroids.length;
  const frames = Math.floor(total / DIARIZATION.frame) + 1;
  const votes = Array.from({ length: k }, () => new Float64Array(frames));
  const covered = new Float64Array(frames);
  const talking = new Float64Array(frames);
  for (const w of windows) {
    const f0 = Math.floor(w.start / DIARIZATION.frame);
    const m = Math.min(w.masks.length, frames - f0);
    for (let f = 0; f < m; f++) {
      const bits = w.masks[f] ?? 0;
      covered[f0 + f] = (covered[f0 + f] ?? 0) + 1;
      talking[f0 + f] = (talking[f0 + f] ?? 0) + (bits & 1) + ((bits >> 1) & 1) + ((bits >> 2) & 1);
    }
    const locals = [...w.embeddings.keys()].toSorted((a, b) => a - b);
    if (locals.length === 0) continue;
    const sim = locals.map((j) => {
      const e = centred(w.embeddings.get(j) ?? new Float32Array(), voices.mean);
      return voices.centroids.map((c) => dot(e, c));
    });
    for (const [r, c] of bestAssignment(sim, k)) {
      const bit = 1 << (locals[r] ?? 0);
      const v = votes[c];
      if (!v) continue;
      for (let f = 0; f < m; f++) if ((w.masks[f] ?? 0) & bit) v[f0 + f] = (v[f0 + f] ?? 0) + 1;
    }
  }
  const active = Array.from({ length: k }, () => new Uint8Array(frames));
  const order = Array.from({ length: k }, (_, i) => i);
  for (let t = 0; t < frames; t++) {
    const n = Math.max(covered[t] ?? 0, 1);
    const count = Math.floor((talking[t] ?? 0) / n + 0.5);
    const score = (s: number) => (votes[s]?.[t] ?? 0) / n;
    const ranked = order.toSorted((a, b) => score(b) - score(a));
    for (let r = 0; r < Math.min(count, k); r++) {
      const s = ranked[r] ?? 0;
      if (score(s) > 0) (active[s] ?? new Uint8Array(frames))[t] = 1;
    }
  }
  return active;
}
