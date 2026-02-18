/**
 * Portable benchmark — works with Node, Bun, and Deno.
 *
 *   node tests/node/bench.mjs
 *   bun  tests/node/bench.mjs
 *   deno run --allow-read tests/node/bench.mjs
 *
 * Outputs JSON to stdout. Requires `deno task build` first.
 */

import { HNSWIndex, defaultHNSWConfig } from "../../.npm/esm/mod.js";

const runtime =
  typeof Deno !== "undefined" ? `Deno ${Deno.version.deno}` :
  typeof Bun  !== "undefined" ? `Bun ${Bun.version}` :
  `Node ${process.version}`;

function generateVectors(count, dim, seed = 12345) {
  let s = seed;
  const rnd = () => { s = (s * 1664525 + 1013904223) & 0xFFFFFFFF; return (s >>> 0) / 0xFFFFFFFF; };
  return Array.from({ length: count }, () => {
    const v = new Float32Array(dim);
    for (let d = 0; d < dim; d++) v[d] = rnd() * 2 - 1;
    return v;
  });
}

function euclidean(a, b) {
  const len = a.length;
  let sum = 0, i = 0;
  const limit = len - 3;
  for (; i < limit; i += 4) {
    const d0 = a[i] - b[i], d1 = a[i+1] - b[i+1], d2 = a[i+2] - b[i+2], d3 = a[i+3] - b[i+3];
    sum += d0*d0 + d1*d1 + d2*d2 + d3*d3;
  }
  for (; i < len; i++) { const d = a[i] - b[i]; sum += d*d; }
  return sum;
}

const results = { runtime, insertion: {}, search: {}, distance: {} };

// ── Insertion ──────────────────────────────────────────────

for (const n of [1_000, 5_000, 10_000]) {
  const dim = 128;
  const vecs = generateVectors(n, dim);
  const config = defaultHNSWConfig(dim, { seed: 42, M: 16, efConstruction: 200 });
  const index = new HNSWIndex(config);

  const t0 = performance.now();
  for (let i = 0; i < n; i++) index.insert("v" + i, vecs[i]);
  const ms = performance.now() - t0;

  results.insertion[n] = { opsPerSec: Math.round(n / ms * 1000), timeMs: Math.round(ms) };
}

// ── Search ─────────────────────────────────────────────────

{
  const dim = 128, n = 10_000;
  const config = defaultHNSWConfig(dim, { seed: 42, M: 16, efConstruction: 200 });
  const index = new HNSWIndex(config);
  const vecs = generateVectors(n, dim);
  for (let i = 0; i < n; i++) index.insert("v" + i, vecs[i]);

  const queries = generateVectors(200, dim, 99999);

  for (const ef of [10, 50, 100, 200, 400]) {
    for (let i = 0; i < 100; i++) index.search(queries[i % 200], 10, ef);

    const iters = 1000;
    const t0 = performance.now();
    for (let i = 0; i < iters; i++) index.search(queries[i % 200], 10, ef);
    const ms = performance.now() - t0;
    const usPerQuery = (ms / iters) * 1000;

    results.search[ef] = { usPerQuery: Math.round(usPerQuery), qps: Math.round(1e6 / usPerQuery) };
  }
}

// ── Distance ───────────────────────────────────────────────

for (const dim of [128, 256, 768, 1536]) {
  const [a, b] = generateVectors(2, dim);
  const iters = dim <= 256 ? 2_000_000 : 500_000;
  const t0 = performance.now();
  for (let i = 0; i < iters; i++) euclidean(a, b);
  const ops = iters / (performance.now() - t0) * 1000;

  results.distance[dim] = { opsPerSec: Math.round(ops) };
}

console.log(JSON.stringify(results));
