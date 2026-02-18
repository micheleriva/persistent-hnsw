/**
 * Comprehensive HNSW benchmark suite.
 * Measures insertion throughput, search latency, recall, and memory across dataset sizes.
 *
 * Run: deno run --allow-write bench/full_benchmark.ts
 */

import { HNSWIndex } from "../src/hnsw_index.ts";
import { defaultHNSWConfig } from "../src/types.ts";
import { euclidean } from "../src/distances.ts";
import { decodeShard, encodeShard } from "../src/storage/serialization.ts";

// ─── Helpers ────────────────────────────────────────────────────────────

function generateVectors(
  count: number,
  dim: number,
  seed = 12345,
): Float32Array[] {
  let s = seed;
  function rnd(): number {
    s = (s * 1664525 + 1013904223) & 0xFFFFFFFF;
    return (s >>> 0) / 0xFFFFFFFF;
  }
  const vecs: Float32Array[] = [];
  for (let i = 0; i < count; i++) {
    const v = new Float32Array(dim);
    for (let d = 0; d < dim; d++) v[d] = rnd() * 2 - 1;
    vecs.push(v);
  }
  return vecs;
}

function bruteForceKNN(
  vectors: Float32Array[],
  query: Float32Array,
  k: number,
): Set<number> {
  const dists = vectors.map((v, i) => ({ i, d: euclidean(query, v) }));
  dists.sort((a, b) => a.d - b.d);
  return new Set(dists.slice(0, k).map((x) => x.i));
}

function median(arr: number[]): number {
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0
    ? sorted[mid]
    : (sorted[mid - 1] + sorted[mid]) / 2;
}

function p99(arr: number[]): number {
  const sorted = [...arr].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length * 0.99)];
}

function formatNum(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return n.toFixed(0);
}

function padRight(s: string, len: number): string {
  return s + " ".repeat(Math.max(0, len - s.length));
}

function padLeft(s: string, len: number): string {
  return " ".repeat(Math.max(0, len - s.length)) + s;
}

/**
 * Batch-measure search latency: run many searches, divide total time.
 * This avoids timer resolution issues with sub-millisecond operations.
 */
function measureSearchLatency(
  index: HNSWIndex,
  queries: Float32Array[],
  k: number,
  efSearch: number,
): { medianUs: number; p99Us: number } {
  const nq = queries.length;

  // Always use batch measurement for reliable sub-ms timing.
  // Run several batches of 100 queries each, compute average per query per batch.
  const batchSize = 100;
  const numBatches = 10;

  // Warmup
  for (let i = 0; i < batchSize; i++) {
    index.search(queries[i % nq], k, efSearch);
  }

  const batchAvgs: number[] = [];
  for (let b = 0; b < numBatches; b++) {
    const start = performance.now();
    for (let i = 0; i < batchSize; i++) {
      index.search(queries[(b * batchSize + i) % nq], k, efSearch);
    }
    const elapsedUs = (performance.now() - start) * 1000;
    batchAvgs.push(elapsedUs / batchSize);
  }

  return { medianUs: median(batchAvgs), p99Us: p99(batchAvgs) };
}

// ─── Renderers ──────────────────────────────────────────────────────────

function barChart(
  title: string,
  data: { label: string; value: number }[],
  opts: { unit: string; maxWidth?: number; precision?: number } = { unit: "" },
): string {
  const maxWidth = opts.maxWidth ?? 40;
  const precision = opts.precision ?? 1;
  const maxVal = Math.max(...data.map((d) => d.value));
  const maxLabel = Math.max(...data.map((d) => d.label.length));
  const lines: string[] = [];
  lines.push("");
  lines.push(`### ${title}`);
  lines.push("```");
  for (const d of data) {
    const barLen = Math.max(1, Math.round((d.value / maxVal) * maxWidth));
    const bar = "█".repeat(barLen);
    const valStr = d.value >= 1000
      ? formatNum(d.value)
      : d.value.toFixed(precision);
    lines.push(`${padRight(d.label, maxLabel)} │${bar} ${valStr} ${opts.unit}`);
  }
  lines.push("```");
  return lines.join("\n");
}

function table(headers: string[], rows: string[][]): string {
  const colWidths = headers.map((h, i) =>
    Math.max(h.length, ...rows.map((r) => (r[i] ?? "").length))
  );
  const lines: string[] = [];
  lines.push(
    `| ${headers.map((h, i) => padRight(h, colWidths[i])).join(" | ")} |`,
  );
  lines.push(`| ${colWidths.map((w) => "-".repeat(w)).join(" | ")} |`);
  for (const row of rows) {
    lines.push(
      `| ${row.map((c, i) => padLeft(c, colWidths[i])).join(" | ")} |`,
    );
  }
  return lines.join("\n");
}

// ─── Main ───────────────────────────────────────────────────────────────

console.log("HNSW Benchmark Suite");
console.log("====================\n");

const output: string[] = [];
output.push("# HNSW Benchmark Results\n");
output.push(
  `> Benchmarked on ${
    new Date().toISOString().split("T")[0]
  } — Deno ${Deno.version.deno}, ${Deno.build.arch}-${Deno.build.os}\n`,
);

// ─── 1. Dataset scaling ─────────────────────────────────────────────────

console.log("▸ Scaling benchmark (1K → 100K vectors)...");

interface ScaleResult {
  n: number;
  insertTimeMs: number;
  insertOpsPerSec: number;
  searchMedianUs: number;
  searchP99Us: number;
  searchOpsPerSec: number;
  recall200: number;
  recall500: number;
  memoryMB: number;
  serializedMB: number;
}

const scaleResults: ScaleResult[] = [];
const scaleSizes = [1_000, 5_000, 10_000, 25_000, 50_000];
const dim = 128;
const k = 10;

for (const n of scaleSizes) {
  console.log(`  ${formatNum(n)} × ${dim}d...`);
  const config = defaultHNSWConfig(dim, {
    seed: 42,
    M: 16,
    efConstruction: 200,
    efSearch: 200,
  });
  const index = new HNSWIndex(config);
  const vectors = generateVectors(n, dim);

  // Insertion
  const t0 = performance.now();
  for (let i = 0; i < n; i++) index.insert(`v${i}`, vectors[i]);
  const insertMs = performance.now() - t0;

  // Search latency (at efSearch=200)
  const queries = generateVectors(200, dim, 99999);
  const sl = measureSearchLatency(index, queries, k, 200);

  // Recall — use fewer brute-force queries for large datasets (BF is O(n) per query)
  const recallN = n > 10_000 ? 30 : 50;
  let recall200 = 0, recall500 = 0;
  for (let i = 0; i < recallN; i++) {
    const bfIds = bruteForceKNN(vectors, queries[i], k);
    const ids200 = new Set(index.search(queries[i], k, 200).map((r) => r.id));
    const ids500 = new Set(index.search(queries[i], k, 500).map((r) => r.id));
    let h200 = 0, h500 = 0;
    for (const idx of bfIds) {
      if (ids200.has(`v${idx}`)) h200++;
      if (ids500.has(`v${idx}`)) h500++;
    }
    recall200 += h200 / k;
    recall500 += h500 / k;
  }

  const encoded = encodeShard(index);
  const memoryMB = index.memoryUsage() / (1024 * 1024);
  index.shrinkToFit();
  const memoryShrunkMB = index.memoryUsage() / (1024 * 1024);

  const r: ScaleResult = {
    n,
    insertTimeMs: insertMs,
    insertOpsPerSec: (n / insertMs) * 1000,
    searchMedianUs: sl.medianUs,
    searchP99Us: sl.p99Us,
    searchOpsPerSec: 1_000_000 / sl.medianUs,
    recall200: recall200 / recallN,
    recall500: recall500 / recallN,
    memoryMB: memoryShrunkMB,
    serializedMB: encoded.byteLength / (1024 * 1024),
  };
  scaleResults.push(r);
  console.log(
    `    insert: ${formatNum(r.insertOpsPerSec)}/s  search: ${
      r.searchMedianUs.toFixed(0)
    }µs  recall@ef200: ${(r.recall200 * 100).toFixed(1)}%  recall@ef500: ${
      (r.recall500 * 100).toFixed(1)
    }%  mem: ${r.memoryMB.toFixed(1)}MB`,
  );
}

output.push("## Scaling (128 dimensions, M=16, efConstruction=200)\n");
output.push(table(
  [
    "Vectors",
    "Insert ops/s",
    "Build Time",
    "Search (ef=200)",
    "Search p99",
    "QPS (ef=200)",
    "Recall@ef200",
    "Recall@ef500",
    "Memory",
    "On Disk",
  ],
  scaleResults.map((r) => [
    formatNum(r.n),
    formatNum(r.insertOpsPerSec),
    r.insertTimeMs < 1000
      ? `${r.insertTimeMs.toFixed(0)}ms`
      : `${(r.insertTimeMs / 1000).toFixed(1)}s`,
    `${r.searchMedianUs.toFixed(0)}µs`,
    `${r.searchP99Us.toFixed(0)}µs`,
    formatNum(r.searchOpsPerSec),
    `${(r.recall200 * 100).toFixed(1)}%`,
    `${(r.recall500 * 100).toFixed(1)}%`,
    `${r.memoryMB.toFixed(1)} MB`,
    `${r.serializedMB.toFixed(1)} MB`,
  ]),
));

output.push(barChart(
  "Insertion Throughput",
  scaleResults.map((r) => ({
    label: `${formatNum(r.n)} vectors`,
    value: r.insertOpsPerSec,
  })),
  { unit: "ops/s" },
));

output.push(barChart(
  "Search Latency (median)",
  scaleResults.map((r) => ({
    label: `${formatNum(r.n)} vectors`,
    value: r.searchMedianUs,
  })),
  { unit: "µs", precision: 0 },
));

output.push(barChart(
  "Memory Usage",
  scaleResults.map((r) => ({
    label: `${formatNum(r.n)} vectors`,
    value: r.memoryMB,
  })),
  { unit: "MB" },
));

// ─── 2. efSearch tradeoff ───────────────────────────────────────────────

console.log("\n▸ Recall vs latency tradeoff (10K, 128d)...");
{
  const n = 10_000;
  const config = defaultHNSWConfig(dim, {
    seed: 42,
    M: 16,
    efConstruction: 200,
  });
  const index = new HNSWIndex(config);
  const vectors = generateVectors(n, dim);
  for (let i = 0; i < n; i++) index.insert(`v${i}`, vectors[i]);
  const queries = generateVectors(100, dim, 99999);

  interface EfRow {
    ef: number;
    recall: number;
    medianUs: number;
    opsPerSec: number;
  }
  const efRows: EfRow[] = [];

  for (const ef of [10, 50, 100, 200, 400]) {
    const sl = measureSearchLatency(index, queries, k, ef);
    let recall = 0;
    for (let i = 0; i < 100; i++) {
      const ids = new Set(index.search(queries[i], k, ef).map((r) => r.id));
      const bf = bruteForceKNN(vectors, queries[i], k);
      let hits = 0;
      for (const idx of bf) if (ids.has(`v${idx}`)) hits++;
      recall += hits / k;
    }
    efRows.push({
      ef,
      recall: recall / 100,
      medianUs: sl.medianUs,
      opsPerSec: 1_000_000 / sl.medianUs,
    });
    console.log(
      `  ef=${ef}: recall=${recall.toFixed(1)}%  latency=${
        sl.medianUs.toFixed(0)
      }µs`,
    );
  }

  output.push("\n## Recall vs Latency Tradeoff (10K vectors, 128d, M=16)\n");
  output.push(table(
    ["efSearch", "Recall@10", "Median Latency", "Throughput"],
    efRows.map((r) => [
      String(r.ef),
      `${(r.recall * 100).toFixed(1)}%`,
      `${r.medianUs.toFixed(0)}µs`,
      `${formatNum(r.opsPerSec)} ops/s`,
    ]),
  ));

  output.push("\n### Recall–Latency Curve\n");
  output.push("```");
  output.push("  Recall@10 │");
  const maxLat = Math.max(...efRows.map((r) => r.medianUs));
  const chartW = 56;
  for (let pct = 100; pct >= 40; pct -= 5) {
    let line = `${padLeft(String(pct) + "%", 7)} │`;
    const chars = new Array(chartW).fill(" ");
    for (const r of efRows) {
      const recallPct = r.recall * 100;
      if (Math.abs(recallPct - pct) < 2.5) {
        const col = Math.min(
          chartW - 1,
          Math.round((r.medianUs / maxLat) * (chartW - 1)),
        );
        chars[col] = "●";
      }
    }
    line += chars.join("");
    output.push(line);
  }
  output.push("        └" + "─".repeat(chartW));
  output.push(
    `         ${efRows[0].medianUs.toFixed(0)}µs${" ".repeat(chartW - 12)}${
      efRows[efRows.length - 1].medianUs.toFixed(0)
    }µs`,
  );
  output.push("                     Search Latency →");
  output.push("```");
}

// ─── 3. Dimension scaling — summary + per-dimension details ─────────────

console.log("\n▸ Dimension scaling (5K vectors)...");
{
  const n = 5_000;
  const dimSizes = [32, 64, 128, 256, 384, 512, 768, 1024, 1536];
  interface DimRow {
    dim: number;
    insertOps: number;
    buildTime: number;
    searchOps: number;
    searchMedianUs: number;
    searchP99Us: number;
    recall: number;
    memMB: number;
    memShrunkMB: number;
    serializedMB: number;
  }
  const dimRows: DimRow[] = [];

  for (const d of dimSizes) {
    console.log(`  ${d}d...`);
    const config = defaultHNSWConfig(d, {
      seed: 42,
      M: 16,
      efConstruction: 100,
      efSearch: 100,
    });
    const index = new HNSWIndex(config);
    const vectors = generateVectors(n, d);
    const t0 = performance.now();
    for (let i = 0; i < n; i++) index.insert(`v${i}`, vectors[i]);
    const insertMs = performance.now() - t0;
    const queries = generateVectors(100, d, 99999);
    const sl = measureSearchLatency(index, queries, 10, 100);

    // Recall
    let recall = 0;
    const rn = 30;
    for (let i = 0; i < rn; i++) {
      const ids = new Set(index.search(queries[i], 10, 100).map((r) => r.id));
      const bf = bruteForceKNN(vectors, queries[i], 10);
      let hits = 0;
      for (const idx of bf) if (ids.has(`v${idx}`)) hits++;
      recall += hits / 10;
    }

    const memMB = index.memoryUsage() / (1024 * 1024);
    const encoded = encodeShard(index);
    const serializedMB = encoded.byteLength / (1024 * 1024);
    index.shrinkToFit();
    const memShrunkMB = index.memoryUsage() / (1024 * 1024);

    dimRows.push({
      dim: d,
      insertOps: (n / insertMs) * 1000,
      buildTime: insertMs,
      searchOps: 1_000_000 / sl.medianUs,
      searchMedianUs: sl.medianUs,
      searchP99Us: sl.p99Us,
      recall: recall / rn,
      memMB,
      memShrunkMB,
      serializedMB,
    });
    console.log(
      `    insert ${formatNum((n / insertMs) * 1000)}/s  search ${
        formatNum(1_000_000 / sl.medianUs)
      }/s  mem ${memShrunkMB.toFixed(1)}MB`,
    );
  }

  // Summary table
  output.push(
    "\n## Dimension Scaling (5K vectors, M=16, efConstruction=100, efSearch=100)\n",
  );
  output.push(table(
    [
      "Dimensions",
      "Insert ops/s",
      "Search ops/s",
      "Search Median",
      "Recall@10",
      "Memory",
      "On Disk",
    ],
    dimRows.map((r) => [
      String(r.dim),
      formatNum(r.insertOps),
      formatNum(r.searchOps),
      `${r.searchMedianUs.toFixed(0)}µs`,
      `${(r.recall * 100).toFixed(1)}%`,
      `${r.memShrunkMB.toFixed(1)} MB`,
      `${r.serializedMB.toFixed(1)} MB`,
    ]),
  ));

  output.push(barChart(
    "Search Throughput by Dimension",
    dimRows.map((r) => ({ label: `${r.dim}d`, value: r.searchOps })),
    { unit: "ops/s" },
  ));

  output.push(barChart(
    "Memory by Dimension (after shrinkToFit)",
    dimRows.map((r) => ({ label: `${r.dim}d`, value: r.memShrunkMB })),
    { unit: "MB" },
  ));

  // Per-dimension detail dropdowns
  output.push("\n### Per-Dimension Details\n");
  for (const r of dimRows) {
    output.push(`<details>`);
    output.push(
      `<summary><b>${r.dim} dimensions</b> — ${
        formatNum(r.searchOps)
      } search/s, ${r.memShrunkMB.toFixed(1)} MB</summary>\n`,
    );
    output.push(table(
      ["Metric", "Value"],
      [
        ["Vectors", String(n)],
        ["Dimensions", String(r.dim)],
        ["Insert throughput", `${formatNum(r.insertOps)} ops/s`],
        [
          "Build time",
          r.buildTime < 1000
            ? `${r.buildTime.toFixed(0)}ms`
            : `${(r.buildTime / 1000).toFixed(1)}s`,
        ],
        ["Search median latency", `${r.searchMedianUs.toFixed(0)}µs`],
        ["Search p99 latency", `${r.searchP99Us.toFixed(0)}µs`],
        ["Search throughput", `${formatNum(r.searchOps)} ops/s`],
        ["Recall@10 (ef=100)", `${(r.recall * 100).toFixed(1)}%`],
        ["Memory (live)", `${r.memMB.toFixed(1)} MB`],
        ["Memory (after shrinkToFit)", `${r.memShrunkMB.toFixed(1)} MB`],
        ["Serialized size", `${r.serializedMB.toFixed(1)} MB`],
        [
          "Bytes per vector (memory)",
          `${(r.memShrunkMB * 1024 * 1024 / n).toFixed(0)} B`,
        ],
        [
          "Bytes per vector (disk)",
          `${(r.serializedMB * 1024 * 1024 / n).toFixed(0)} B`,
        ],
      ],
    ));
    output.push("\n</details>\n");
  }
}

// ─── 4. Distance throughput ─────────────────────────────────────────────

console.log("\n▸ Distance function throughput...");
{
  const distDims = [128, 256, 768, 1536];
  const distRows: { label: string; value: number }[] = [];

  for (const d of distDims) {
    const [a, b] = generateVectors(2, d);
    const iters = d <= 256 ? 2_000_000 : 500_000;
    const t0 = performance.now();
    for (let i = 0; i < iters; i++) euclidean(a, b);
    const ops = (iters / (performance.now() - t0)) * 1000;
    distRows.push({ label: `${d}d`, value: ops });
    console.log(`  euclidean ${d}d: ${formatNum(ops)} ops/s`);
  }

  output.push("\n## Distance Function Throughput (Euclidean L2)\n");
  output.push(barChart(
    "Euclidean Distance ops/sec",
    distRows,
    { unit: "ops/s" },
  ));
}

// ─── 5. Serialization ──────────────────────────────────────────────────

console.log("\n▸ Serialization throughput...");
{
  output.push("\n## Serialization\n");
  const serRows: string[][] = [];

  for (const n of [1_000, 10_000, 25_000]) {
    const config = defaultHNSWConfig(128, {
      seed: 42,
      M: 16,
      efConstruction: 100,
    });
    const index = new HNSWIndex(config);
    const vectors = generateVectors(n, 128);
    for (let i = 0; i < n; i++) index.insert(`v${i}`, vectors[i]);

    const runs = n <= 10_000 ? 5 : 2;
    let encMs = 0;
    let encoded!: Uint8Array;
    for (let r = 0; r < runs; r++) {
      const t = performance.now();
      encoded = encodeShard(index);
      encMs += performance.now() - t;
    }
    encMs /= runs;
    const sizeMB = encoded.byteLength / (1024 * 1024);

    let decMs = 0;
    for (let r = 0; r < runs; r++) {
      const t = performance.now();
      decodeShard(encoded);
      decMs += performance.now() - t;
    }
    decMs /= runs;

    serRows.push([
      formatNum(n),
      `${sizeMB.toFixed(1)} MB`,
      `${encMs.toFixed(0)}ms`,
      `${(sizeMB / (encMs / 1000)).toFixed(0)} MB/s`,
      `${decMs.toFixed(0)}ms`,
      `${(sizeMB / (decMs / 1000)).toFixed(0)} MB/s`,
    ]);
    console.log(
      `  ${formatNum(n)}: ${sizeMB.toFixed(1)} MB  enc ${
        encMs.toFixed(0)
      }ms  dec ${decMs.toFixed(0)}ms`,
    );
  }

  output.push(table(
    [
      "Vectors",
      "Size",
      "Encode Time",
      "Encode Speed",
      "Decode Time",
      "Decode Speed",
    ],
    serRows,
  ));
}

// ─── 6. M parameter impact ──────────────────────────────────────────────

console.log("\n▸ M parameter impact (10K, 128d)...");
{
  const n = 10_000;
  const mValues = [4, 8, 16, 32];
  interface MRow {
    M: number;
    insertOps: number;
    searchOps: number;
    recall: number;
    memMB: number;
  }
  const mRows: MRow[] = [];

  for (const M of mValues) {
    const config = defaultHNSWConfig(dim, {
      seed: 42,
      M,
      efConstruction: 200,
      efSearch: 200,
    });
    const index = new HNSWIndex(config);
    const vectors = generateVectors(n, dim);
    const t0 = performance.now();
    for (let i = 0; i < n; i++) index.insert(`v${i}`, vectors[i]);
    const insertMs = performance.now() - t0;

    const queries = generateVectors(50, dim, 99999);
    const sl = measureSearchLatency(index, queries, 10, 200);
    let recall = 0;
    for (let i = 0; i < 50; i++) {
      const ids = new Set(index.search(queries[i], 10, 200).map((r) => r.id));
      const bf = bruteForceKNN(vectors, queries[i], 10);
      let hits = 0;
      for (const idx of bf) if (ids.has(`v${idx}`)) hits++;
      recall += hits / 10;
    }

    index.shrinkToFit();
    mRows.push({
      M,
      insertOps: (n / insertMs) * 1000,
      searchOps: 1_000_000 / sl.medianUs,
      recall: recall / 50,
      memMB: index.memoryUsage() / (1024 * 1024),
    });
    console.log(
      `  M=${M}: recall=${(recall / 50 * 100).toFixed(1)}%  search=${
        formatNum(1_000_000 / sl.medianUs)
      }/s  mem=${(index.memoryUsage() / (1024 * 1024)).toFixed(1)}MB`,
    );
  }

  output.push(
    "\n## Impact of M Parameter (10K vectors, 128d, efConstruction=200, efSearch=200)\n",
  );
  output.push(table(
    ["M", "Insert ops/s", "Search ops/s", "Recall@10", "Memory"],
    mRows.map((r) => [
      String(r.M),
      formatNum(r.insertOps),
      formatNum(r.searchOps),
      `${(r.recall * 100).toFixed(1)}%`,
      `${r.memMB.toFixed(1)} MB`,
    ]),
  ));
}

// ─── Write output ───────────────────────────────────────────────────────

const md = output.join("\n") + "\n";
await Deno.writeTextFile("bench/RESULTS.md", md);
console.log("\n" + "=".repeat(60));
console.log("Written to bench/RESULTS.md\n");
console.log(md);
