import { assert } from "https://deno.land/std@0.224.0/assert/assert.ts";
import { HNSWIndex } from "../src/hnsw_index.ts";
import { defaultHNSWConfig } from "../src/types.ts";
import { euclidean } from "../src/distances.ts";

/**
 * Brute-force k-nearest-neighbor search for ground truth.
 */
function bruteForceKNN(
  vectors: Float32Array[],
  query: Float32Array,
  k: number,
): { index: number; distance: number }[] {
  const distances = vectors.map((v, i) => ({
    index: i,
    distance: euclidean(query, v),
  }));
  distances.sort((a, b) => a.distance - b.distance);
  return distances.slice(0, k);
}

/**
 * Compute recall@k: fraction of true k-nearest found by HNSW.
 */
function computeRecall(
  hnswResults: string[],
  bruteForceResults: number[],
  idMap: Map<number, string>,
): number {
  const trueSet = new Set(bruteForceResults.map((i) => idMap.get(i)!));
  let hits = 0;
  for (const id of hnswResults) {
    if (trueSet.has(id)) hits++;
  }
  return hits / trueSet.size;
}

Deno.test("Recall@10 >= 0.95 on 10K random 128-dim vectors", () => {
  const n = 10_000;
  const dim = 128;
  const k = 10;
  const numQueries = 100;

  const config = defaultHNSWConfig(dim, {
    seed: 42,
    M: 16,
    efConstruction: 200,
    efSearch: 200,
  });

  const index = new HNSWIndex(config);

  // Generate random vectors with a seeded RNG for reproducibility
  let seed = 12345;
  function nextRand(): number {
    seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF;
    return (seed >>> 0) / 0xFFFFFFFF;
  }

  const vectors: Float32Array[] = [];
  const idMap = new Map<number, string>();

  for (let i = 0; i < n; i++) {
    const vec = new Float32Array(dim);
    for (let d = 0; d < dim; d++) {
      vec[d] = nextRand() * 2 - 1;
    }
    vectors.push(vec);
    const id = `v${i}`;
    idMap.set(i, id);
    index.insert(id, vec);
  }

  // Run queries and measure recall
  let totalRecall = 0;

  for (let q = 0; q < numQueries; q++) {
    const query = new Float32Array(dim);
    for (let d = 0; d < dim; d++) {
      query[d] = nextRand() * 2 - 1;
    }

    const hnswResults = index.search(query, k);
    const bfResults = bruteForceKNN(vectors, query, k);

    const recall = computeRecall(
      hnswResults.map((r) => r.id),
      bfResults.map((r) => r.index),
      idMap,
    );
    totalRecall += recall;
  }

  const avgRecall = totalRecall / numQueries;
  console.log(`  Average recall@${k}: ${(avgRecall * 100).toFixed(2)}%`);

  assert(
    avgRecall >= 0.95,
    `Recall@${k} is ${(avgRecall * 100).toFixed(2)}%, expected >= 95%`,
  );
});
