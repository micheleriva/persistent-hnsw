import type { DistanceFunction, DistanceMetric, Vector } from "./types.ts";

/**
 * Squared Euclidean (L2²) distance.
 * 4-wide unrolled loop for V8 optimization.
 */
export function euclidean(a: Vector, b: Vector): number {
  const len = a.length;
  let sum = 0;
  let i = 0;

  // 4-wide unrolled main loop
  const limit = len - 3;
  for (; i < limit; i += 4) {
    const d0 = a[i] - b[i];
    const d1 = a[i + 1] - b[i + 1];
    const d2 = a[i + 2] - b[i + 2];
    const d3 = a[i + 3] - b[i + 3];
    sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
  }

  // Handle remainder
  for (; i < len; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }

  return sum;
}

/**
 * Cosine distance: 1 - cos(a, b).
 * Single-pass computes dot product and both norms simultaneously.
 * 4-wide unrolled loop.
 */
export function cosine(a: Vector, b: Vector): number {
  const len = a.length;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  let i = 0;

  const limit = len - 3;
  for (; i < limit; i += 4) {
    const a0 = a[i], a1 = a[i + 1], a2 = a[i + 2], a3 = a[i + 3];
    const b0 = b[i], b1 = b[i + 1], b2 = b[i + 2], b3 = b[i + 3];
    dot += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
    normA += a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
    normB += b0 * b0 + b1 * b1 + b2 * b2 + b3 * b3;
  }

  for (; i < len; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const denom = Math.sqrt(normA * normB);
  if (denom === 0) return 1;
  return 1 - dot / denom;
}

/**
 * Inner product distance: -dot(a, b).
 * Negated so lower = more similar (consistent with other metrics).
 * 4-wide unrolled loop.
 */
export function innerProduct(a: Vector, b: Vector): number {
  const len = a.length;
  let dot = 0;
  let i = 0;

  const limit = len - 3;
  for (; i < limit; i += 4) {
    dot += a[i] * b[i] + a[i + 1] * b[i + 1] +
      a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3];
  }

  for (; i < len; i++) {
    dot += a[i] * b[i];
  }

  return -dot;
}

/**
 * Compute the L2 norm of a vector. Used for the norm cache.
 * 4-wide unrolled.
 */
export function computeNorm(v: Vector): number {
  const len = v.length;
  let sum = 0;
  let i = 0;

  const limit = len - 3;
  for (; i < limit; i += 4) {
    sum += v[i] * v[i] + v[i + 1] * v[i + 1] +
      v[i + 2] * v[i + 2] + v[i + 3] * v[i + 3];
  }

  for (; i < len; i++) {
    sum += v[i] * v[i];
  }

  return Math.sqrt(sum);
}

/** Returns the distance function for the given metric. */
export function getDistanceFunction(metric: DistanceMetric): DistanceFunction {
  switch (metric) {
    case "euclidean":
      return euclidean;
    case "cosine":
      return cosine;
    case "inner_product":
      return innerProduct;
    default:
      throw new Error(`Unknown distance metric: ${metric}`);
  }
}
