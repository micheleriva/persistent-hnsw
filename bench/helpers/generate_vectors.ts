/** Generate random Float32Array vectors with a simple seeded PRNG. */
export function generateVectors(
  count: number,
  dimensions: number,
  seed = 12345,
): Float32Array[] {
  let s = seed;
  function rnd(): number {
    s = (s * 1664525 + 1013904223) & 0xFFFFFFFF;
    return (s >>> 0) / 0xFFFFFFFF;
  }

  const vectors: Float32Array[] = [];
  for (let i = 0; i < count; i++) {
    const v = new Float32Array(dimensions);
    for (let d = 0; d < dimensions; d++) {
      v[d] = rnd() * 2 - 1;
    }
    vectors.push(v);
  }
  return vectors;
}
