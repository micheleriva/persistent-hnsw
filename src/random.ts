/**
 * Seedable xoshiro128** PRNG.
 * Fast, good distribution, deterministic.
 */
export function createRng(seed?: number): () => number {
  // Initialize state from seed using splitmix32
  let s = (seed ?? (Math.random() * 0xFFFFFFFF)) >>> 0;

  function splitmix32(): number {
    s = (s + 0x9E3779B9) >>> 0;
    let z = s;
    z = (z ^ (z >>> 16)) >>> 0;
    z = Math.imul(z, 0x85EBCA6B) >>> 0;
    z = (z ^ (z >>> 13)) >>> 0;
    z = Math.imul(z, 0xC2B2AE35) >>> 0;
    z = (z ^ (z >>> 16)) >>> 0;
    return z;
  }

  let s0 = splitmix32();
  let s1 = splitmix32();
  let s2 = splitmix32();
  let s3 = splitmix32();

  // Ensure at least one state bit is set
  if ((s0 | s1 | s2 | s3) === 0) s0 = 1;

  return function xoshiro128ss(): number {
    const result = Math.imul(rotl(Math.imul(s1, 5), 7), 9) >>> 0;
    const t = (s1 << 9) >>> 0;

    s2 = (s2 ^ s0) >>> 0;
    s3 = (s3 ^ s1) >>> 0;
    s1 = (s1 ^ s2) >>> 0;
    s0 = (s0 ^ s3) >>> 0;
    s2 = (s2 ^ t) >>> 0;
    s3 = rotl(s3, 11);

    // Return value in (0, 1) — never exactly 0 or 1
    return (result >>> 1) / 0x80000000;
  };
}

function rotl(x: number, k: number): number {
  return ((x << k) | (x >>> (32 - k))) >>> 0;
}
