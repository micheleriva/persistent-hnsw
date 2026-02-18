import { assertEquals } from 'https://deno.land/std@0.224.0/assert/assert_equals.ts'
import { assertAlmostEquals } from 'https://deno.land/std@0.224.0/assert/assert_almost_equals.ts'
import { computeNorm, cosine, euclidean, getDistanceFunction, innerProduct } from '../src/distances.ts'

Deno.test('euclidean: zero distance for identical vectors', () => {
  const a = new Float32Array([1, 2, 3, 4])
  assertEquals(euclidean(a, a), 0)
})

Deno.test('euclidean: known distance', () => {
  const a = new Float32Array([0, 0, 0])
  const b = new Float32Array([3, 4, 0])
  // L2 squared = 9 + 16 = 25
  assertEquals(euclidean(a, b), 25)
})

Deno.test('euclidean: works with non-multiple-of-4 dimensions', () => {
  const a = new Float32Array([1, 2, 3, 4, 5])
  const b = new Float32Array([5, 4, 3, 2, 1])
  // (4^2 + 2^2 + 0 + 2^2 + 4^2) = 16+4+0+4+16 = 40
  assertEquals(euclidean(a, b), 40)
})

Deno.test('cosine: zero distance for identical vectors', () => {
  const a = new Float32Array([1, 2, 3, 4])
  assertAlmostEquals(cosine(a, a), 0, 1e-6)
})

Deno.test('cosine: orthogonal vectors have distance 1', () => {
  const a = new Float32Array([1, 0, 0, 0])
  const b = new Float32Array([0, 1, 0, 0])
  assertAlmostEquals(cosine(a, b), 1, 1e-6)
})

Deno.test('cosine: opposite vectors have distance 2', () => {
  const a = new Float32Array([1, 0])
  const b = new Float32Array([-1, 0])
  assertAlmostEquals(cosine(a, b), 2, 1e-6)
})

Deno.test('cosine: handles zero vector', () => {
  const a = new Float32Array([0, 0, 0])
  const b = new Float32Array([1, 2, 3])
  assertEquals(cosine(a, b), 1)
})

Deno.test('innerProduct: negated dot product', () => {
  const a = new Float32Array([1, 2, 3, 4])
  const b = new Float32Array([4, 3, 2, 1])
  // dot = 4+6+6+4 = 20, negated = -20
  assertEquals(innerProduct(a, b), -20)
})

Deno.test('innerProduct: works with non-multiple-of-4 dimensions', () => {
  const a = new Float32Array([1, 2, 3])
  const b = new Float32Array([4, 5, 6])
  // dot = 4+10+18 = 32
  assertEquals(innerProduct(a, b), -32)
})

Deno.test('computeNorm: known norm', () => {
  const v = new Float32Array([3, 4])
  assertEquals(computeNorm(v), 5)
})

Deno.test('computeNorm: unit vector', () => {
  const v = new Float32Array([1, 0, 0, 0])
  assertEquals(computeNorm(v), 1)
})

Deno.test('getDistanceFunction: returns correct functions', () => {
  assertEquals(getDistanceFunction('euclidean'), euclidean)
  assertEquals(getDistanceFunction('cosine'), cosine)
  assertEquals(getDistanceFunction('inner_product'), innerProduct)
})

Deno.test('euclidean: large vector (128 dims)', () => {
  const a = new Float32Array(128).fill(1)
  const b = new Float32Array(128).fill(2)
  // Each dim contributes 1, so sum = 128
  assertEquals(euclidean(a, b), 128)
})
