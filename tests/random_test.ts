import { assertEquals } from 'https://deno.land/std@0.224.0/assert/assert_equals.ts'
import { assert } from 'https://deno.land/std@0.224.0/assert/assert.ts'
import { createRng } from '../src/random.ts'

Deno.test('createRng: deterministic with same seed', () => {
  const rng1 = createRng(42)
  const rng2 = createRng(42)

  for (let i = 0; i < 100; i++) {
    assertEquals(rng1(), rng2())
  }
})

Deno.test('createRng: different seeds produce different sequences', () => {
  const rng1 = createRng(1)
  const rng2 = createRng(2)

  let allSame = true
  for (let i = 0; i < 10; i++) {
    if (rng1() !== rng2()) {
      allSame = false
      break
    }
  }
  assertEquals(allSame, false)
})

Deno.test('createRng: values in (0, 1)', () => {
  const rng = createRng(123)
  for (let i = 0; i < 10000; i++) {
    const val = rng()
    assert(val > 0, `Value ${val} is not > 0`)
    assert(val < 1, `Value ${val} is not < 1`)
  }
})

Deno.test('createRng: reasonable distribution', () => {
  const rng = createRng(456)
  let below = 0
  const n = 10000
  for (let i = 0; i < n; i++) {
    if (rng() < 0.5) below++
  }
  // Should be roughly 50% — allow 45-55%
  assert(below > n * 0.45, `Too few below 0.5: ${below}/${n}`)
  assert(below < n * 0.55, `Too many below 0.5: ${below}/${n}`)
})

Deno.test('createRng: seed 0 works', () => {
  const rng = createRng(0)
  const val = rng()
  assert(val > 0 && val < 1)
})
