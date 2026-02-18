import { assertEquals } from "https://deno.land/std@0.224.0/assert/assert_equals.ts";
import { Bitset } from "../src/bitset.ts";

Deno.test("Bitset: set and has", () => {
  const bs = new Bitset(100);
  assertEquals(bs.has(0), false);
  bs.set(0);
  assertEquals(bs.has(0), true);

  assertEquals(bs.has(50), false);
  bs.set(50);
  assertEquals(bs.has(50), true);

  assertEquals(bs.has(99), false);
  bs.set(99);
  assertEquals(bs.has(99), true);
});

Deno.test("Bitset: clear resets all bits", () => {
  const bs = new Bitset(64);
  bs.set(0);
  bs.set(31);
  bs.set(32);
  bs.set(63);
  bs.clear();
  assertEquals(bs.has(0), false);
  assertEquals(bs.has(31), false);
  assertEquals(bs.has(32), false);
  assertEquals(bs.has(63), false);
});

Deno.test("Bitset: bits at word boundaries", () => {
  const bs = new Bitset(64);
  bs.set(31);
  bs.set(32);
  assertEquals(bs.has(31), true);
  assertEquals(bs.has(32), true);
  assertEquals(bs.has(30), false);
  assertEquals(bs.has(33), false);
});

Deno.test("Bitset: grow expands capacity", () => {
  const bs = new Bitset(32);
  assertEquals(bs.capacity, 32);
  bs.set(10);

  bs.grow(100);
  assertEquals(bs.capacity >= 100, true);
  // Old data preserved
  assertEquals(bs.has(10), true);
  // New range accessible
  bs.set(90);
  assertEquals(bs.has(90), true);
});

Deno.test("Bitset: grow is no-op when already large enough", () => {
  const bs = new Bitset(128);
  const cap = bs.capacity;
  bs.grow(64);
  assertEquals(bs.capacity, cap);
});

Deno.test("Bitset: capacity is correct", () => {
  const bs = new Bitset(1);
  assertEquals(bs.capacity, 32); // Minimum one Uint32
});
