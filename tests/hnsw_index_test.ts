import { assertEquals } from "https://deno.land/std@0.224.0/assert/assert_equals.ts";
import { assert } from "https://deno.land/std@0.224.0/assert/assert.ts";
import { assertThrows } from "https://deno.land/std@0.224.0/assert/assert_throws.ts";
import { HNSWIndex } from "../src/hnsw_index.ts";
import { defaultHNSWConfig } from "../src/types.ts";

function makeConfig(dims: number, overrides?: Record<string, unknown>) {
  return defaultHNSWConfig(dims, {
    seed: 42,
    efConstruction: 200,
    efSearch: 50,
    ...overrides,
  });
}

Deno.test("HNSWIndex: insert and search single vector", () => {
  const config = makeConfig(4);
  const index = new HNSWIndex(config);

  index.insert("a", new Float32Array([1, 0, 0, 0]));
  assertEquals(index.size, 1);

  const results = index.search(new Float32Array([1, 0, 0, 0]), 1);
  assertEquals(results.length, 1);
  assertEquals(results[0].id, "a");
  assertEquals(results[0].distance, 0);
});

Deno.test("HNSWIndex: insert and search multiple vectors", () => {
  const config = makeConfig(3);
  const index = new HNSWIndex(config);

  index.insert("a", [1, 0, 0]);
  index.insert("b", [0, 1, 0]);
  index.insert("c", [0, 0, 1]);
  index.insert("d", [1, 1, 0]);

  const results = index.search([1, 0, 0], 2);
  assertEquals(results.length, 2);
  assertEquals(results[0].id, "a");
  // Second nearest should be "d" ([1,1,0]) at distance 1
  assertEquals(results[1].id, "d");
});

Deno.test("HNSWIndex: search returns correct number of results", () => {
  const config = makeConfig(2);
  const index = new HNSWIndex(config);

  for (let i = 0; i < 20; i++) {
    index.insert(`v${i}`, [Math.cos(i), Math.sin(i)]);
  }

  const results5 = index.search([1, 0], 5);
  assertEquals(results5.length, 5);

  const results10 = index.search([1, 0], 10);
  assertEquals(results10.length, 10);
});

Deno.test("HNSWIndex: search empty index returns empty", () => {
  const config = makeConfig(3);
  const index = new HNSWIndex(config);

  const results = index.search([1, 0, 0], 5);
  assertEquals(results.length, 0);
});

Deno.test("HNSWIndex: duplicate ID throws", () => {
  const config = makeConfig(2);
  const index = new HNSWIndex(config);

  index.insert("a", [1, 0]);
  assertThrows(
    () => index.insert("a", [0, 1]),
    Error,
    "Duplicate ID",
  );
});

Deno.test("HNSWIndex: dimension mismatch throws", () => {
  const config = makeConfig(3);
  const index = new HNSWIndex(config);

  assertThrows(
    () => index.insert("a", [1, 0]),
    Error,
    "dimension mismatch",
  );
});

Deno.test("HNSWIndex: delete marks as deleted", () => {
  const config = makeConfig(2);
  const index = new HNSWIndex(config);

  index.insert("a", [1, 0]);
  index.insert("b", [0, 1]);

  assertEquals(index.size, 2);
  assert(index.has("a"));

  assertEquals(index.delete("a"), true);
  assertEquals(index.size, 1);
  assert(!index.has("a"));

  // Deleted items not returned in search
  const results = index.search([1, 0], 5);
  assertEquals(results.length, 1);
  assertEquals(results[0].id, "b");
});

Deno.test("HNSWIndex: delete non-existent returns false", () => {
  const config = makeConfig(2);
  const index = new HNSWIndex(config);

  assertEquals(index.delete("nonexistent"), false);
});

Deno.test("HNSWIndex: double delete returns false", () => {
  const config = makeConfig(2);
  const index = new HNSWIndex(config);

  index.insert("a", [1, 0]);
  assertEquals(index.delete("a"), true);
  assertEquals(index.delete("a"), false);
});

Deno.test("HNSWIndex: getVector returns vector", () => {
  const config = makeConfig(3);
  const index = new HNSWIndex(config);

  index.insert("a", [1, 2, 3]);
  const vec = index.getVector("a");
  assertEquals(vec !== null, true);
  assertEquals(Array.from(vec!), [1, 2, 3]);
});

Deno.test("HNSWIndex: getVector returns null for deleted", () => {
  const config = makeConfig(2);
  const index = new HNSWIndex(config);

  index.insert("a", [1, 0]);
  index.delete("a");
  assertEquals(index.getVector("a"), null);
});

Deno.test("HNSWIndex: compact removes deleted nodes", () => {
  const config = makeConfig(2);
  const index = new HNSWIndex(config);

  index.insert("a", [1, 0]);
  index.insert("b", [0, 1]);
  index.insert("c", [1, 1]);

  index.delete("b");
  assertEquals(index.size, 2);

  const compacted = index.compact();
  assertEquals(compacted.size, 2);
  assert(compacted.has("a"));
  assert(!compacted.has("b"));
  assert(compacted.has("c"));
});

Deno.test("HNSWIndex: search with filter", () => {
  const config = makeConfig(2);
  const index = new HNSWIndex(config);

  index.insert("a", [1, 0]);
  index.insert("b", [0.9, 0.1]);
  index.insert("c", [0, 1]);

  // Filter out "a" and "b"
  const results = index.search([1, 0], 5, undefined, (id) => id === "c");
  assertEquals(results.length, 1);
  assertEquals(results[0].id, "c");
});

Deno.test("HNSWIndex: cosine metric works", () => {
  const config = makeConfig(3, { metric: "cosine" });
  const index = new HNSWIndex(config);

  index.insert("a", [1, 0, 0]);
  index.insert("b", [0, 1, 0]);
  index.insert("c", [0.9, 0.1, 0]);

  const results = index.search([1, 0, 0], 3);
  assertEquals(results[0].id, "a");
  // "c" should be second nearest by cosine
  assertEquals(results[1].id, "c");
});

Deno.test("HNSWIndex: inner_product metric works", () => {
  const config = makeConfig(3, { metric: "inner_product" });
  const index = new HNSWIndex(config);

  index.insert("a", [1, 0, 0]);
  index.insert("b", [10, 0, 0]);
  index.insert("c", [0.1, 0, 0]);

  const results = index.search([1, 0, 0], 3);
  // Higher dot product = lower distance (negated), so "b" first
  assertEquals(results[0].id, "b");
  assertEquals(results[1].id, "a");
  assertEquals(results[2].id, "c");
});

Deno.test("HNSWIndex: handles capacity growth", () => {
  const config = makeConfig(4, { M: 4, efConstruction: 20 });
  const index = new HNSWIndex(config);

  // Insert more than initial capacity (1024)
  for (let i = 0; i < 1500; i++) {
    const vec = new Float32Array(4);
    vec[0] = Math.cos(i * 0.1);
    vec[1] = Math.sin(i * 0.1);
    vec[2] = Math.cos(i * 0.2);
    vec[3] = Math.sin(i * 0.2);
    index.insert(`v${i}`, vec);
  }

  assertEquals(index.size, 1500);

  const results = index.search([1, 0, 1, 0], 5);
  assertEquals(results.length, 5);
});

Deno.test("HNSWIndex: simple neighbor selection (no heuristic)", () => {
  const config = makeConfig(2, { useHeuristic: false });
  const index = new HNSWIndex(config);

  for (let i = 0; i < 50; i++) {
    index.insert(`v${i}`, [Math.cos(i * 0.1), Math.sin(i * 0.1)]);
  }

  const results = index.search([1, 0], 5);
  assertEquals(results.length, 5);
});

Deno.test("HNSWIndex: memoryUsage returns positive number", () => {
  const config = makeConfig(4);
  const index = new HNSWIndex(config);

  index.insert("a", [1, 2, 3, 4]);
  assert(index.memoryUsage() > 0);
});
