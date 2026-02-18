import { assertEquals } from "https://deno.land/std@0.224.0/assert/assert_equals.ts";
import { assertAlmostEquals } from "https://deno.land/std@0.224.0/assert/assert_almost_equals.ts";
import { assert } from "https://deno.land/std@0.224.0/assert/assert.ts";
import { HNSWIndex } from "../src/hnsw_index.ts";
import { defaultHNSWConfig } from "../src/types.ts";
import {
  decodeShard,
  encodeShard,
  readHeader,
} from "../src/storage/serialization.ts";
import { InMemoryStorage } from "../src/storage/in_memory_storage.ts";
import { FileSystemStorage } from "../src/storage/file_system_storage.ts";

function makeIndex(
  dims: number,
  metric: "euclidean" | "cosine" | "inner_product" = "euclidean",
) {
  return new HNSWIndex(defaultHNSWConfig(dims, { seed: 42, metric }));
}

Deno.test("serialization: roundtrip empty index", () => {
  const index = makeIndex(4);
  const encoded = encodeShard(index);
  const decoded = decodeShard(encoded);
  assertEquals(decoded.size, 0);
});

Deno.test("serialization: roundtrip with vectors", () => {
  const index = makeIndex(3);
  index.insert("hello", [1, 2, 3]);
  index.insert("world", [4, 5, 6]);
  index.insert("test", [7, 8, 9]);

  const encoded = encodeShard(index);
  const decoded = decodeShard(encoded);

  assertEquals(decoded.size, 3);
  assertEquals(decoded.has("hello"), true);
  assertEquals(decoded.has("world"), true);
  assertEquals(decoded.has("test"), true);

  const vec = decoded.getVector("hello")!;
  assertEquals(Array.from(vec), [1, 2, 3]);
});

Deno.test("serialization: search works after roundtrip", () => {
  const index = makeIndex(4);

  for (let i = 0; i < 100; i++) {
    index.insert(`v${i}`, [
      Math.cos(i * 0.1),
      Math.sin(i * 0.1),
      Math.cos(i * 0.2),
      Math.sin(i * 0.2),
    ]);
  }

  // Search before serialization
  const query = new Float32Array([1, 0, 1, 0]);
  const resultsBefore = index.search(query, 5);

  // Roundtrip
  const encoded = encodeShard(index);
  const decoded = decodeShard(encoded);

  // Search after deserialization
  const resultsAfter = decoded.search(query, 5);

  assertEquals(resultsAfter.length, resultsBefore.length);
  for (let i = 0; i < resultsBefore.length; i++) {
    assertEquals(resultsAfter[i].id, resultsBefore[i].id);
    assertAlmostEquals(
      resultsAfter[i].distance,
      resultsBefore[i].distance,
      1e-6,
    );
  }
});

Deno.test("serialization: cosine metric roundtrip", () => {
  const index = makeIndex(3, "cosine");
  index.insert("a", [1, 0, 0]);
  index.insert("b", [0, 1, 0]);
  index.insert("c", [0.7, 0.7, 0]);

  const encoded = encodeShard(index);
  const decoded = decodeShard(encoded);

  assertEquals(decoded.config.metric, "cosine");
  const results = decoded.search([1, 0, 0], 3);
  assertEquals(results[0].id, "a");
});

Deno.test("serialization: readHeader returns correct metadata", () => {
  const index = makeIndex(128);
  for (let i = 0; i < 50; i++) {
    const v = new Float32Array(128);
    v[0] = i;
    index.insert(`v${i}`, v);
  }

  const encoded = encodeShard(index);
  const header = readHeader(encoded);

  assertEquals(header.dimensions, 128);
  assertEquals(header.count, 50);
  assertEquals(header.metric, "euclidean");
  assertEquals(header.M, 16);
});

Deno.test("serialization: unicode IDs roundtrip", () => {
  const index = makeIndex(2);
  index.insert("日本語", [1, 0]);
  index.insert("emoji-🎉", [0, 1]);

  const encoded = encodeShard(index);
  const decoded = decodeShard(encoded);

  assertEquals(decoded.has("日本語"), true);
  assertEquals(decoded.has("emoji-🎉"), true);
});

// Storage backend tests

Deno.test("InMemoryStorage: write/read/delete/list/exists", async () => {
  const storage = new InMemoryStorage();

  assertEquals(await storage.exists("key1"), false);
  assertEquals(await storage.list(), []);

  await storage.write("key1", new Uint8Array([1, 2, 3]));
  assertEquals(await storage.exists("key1"), true);

  const data = await storage.read("key1");
  assertEquals(Array.from(data!), [1, 2, 3]);

  const keys = await storage.list();
  assertEquals(keys, ["key1"]);

  assertEquals(await storage.delete("key1"), true);
  assertEquals(await storage.exists("key1"), false);
  assertEquals(await storage.read("key1"), null);
  assertEquals(await storage.delete("key1"), false);
});

Deno.test("FileSystemStorage: write/read/delete/list/exists", async () => {
  const tmpDir = await Deno.makeTempDir();
  const storage = new FileSystemStorage(tmpDir);

  try {
    assertEquals(await storage.exists("shard0"), false);
    assertEquals(await storage.list(), []);

    const testData = new Uint8Array([10, 20, 30, 40]);
    await storage.write("shard0", testData);

    assertEquals(await storage.exists("shard0"), true);

    const readBack = await storage.read("shard0");
    assert(readBack !== null);
    assertEquals(Array.from(readBack), [10, 20, 30, 40]);

    const keys = await storage.list();
    assertEquals(keys, ["shard0"]);

    assertEquals(await storage.delete("shard0"), true);
    assertEquals(await storage.exists("shard0"), false);
    assertEquals(await storage.read("shard0"), null);
  } finally {
    await Deno.remove(tmpDir, { recursive: true });
  }
});

Deno.test("FileSystemStorage: full serialization roundtrip", async () => {
  const tmpDir = await Deno.makeTempDir();
  const storage = new FileSystemStorage(tmpDir);

  try {
    const index = makeIndex(4);
    for (let i = 0; i < 50; i++) {
      index.insert(`v${i}`, [
        Math.cos(i * 0.1),
        Math.sin(i * 0.1),
        Math.cos(i * 0.2),
        Math.sin(i * 0.2),
      ]);
    }

    const encoded = encodeShard(index);
    await storage.write("test-shard", encoded);

    const loaded = await storage.read("test-shard");
    assert(loaded !== null);

    const decoded = decodeShard(loaded);
    assertEquals(decoded.size, 50);

    const query = new Float32Array([1, 0, 1, 0]);
    const resultsBefore = index.search(query, 5);
    const resultsAfter = decoded.search(query, 5);

    for (let i = 0; i < 5; i++) {
      assertEquals(resultsAfter[i].id, resultsBefore[i].id);
    }
  } finally {
    await Deno.remove(tmpDir, { recursive: true });
  }
});
