import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  HNSWIndex,
  defaultHNSWConfig,
  InMemoryStorage,
  FileSystemStorage,
  VectorStore,
} from "../../.npm/esm/mod.js";

// HNSWIndex
const config = defaultHNSWConfig(128, { seed: 42 });
const index = new HNSWIndex(config);
const vecs = Array.from({ length: 100 }, (_, i) => {
  const v = new Float32Array(128);
  v[0] = Math.cos(i * 0.1);
  v[1] = Math.sin(i * 0.1);
  return v;
});
vecs.forEach((v, i) => index.insert("v" + i, v));

const results = index.search(vecs[0], 5);
assert.equal(results[0].id, "v0");
assert.equal(results.length, 5);
console.log("  HNSWIndex: OK");

// VectorStore + InMemoryStorage
const store = VectorStore.create({
  hnsw: { dimensions: 4 },
  storage: new InMemoryStorage(),
});
await store.insert([
  { id: "x", vector: [1, 0, 0, 0] },
  { id: "y", vector: [0, 1, 0, 0] },
]);
const sr = await store.search([1, 0, 0, 0], 1);
assert.equal(sr[0].id, "x");
await store.close();
console.log("  VectorStore: OK");

// FileSystemStorage
const dir = await mkdtemp(join(tmpdir(), "hnsw-esm-"));
try {
  const fs = new FileSystemStorage(dir);
  await fs.write("k", new Uint8Array([1, 2, 3]));
  const data = await fs.read("k");
  assert.equal(data.length, 3);
  const keys = await fs.list();
  assert.equal(keys.length, 1);
  assert.equal(await fs.delete("k"), true);
  assert.equal(await fs.exists("k"), false);
} finally {
  await rm(dir, { recursive: true });
}
console.log("  FileSystemStorage: OK");

console.log("Node ESM: all passed");
