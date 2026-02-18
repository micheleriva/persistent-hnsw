const assert = require("node:assert/strict");
const { mkdtemp, rm } = require("node:fs/promises");
const { tmpdir } = require("node:os");
const { join } = require("node:path");

const {
  HNSWIndex,
  defaultHNSWConfig,
  InMemoryStorage,
  FileSystemStorage,
  VectorStore,
} = require("../../.npm/script/mod.js");

async function main() {
  // HNSWIndex
  const config = defaultHNSWConfig(4, { seed: 42 });
  const index = new HNSWIndex(config);
  index.insert("a", [1, 0, 0, 0]);
  index.insert("b", [0, 1, 0, 0]);
  index.insert("c", [0.9, 0.1, 0, 0]);

  const results = index.search([1, 0, 0, 0], 2);
  assert.equal(results[0].id, "a");
  assert.equal(results.length, 2);
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
  const dir = await mkdtemp(join(tmpdir(), "hnsw-cjs-"));
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

  console.log("Node CJS: all passed");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
