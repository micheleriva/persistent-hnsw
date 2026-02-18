import { VectorStore } from '../src/vector_store.ts'
import { InMemoryStorage } from '../src/storage/in_memory_storage.ts'
import { generateVectors } from './helpers/generate_vectors.ts'

const dim = 64
const n = 5_000
const vectors = generateVectors(n, dim)
const queries = generateVectors(20, dim, 99999)

// Single shard store
const singleStore = VectorStore.create({
  hnsw: { dimensions: dim, seed: 42, M: 8, efConstruction: 50 },
  sharding: { maxVectorsPerShard: 100_000 },
  storage: new InMemoryStorage(),
})
for (let i = 0; i < n; i++) {
  await singleStore.insert({ id: `v${i}`, vector: vectors[i] })
}

// Multi-shard store (1000 per shard = 5 shards)
const multiStore = VectorStore.create({
  hnsw: { dimensions: dim, seed: 42, M: 8, efConstruction: 50 },
  sharding: { maxVectorsPerShard: 1_000 },
  storage: new InMemoryStorage(),
})
for (let i = 0; i < n; i++) {
  await multiStore.insert({ id: `v${i}`, vector: vectors[i] })
}

let qi = 0

Deno.bench('search single shard (5K vectors)', async () => {
  await singleStore.search(queries[qi % queries.length], 10)
  qi++
})

Deno.bench('search 5 shards (5K vectors total)', async () => {
  await multiStore.search(queries[qi % queries.length], 10)
  qi++
})
