import { HNSWIndex } from '../src/hnsw_index.ts'
import { defaultHNSWConfig } from '../src/types.ts'
import { decodeShard, encodeShard } from '../src/storage/serialization.ts'
import { generateVectors } from './helpers/generate_vectors.ts'

const dim = 128
const n = 10_000

// Pre-build index
const config = defaultHNSWConfig(dim, {
  seed: 42,
  M: 16,
  efConstruction: 100,
})
const index = new HNSWIndex(config)
const vectors = generateVectors(n, dim)
for (let i = 0; i < n; i++) {
  index.insert(`v${i}`, vectors[i])
}

// Pre-encode for decode benchmark
const encoded = encodeShard(index)
console.log(
  `  Serialized size: ${(encoded.byteLength / 1024 / 1024).toFixed(2)} MB`,
)

Deno.bench(`encode 10K vectors (${dim}d)`, () => {
  encodeShard(index)
})

Deno.bench(`decode 10K vectors (${dim}d)`, () => {
  decodeShard(encoded)
})
