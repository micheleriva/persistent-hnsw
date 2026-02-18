import { HNSWIndex } from '../src/hnsw_index.ts'
import { defaultHNSWConfig } from '../src/types.ts'
import { generateVectors } from './helpers/generate_vectors.ts'

const dim = 128
const n = 10_000
const k = 10

// Pre-build index
const config = defaultHNSWConfig(dim, {
  seed: 42,
  M: 16,
  efConstruction: 200,
})
const index = new HNSWIndex(config)
const vectors = generateVectors(n, dim)
for (let i = 0; i < n; i++) {
  index.insert(`v${i}`, vectors[i])
}

// Generate query vectors
const queries = generateVectors(100, dim, 99999)
let qi = 0

for (const ef of [10, 50, 100, 200]) {
  Deno.bench(`search k=${k} ef=${ef} (10K index)`, () => {
    index.search(queries[qi % queries.length], k, ef)
    qi++
  })
}
