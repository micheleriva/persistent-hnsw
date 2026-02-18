import { HNSWIndex } from "../src/hnsw_index.ts";
import { defaultHNSWConfig } from "../src/types.ts";
import { generateVectors } from "./helpers/generate_vectors.ts";

const dim = 128;
const vectors10k = generateVectors(10_000, dim);

Deno.bench("insert 10K vectors (M=16, efC=200)", () => {
  const config = defaultHNSWConfig(dim, {
    seed: 42,
    M: 16,
    efConstruction: 200,
  });
  const index = new HNSWIndex(config);
  for (let i = 0; i < vectors10k.length; i++) {
    index.insert(`v${i}`, vectors10k[i]);
  }
});

Deno.bench("insert 10K vectors (M=8, efC=100)", () => {
  const config = defaultHNSWConfig(dim, {
    seed: 42,
    M: 8,
    efConstruction: 100,
  });
  const index = new HNSWIndex(config);
  for (let i = 0; i < vectors10k.length; i++) {
    index.insert(`v${i}`, vectors10k[i]);
  }
});

Deno.bench("insert 10K vectors (M=32, efC=200)", () => {
  const config = defaultHNSWConfig(dim, {
    seed: 42,
    M: 32,
    efConstruction: 200,
  });
  const index = new HNSWIndex(config);
  for (let i = 0; i < vectors10k.length; i++) {
    index.insert(`v${i}`, vectors10k[i]);
  }
});
