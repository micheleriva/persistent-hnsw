# Persistent HNSW

Pure-TypeScript HNSW vector search. No native dependencies, no WASM - just
TypeScript.

Scales to tens of thousands of vectors in-memory with sub-millisecond search.
Supports persistence via pluggable storage backends, automatic sharding with LRU
eviction, and three distance metrics.

## Install

```sh
npm i persistent-hnsw
```

```ts
import {
  FileSystemStorage,
  HNSWIndex,
  InMemoryStorage,
  VectorStore,
} from "persistent-hnsw";
```

## Quick Start

```ts
import { VectorStore } from "persistent-hnsw";

const store = VectorStore.create({
  hnsw: { dimensions: 128 },
});

// Insert vectors
await store.insert([
  { id: "doc-1", vector: new Float32Array(128).fill(0.1) },
  { id: "doc-2", vector: new Float32Array(128).fill(0.5) },
  { id: "doc-3", vector: new Float32Array(128).fill(0.9) },
]);

// Search
const results = await store.search(new Float32Array(128).fill(0.5), 2);
console.log(results);
// [{ id: "doc-2", distance: 0 }, { id: "doc-1", distance: 2.56 }]

// Delete
await store.delete("doc-1");
```

## API

### `VectorStore`

The high-level API. Handles sharding, persistence, and configuration.

```ts
// Create a new store
const store = VectorStore.create({
  hnsw: {
    dimensions: 128, // required
    M: 16, // max neighbors per node (default: 16)
    efConstruction: 200, // build-time beam width (default: 200)
    efSearch: 50, // search-time beam width (default: 50)
    metric: "euclidean", // "euclidean" | "cosine" | "inner_product"
  },
  sharding: {
    maxVectorsPerShard: 100_000,
    maxLoadedShards: 4,
  },
  storage: new FileSystemStorage("./data"),
});

// Insert one or many
await store.insert({ id: "a", vector: [1, 2, 3] });
await store.insert([
  { id: "b", vector: [4, 5, 6] },
  { id: "c", vector: [7, 8, 9] },
]);

// Search — returns { id, distance }[]
const results = await store.search([1, 2, 3], 10);

// Search with options
const filtered = await store.search([1, 2, 3], 10, {
  efSearch: 200, // override beam width
  filter: (id) => id !== "b", // pre-filter by ID
  includeVectors: true, // attach vector data to results
});

// Delete
await store.delete("a");

// Persistence
await store.flush(); // write dirty shards to storage
await store.close(); // flush + release resources

// Reopen from storage
const reopened = await VectorStore.open({
  hnsw: { dimensions: 128 },
  storage: new FileSystemStorage("./data"),
});
```

### `HNSWIndex`

The low-level index. Synchronous, zero async overhead. Use this when you don't
need sharding or persistence.

```ts
import { defaultHNSWConfig, HNSWIndex } from "persistent-hnsw";

const config = defaultHNSWConfig(128, {
  M: 16,
  efConstruction: 200,
  metric: "cosine",
  seed: 42, // deterministic builds
});

const index = new HNSWIndex(config);

// Insert
index.insert("vec-1", new Float32Array([/* 128 values */]));
index.insert("vec-2", [0.1, 0.2 /* ... */]); // number[] also works

// Search — returns { id, distance }[]
const results = index.search([0.1, 0.2 /* ... */], 10);

// Search with custom efSearch
const precise = index.search(query, 10, 400);

// Search with filter
const filtered = index.search(
  query,
  10,
  undefined,
  (id) => id.startsWith("doc-"),
);

// Delete (lazy tombstone)
index.delete("vec-1");

// Rebuild without deleted nodes
const compacted = index.compact();

// Reclaim unused memory after bulk insert
index.shrinkToFit();

// Check state
index.size; // number of live vectors
index.has("vec-2"); // true
index.getVector("vec-2"); // Float32Array | null
index.memoryUsage(); // bytes
```

### Storage Backends

```ts
import { FileSystemStorage, InMemoryStorage } from "persistent-hnsw";

// In-memory — works in browsers, no filesystem needed
const mem = new InMemoryStorage();

// Filesystem — uses Deno.readFile/writeFile
const fs = new FileSystemStorage("./hnsw-data");
```

Both implement `StorageBackend`:

```ts
interface StorageBackend {
  write(key: string, data: Uint8Array): Promise<void>;
  read(key: string): Promise<Uint8Array | null>;
  delete(key: string): Promise<boolean>;
  list(): Promise<string[]>;
  exists(key: string): Promise<boolean>;
}
```

You can implement your own for S3, Redis, SQLite, etc.

### Distance Metrics

| Metric            | Use case                            | Notes                                      |
| ----------------- | ----------------------------------- | ------------------------------------------ |
| `"euclidean"`     | General purpose                     | Squared L2, no sqrt (lower = closer)       |
| `"cosine"`        | Text embeddings, normalized vectors | 1 - cos(a,b) (0 = identical, 2 = opposite) |
| `"inner_product"` | Recommendation, MIP search          | Negated dot product (lower = more similar) |

## Configuration Guide

### Choosing M

`M` controls how many neighbors each node maintains. Higher M = better recall,
slower insert, more memory.

| M      | Recall@10 | Search QPS | Memory (10K/128d) |
| ------ | --------- | ---------- | ----------------- |
| 4      | 61%       | 2.9K       | 6.2 MB            |
| 8      | 84%       | 1.8K       | 6.8 MB            |
| **16** | **97%**   | **1.2K**   | **8.0 MB**        |
| 32     | 99%       | 840        | 9.8 MB            |

**Default: 16.** Good balance for most use cases. Use 32 if you need >99%
recall.

### Choosing efSearch

`efSearch` controls search accuracy at query time. Higher = better recall,
slower search.

| efSearch | Recall@10 | Latency   | QPS      |
| -------- | --------- | --------- | -------- |
| 10       | 34%       | 87µs      | 11.5K    |
| 50       | 74%       | 322µs     | 3.1K     |
| 100      | 88%       | 523µs     | 1.9K     |
| **200**  | **97%**   | **950µs** | **1.1K** |
| 400      | 99.7%     | 1.5ms     | 652      |

**Default: 50.** Increase to 200+ for high-recall workloads.

### Choosing efConstruction

`efConstruction` controls graph quality at build time. Higher = better graph,
slower insert. Only affects index building, not search.

**Default: 200.** Rarely needs changing. Lower to 100 for faster builds if
recall is acceptable.

## Benchmarks

All benchmarks on Apple M2 Pro, Deno 1.36.4. Run with `deno task bench:full`.

### Scaling (128d, M=16, efConstruction=200)

| Vectors | Insert ops/s | Build Time | Search (ef=200) | QPS  | Recall@ef200 | Recall@ef500 | Memory  | On Disk |
| ------- | ------------ | ---------- | --------------- | ---- | ------------ | ------------ | ------- | ------- |
| 1K      | 591          | 1.7s       | 313µs           | 3.2K | 100%         | 100%         | 0.8 MB  | 0.8 MB  |
| 5K      | 371          | 13.5s      | 695µs           | 1.4K | 99%          | 100%         | 4.0 MB  | 4.0 MB  |
| 10K     | 313          | 31.9s      | 877µs           | 1.1K | 97%          | 100%         | 8.0 MB  | 8.1 MB  |
| 25K     | 267          | 93.8s      | 1095µs          | 913  | 89%          | 98%          | 20.1 MB | 20.2 MB |
| 50K     | 242          | 206.4s     | 1376µs          | 727  | 79%          | 93%          | 49.4 MB | 49.7 MB |

```
Search Latency (median)
1.0K vectors  |#########                                317 us
5.0K vectors  |####################                     695 us
10.0K vectors |#########################                877 us
25.0K vectors |################################         1.1K us
50.0K vectors |########################################  1.4K us
```

### Dimension Scaling (5K vectors, M=16)

| Dimensions | Insert ops/s | Search ops/s | Latency | Recall@10 | Memory  |
| ---------- | ------------ | ------------ | ------- | --------- | ------- |
| 32         | 759          | 4.6K         | 216µs   | 99.7%     | 2.2 MB  |
| 128        | 391          | 2.3K         | 443µs   | 92.7%     | 4.0 MB  |
| 384        | 170          | 963          | 1039µs  | 89.7%     | 8.9 MB  |
| 768        | 97           | 538          | 1859µs  | 79.3%     | 16.2 MB |
| 1536       | 52           | 292          | 3427µs  | 76.3%     | 30.9 MB |

<details>
<summary>All dimension benchmarks (32d — 1536d)</summary>

| Dimensions | Insert ops/s | Search ops/s | Search Median | Recall@10 | Memory  | On Disk |
| ---------- | ------------ | ------------ | ------------- | --------- | ------- | ------- |
| 32         | 759          | 4.6K         | 216µs         | 99.7%     | 2.2 MB  | 2.2 MB  |
| 64         | 569          | 3.4K         | 295µs         | 98.0%     | 2.8 MB  | 2.8 MB  |
| 128        | 391          | 2.3K         | 443µs         | 92.7%     | 4.0 MB  | 4.0 MB  |
| 256        | 246          | 1.2K         | 824µs         | 88.3%     | 6.5 MB  | 6.5 MB  |
| 384        | 170          | 963          | 1039µs        | 89.7%     | 8.9 MB  | 8.9 MB  |
| 512        | 140          | 723          | 1384µs        | 88.7%     | 11.3 MB | 11.4 MB |
| 768        | 97           | 538          | 1859µs        | 79.3%     | 16.2 MB | 16.2 MB |
| 1024       | 77           | 431          | 2319µs        | 83.0%     | 21.1 MB | 21.1 MB |
| 1536       | 52           | 292          | 3427µs        | 76.3%     | 30.9 MB | 30.9 MB |

</details>

### Distance Function Throughput

```
Euclidean (L2 squared) ops/sec
128d  |########################################  10.5M ops/s
256d  |####################                      5.3M ops/s
768d  |#######                                   1.8M ops/s
1536d |###                                       914K ops/s
```

### Serialization

| Vectors | Size    | Encode    | Decode    |
| ------- | ------- | --------- | --------- |
| 1K      | 0.8 MB  | 1592 MB/s | 450 MB/s  |
| 10K     | 8.1 MB  | 801 MB/s  | 1570 MB/s |
| 25K     | 20.2 MB | 2061 MB/s | 1634 MB/s |

Full benchmark results with per-dimension breakdowns are in
[`bench/RESULTS.md`](bench/RESULTS.md).

## Architecture

```
VectorStore          High-level async API
  └─ ShardManager    Multi-shard orchestrator, LRU eviction
       └─ HNSWIndex  Core HNSW graph — synchronous, zero async overhead
```

The hot path (distance calculations, graph traversal) is fully synchronous.
Async only happens at storage/shard boundaries.

Key internals:

- **Flat typed arrays** for vectors and adjacency lists (cache-friendly, trivial
  serialization)
- **Bitset** visited set (Uint32Array-backed, replaces Set<number>)
- **Pooled heaps and bitset** reused across search calls (no per-query
  allocation)
- **4-wide unrolled loops** in distance functions for V8 optimization
- **1.5x growth factor** with `shrinkToFit()` to reclaim unused capacity
- **Lazy deletion** with tombstones; `compact()` rebuilds the graph

## Development

```bash
deno task test        # run all tests
deno task bench       # micro-benchmarks
deno task bench:full  # full benchmark suite → bench/RESULTS.md
deno task check       # type-check
deno task lint        # lint
deno task fmt         # format
```

## License

[MIT](/LICENSE.md)
