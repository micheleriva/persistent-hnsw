/** A vector represented as a Float32Array for performance. */
export type Vector = Float32Array

/** Internal node ID — uint32 range, used as array index. */
export type InternalId = number

/** External ID — user-facing string identifier. */
export type ExternalId = string

/** Supported distance metrics. */
export type DistanceMetric = 'euclidean' | 'cosine' | 'inner_product'

/** A function that computes distance between two vectors. Lower = more similar. */
export type DistanceFunction = (a: Vector, b: Vector) => number

/** Configuration for the HNSW index. */
export interface HNSWConfig {
  /** Number of dimensions for each vector. */
  dimensions: number
  /** Max number of neighbors per node (layers > 0). Default 16. */
  M: number
  /** Max number of neighbors at layer 0. Default 2*M. */
  Mmax0: number
  /** Size of the dynamic candidate list during construction. Default 200. */
  efConstruction: number
  /** Size of the dynamic candidate list during search. Default 50. */
  efSearch: number
  /** Distance metric. Default "euclidean". */
  metric: DistanceMetric
  /** Level generation factor: 1/ln(M). */
  mL: number
  /** Use the neighbor selection heuristic (Algorithm 4). Default true. */
  useHeuristic: boolean
  /** Keep pruned connections when using heuristic. Default true. */
  keepPrunedConnections: boolean
  /** PRNG seed for deterministic layer assignment. */
  seed?: number
}

/** Configuration for sharding. */
export interface ShardConfig {
  /** Max vectors per shard. Default 100_000. */
  maxVectorsPerShard: number
  /** Max number of loaded shards in memory at once. */
  maxLoadedShards: number
}

/** Interface for persistent storage backends. */
export interface StorageBackend {
  write(key: string, data: Uint8Array): Promise<void>
  read(key: string): Promise<Uint8Array | null>
  delete(key: string): Promise<boolean>
  list(): Promise<string[]>
  exists(key: string): Promise<boolean>
}

/** A single search result. */
export interface SearchResult {
  /** External ID of the matched vector. */
  id: ExternalId
  /** Distance from the query. */
  distance: number
  /** The vector data, if requested. */
  vector?: Vector
}

/** An item to insert. */
export interface InsertItem {
  /** External ID. */
  id: ExternalId
  /** The vector data. */
  vector: Vector | number[]
}

/** Top-level configuration for VectorStore. */
export interface VectorStoreConfig {
  hnsw?: Partial<HNSWConfig>
  sharding?: Partial<ShardConfig>
  storage?: StorageBackend
}

/** Options for search queries. */
export interface SearchOptions {
  /** Override the default efSearch for this query. */
  efSearch?: number
  /** Include vector data in results. */
  includeVectors?: boolean
  /** Filter function applied to external IDs. */
  filter?: (id: ExternalId) => boolean
}

/** Sentinel value for empty adjacency slots. */
export const SENTINEL = 0xFFFFFFFF

/** Default HNSW configuration values. */
export function defaultHNSWConfig(
  dimensions: number,
  overrides?: Partial<HNSWConfig>,
): HNSWConfig {
  const M = overrides?.M ?? 16
  return {
    dimensions,
    M,
    Mmax0: overrides?.Mmax0 ?? 2 * M,
    efConstruction: overrides?.efConstruction ?? 200,
    efSearch: overrides?.efSearch ?? 50,
    metric: overrides?.metric ?? 'euclidean',
    mL: overrides?.mL ?? 1 / Math.log(M),
    useHeuristic: overrides?.useHeuristic ?? true,
    keepPrunedConnections: overrides?.keepPrunedConnections ?? true,
    seed: overrides?.seed,
  }
}

/** Default shard configuration values. */
export function defaultShardConfig(
  overrides?: Partial<ShardConfig>,
): ShardConfig {
  return {
    maxVectorsPerShard: overrides?.maxVectorsPerShard ?? 100_000,
    maxLoadedShards: overrides?.maxLoadedShards ?? 4,
  }
}
