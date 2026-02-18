export { VectorStore } from "./src/vector_store.ts";
export { HNSWIndex } from "./src/hnsw_index.ts";
export { InMemoryStorage } from "./src/storage/in_memory_storage.ts";
export { FileSystemStorage } from "./src/storage/file_system_storage.ts";

export type {
  DistanceFunction,
  DistanceMetric,
  ExternalId,
  HNSWConfig,
  InsertItem,
  SearchOptions,
  SearchResult,
  ShardConfig,
  StorageBackend,
  Vector,
  VectorStoreConfig,
} from "./src/types.ts";

export { defaultHNSWConfig, defaultShardConfig } from "./src/types.ts";
