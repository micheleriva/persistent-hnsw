import { ShardManager } from './shard_manager.ts'
import {
  defaultHNSWConfig,
  defaultShardConfig,
  type ExternalId,
  type InsertItem,
  type SearchOptions,
  type SearchResult,
  type Vector,
  type VectorStoreConfig,
} from './types.ts'

export class VectorStore {
  private manager: ShardManager
  private config:
    & Required<Pick<VectorStoreConfig, 'hnsw' | 'sharding'>>
    & VectorStoreConfig

  private constructor(config: VectorStoreConfig, manager: ShardManager) {
    this.config = {
      ...config,
      hnsw: config.hnsw ?? {},
      sharding: config.sharding ?? {},
    }
    this.manager = manager
  }

  /** Create a new VectorStore. Dimensions must be set in config.hnsw.dimensions. */
  static create(config?: Partial<VectorStoreConfig>): VectorStore {
    const cfg = config ?? {}
    if (!cfg.hnsw?.dimensions) {
      throw new Error('hnsw.dimensions is required')
    }

    const hnswConfig = defaultHNSWConfig(cfg.hnsw.dimensions, cfg.hnsw)
    const shardConfig = defaultShardConfig(cfg.sharding)
    const manager = new ShardManager(
      hnswConfig,
      shardConfig,
      cfg.storage ?? null,
    )

    return new VectorStore(cfg, manager)
  }

  /** Open an existing VectorStore from storage. */
  static async open(config: VectorStoreConfig): Promise<VectorStore> {
    if (!config.storage) {
      throw new Error('storage backend is required to open an existing store')
    }
    if (!config.hnsw?.dimensions) {
      throw new Error('hnsw.dimensions is required')
    }

    const hnswConfig = defaultHNSWConfig(config.hnsw.dimensions, config.hnsw)
    const shardConfig = defaultShardConfig(config.sharding)
    const manager = new ShardManager(hnswConfig, shardConfig, config.storage)
    await manager.loadFromStorage()

    return new VectorStore(config, manager)
  }

  /** Insert one or more items. */
  async insert(items: InsertItem | InsertItem[]): Promise<void> {
    const arr = Array.isArray(items) ? items : [items]
    for (const item of arr) {
      const vec = item.vector instanceof Float32Array ? item.vector : new Float32Array(item.vector)
      await this.manager.insert(item.id, vec)
    }
  }

  /** Search for the k nearest neighbors. */
  async search(
    query: Vector | number[],
    k: number,
    options?: SearchOptions,
  ): Promise<SearchResult[]> {
    return this.manager.search(query, k, options)
  }

  /** Delete a vector by ID. */
  async delete(id: ExternalId): Promise<boolean> {
    return this.manager.delete(id)
  }

  /** Flush all dirty shards to storage. */
  async flush(): Promise<void> {
    return this.manager.flush()
  }

  /** Compact all shards (rebuild without deleted nodes). */
  async compact(): Promise<void> {
    return this.manager.compact()
  }

  /** Close the store: flush and release all resources. */
  async close(): Promise<void> {
    return this.manager.close()
  }

  /** Total number of (non-deleted) vectors. */
  get size(): number {
    return this.manager.size
  }
}
