import { HNSWIndex } from "./hnsw_index.ts";
import { BinaryHeap } from "./binary_heap.ts";
import {
  type ExternalId,
  type HNSWConfig,
  type SearchOptions,
  type SearchResult,
  type ShardConfig,
  type StorageBackend,
  type Vector,
} from "./types.ts";
import { decodeShard, encodeShard } from "./storage/serialization.ts";

interface LoadedShard {
  key: string;
  index: HNSWIndex;
  dirty: boolean;
  lastAccess: number;
}

export class ShardManager {
  private readonly hnswConfig: HNSWConfig;
  private readonly shardConfig: ShardConfig;
  private readonly storage: StorageBackend | null;

  // Loaded shards in memory
  private loadedShards = new Map<string, LoadedShard>();

  // Global ID → shard key mapping
  private idToShard = new Map<ExternalId, string>();

  // Ordered list of shard keys
  private shardKeys: string[] = [];

  // Current shard for inserts
  private currentShardKey: string | null = null;
  private currentShardCount = 0;

  // Total vector count
  private totalCount = 0;

  // Access counter for LRU
  private accessCounter = 0;

  constructor(
    hnswConfig: HNSWConfig,
    shardConfig: ShardConfig,
    storage: StorageBackend | null,
  ) {
    this.hnswConfig = hnswConfig;
    this.shardConfig = shardConfig;
    this.storage = storage;
  }

  get size(): number {
    return this.totalCount;
  }

  get shardCount(): number {
    return this.shardKeys.length;
  }

  get loadedShardCount(): number {
    return this.loadedShards.size;
  }

  /** Insert a vector. Routes to the current shard, creating a new one if full. */
  async insert(id: ExternalId, vector: Vector | number[]): Promise<void> {
    if (this.idToShard.has(id)) {
      throw new Error(`Duplicate ID: ${id}`);
    }

    // Get or create current shard
    if (
      this.currentShardKey === null ||
      this.currentShardCount >= this.shardConfig.maxVectorsPerShard
    ) {
      await this.createNewShard();
    }

    const shard = await this.getShard(this.currentShardKey!);
    shard.index.insert(id, vector);
    shard.dirty = true;

    this.idToShard.set(id, this.currentShardKey!);
    this.currentShardCount++;
    this.totalCount++;
  }

  /** Search across all shards and merge results. */
  async search(
    query: Vector | number[],
    k: number,
    options?: SearchOptions,
  ): Promise<SearchResult[]> {
    const q = query instanceof Float32Array ? query : new Float32Array(query);

    // Load all shards and search in parallel
    const shardPromises = this.shardKeys.map(async (key) => {
      const shard = await this.getShard(key);
      return shard.index.search(q, k, options?.efSearch, options?.filter);
    });

    const allResults = await Promise.all(shardPromises);

    // Merge results using a min-heap
    const heap = new BinaryHeap<SearchResult>(
      (a, b) => a.distance - b.distance,
    );

    for (const results of allResults) {
      for (const result of results) {
        heap.push(result);
      }
    }

    // Extract top-k
    const merged: SearchResult[] = [];
    while (merged.length < k && heap.size > 0) {
      const item = heap.pop()!;
      if (options?.includeVectors) {
        // Find vector from the correct shard
        const shardKey = this.idToShard.get(item.id);
        if (shardKey) {
          const shard = await this.getShard(shardKey);
          item.vector = shard.index.getVector(item.id) ?? undefined;
        }
      }
      merged.push(item);
    }

    return merged;
  }

  /** Delete a vector by external ID. */
  async delete(id: ExternalId): Promise<boolean> {
    const shardKey = this.idToShard.get(id);
    if (!shardKey) return false;

    const shard = await this.getShard(shardKey);
    const deleted = shard.index.delete(id);
    if (deleted) {
      shard.dirty = true;
      this.idToShard.delete(id);
      this.totalCount--;
    }
    return deleted;
  }

  /** Flush all dirty shards to storage. */
  async flush(): Promise<void> {
    if (!this.storage) return;

    const promises: Promise<void>[] = [];
    for (const [, shard] of this.loadedShards) {
      if (shard.dirty) {
        const data = encodeShard(shard.index);
        promises.push(
          this.storage.write(shard.key, data).then(() => {
            shard.dirty = false;
          }),
        );
      }
    }
    await Promise.all(promises);
  }

  /** Compact all shards (rebuild without deleted nodes). */
  async compact(): Promise<void> {
    for (const key of this.shardKeys) {
      const shard = await this.getShard(key);
      const compacted = shard.index.compact();
      shard.index = compacted;
      shard.dirty = true;
    }
  }

  /** Close: flush and release all shards. */
  async close(): Promise<void> {
    await this.flush();
    this.loadedShards.clear();
  }

  /** Load shard metadata from storage (for reopening a persisted index). */
  async loadFromStorage(): Promise<void> {
    if (!this.storage) return;

    const keys = await this.storage.list();
    this.shardKeys = keys.sort();

    for (const key of this.shardKeys) {
      const data = await this.storage.read(key);
      if (!data) continue;

      const index = decodeShard(data);
      const state = index.getInternalState();

      // Rebuild ID mapping
      for (let i = 0; i < state.count; i++) {
        const extId = state.internalToExternal[i];
        this.idToShard.set(extId, key);
      }

      this.totalCount += index.size;

      // Keep loaded if we have capacity
      if (this.loadedShards.size < this.shardConfig.maxLoadedShards) {
        this.loadedShards.set(key, {
          key,
          index,
          dirty: false,
          lastAccess: this.accessCounter++,
        });
      }
    }

    // Set current shard
    if (this.shardKeys.length > 0) {
      const lastKey = this.shardKeys[this.shardKeys.length - 1];
      this.currentShardKey = lastKey;
      const shard = await this.getShard(lastKey);
      this.currentShardCount = shard.index.size;
    }
  }

  // --- Private ---

  private async createNewShard(): Promise<void> {
    const idx = this.shardKeys.length;
    const key = `shard-${String(idx).padStart(6, "0")}`;
    this.shardKeys.push(key);
    this.currentShardKey = key;
    this.currentShardCount = 0;

    const index = new HNSWIndex(this.hnswConfig);
    this.loadedShards.set(key, {
      key,
      index,
      dirty: true,
      lastAccess: this.accessCounter++,
    });

    // Evict if needed
    await this.evictIfNeeded();
  }

  private async getShard(key: string): Promise<LoadedShard> {
    const loaded = this.loadedShards.get(key);
    if (loaded) {
      loaded.lastAccess = this.accessCounter++;
      return loaded;
    }

    // Load from storage
    if (!this.storage) {
      throw new Error(`Shard ${key} not loaded and no storage backend`);
    }

    const data = await this.storage.read(key);
    if (!data) {
      throw new Error(`Shard ${key} not found in storage`);
    }

    const index = decodeShard(data);
    const shard: LoadedShard = {
      key,
      index,
      dirty: false,
      lastAccess: this.accessCounter++,
    };

    this.loadedShards.set(key, shard);
    await this.evictIfNeeded();

    return shard;
  }

  private async evictIfNeeded(): Promise<void> {
    while (this.loadedShards.size > this.shardConfig.maxLoadedShards) {
      // Find the least recently used shard
      let lruKey: string | null = null;
      let lruAccess = Infinity;

      for (const [key, shard] of this.loadedShards) {
        // Don't evict the current write shard
        if (key === this.currentShardKey) continue;
        if (shard.lastAccess < lruAccess) {
          lruAccess = shard.lastAccess;
          lruKey = key;
        }
      }

      if (!lruKey) break; // Can't evict anything

      const shard = this.loadedShards.get(lruKey)!;

      // Persist if dirty
      if (shard.dirty && this.storage) {
        const data = encodeShard(shard.index);
        await this.storage.write(shard.key, data);
      }

      this.loadedShards.delete(lruKey);
    }
  }
}
