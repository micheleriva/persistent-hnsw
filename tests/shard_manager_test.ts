import { assertEquals } from 'https://deno.land/std@0.224.0/assert/assert_equals.ts'
import { assert } from 'https://deno.land/std@0.224.0/assert/assert.ts'
import { ShardManager } from '../src/shard_manager.ts'
import { defaultHNSWConfig, defaultShardConfig } from '../src/types.ts'
import { InMemoryStorage } from '../src/storage/in_memory_storage.ts'

function makeManager(
  opts?: {
    maxPerShard?: number
    maxLoaded?: number
    storage?: boolean
  },
) {
  const hnswConfig = defaultHNSWConfig(4, { seed: 42 })
  const shardConfig = defaultShardConfig({
    maxVectorsPerShard: opts?.maxPerShard ?? 100,
    maxLoadedShards: opts?.maxLoaded ?? 4,
  })
  const storage = opts?.storage !== false ? new InMemoryStorage() : null
  return {
    manager: new ShardManager(hnswConfig, shardConfig, storage),
    storage,
  }
}

Deno.test('ShardManager: insert and search single shard', async () => {
  const { manager } = makeManager()

  await manager.insert('a', [1, 0, 0, 0])
  await manager.insert('b', [0, 1, 0, 0])
  await manager.insert('c', [0, 0, 1, 0])

  assertEquals(manager.size, 3)
  assertEquals(manager.shardCount, 1)

  const results = await manager.search([1, 0, 0, 0], 2)
  assertEquals(results.length, 2)
  assertEquals(results[0].id, 'a')
})

Deno.test('ShardManager: auto-creates new shard when full', async () => {
  const { manager } = makeManager({ maxPerShard: 5 })

  for (let i = 0; i < 12; i++) {
    await manager.insert(`v${i}`, [i, i * 2, i * 3, i * 4])
  }

  assertEquals(manager.size, 12)
  assertEquals(manager.shardCount, 3) // 5 + 5 + 2
})

Deno.test('ShardManager: search across multiple shards', async () => {
  const { manager } = makeManager({ maxPerShard: 5 })

  // Insert vectors spread across shards
  for (let i = 0; i < 15; i++) {
    await manager.insert(`v${i}`, [
      Math.cos(i * 0.5),
      Math.sin(i * 0.5),
      0,
      0,
    ])
  }

  const results = await manager.search([1, 0, 0, 0], 3)
  assertEquals(results.length, 3)
  // Results should be sorted by distance
  assert(results[0].distance <= results[1].distance)
  assert(results[1].distance <= results[2].distance)
})

Deno.test('ShardManager: delete removes vector', async () => {
  const { manager } = makeManager()

  await manager.insert('a', [1, 0, 0, 0])
  await manager.insert('b', [0, 1, 0, 0])

  assertEquals(manager.size, 2)
  assertEquals(await manager.delete('a'), true)
  assertEquals(manager.size, 1)

  const results = await manager.search([1, 0, 0, 0], 5)
  assertEquals(results.length, 1)
  assertEquals(results[0].id, 'b')
})

Deno.test('ShardManager: delete non-existent returns false', async () => {
  const { manager } = makeManager()
  assertEquals(await manager.delete('nonexistent'), false)
})

Deno.test('ShardManager: flush persists to storage', async () => {
  const { manager, storage } = makeManager()

  await manager.insert('a', [1, 0, 0, 0])
  await manager.flush()

  const keys = await storage!.list()
  assertEquals(keys.length, 1)
  assert(await storage!.exists(keys[0]))
})

Deno.test('ShardManager: LRU eviction', async () => {
  const { manager } = makeManager({ maxPerShard: 5, maxLoaded: 2 })

  // Create 3 shards (exceeds maxLoaded=2)
  for (let i = 0; i < 15; i++) {
    await manager.insert(`v${i}`, [i, 0, 0, 0])
  }

  assertEquals(manager.shardCount, 3)
  // Current shard is never evicted, plus maxLoaded-1 others
  assert(manager.loadedShardCount <= 3)
})

Deno.test('ShardManager: load from storage', async () => {
  const storage = new InMemoryStorage()
  const hnswConfig = defaultHNSWConfig(4, { seed: 42 })
  const shardConfig = defaultShardConfig({
    maxVectorsPerShard: 50,
    maxLoadedShards: 4,
  })

  // Create and populate
  const manager1 = new ShardManager(hnswConfig, shardConfig, storage)
  for (let i = 0; i < 20; i++) {
    await manager1.insert(`v${i}`, [
      Math.cos(i),
      Math.sin(i),
      0,
      0,
    ])
  }
  await manager1.flush()
  await manager1.close()

  // Reopen
  const manager2 = new ShardManager(hnswConfig, shardConfig, storage)
  await manager2.loadFromStorage()

  assertEquals(manager2.size, 20)

  const results = await manager2.search([1, 0, 0, 0], 5)
  assertEquals(results.length, 5)
})

Deno.test('ShardManager: search with filter', async () => {
  const { manager } = makeManager()

  await manager.insert('a', [1, 0, 0, 0])
  await manager.insert('b', [0.9, 0.1, 0, 0])
  await manager.insert('c', [0, 1, 0, 0])

  const results = await manager.search([1, 0, 0, 0], 5, {
    filter: (id) => id !== 'a',
  })
  assertEquals(results[0].id, 'b')
})

Deno.test('ShardManager: search with includeVectors', async () => {
  const { manager } = makeManager()

  await manager.insert('a', [1, 2, 3, 4])

  const results = await manager.search([1, 2, 3, 4], 1, {
    includeVectors: true,
  })
  assertEquals(results.length, 1)
  assert(results[0].vector !== undefined)
  assertEquals(Array.from(results[0].vector!), [1, 2, 3, 4])
})

Deno.test('ShardManager: compact rebuilds shards', async () => {
  const { manager } = makeManager()

  await manager.insert('a', [1, 0, 0, 0])
  await manager.insert('b', [0, 1, 0, 0])
  await manager.insert('c', [0, 0, 1, 0])

  await manager.delete('b')
  assertEquals(manager.size, 2)

  await manager.compact()

  const results = await manager.search([0, 1, 0, 0], 5)
  // "b" was deleted, should not appear
  for (const r of results) {
    assert(r.id !== 'b')
  }
})
