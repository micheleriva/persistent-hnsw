import { assertEquals } from 'https://deno.land/std@0.224.0/assert/assert_equals.ts'
import { assert } from 'https://deno.land/std@0.224.0/assert/assert.ts'
import { assertRejects } from 'https://deno.land/std@0.224.0/assert/assert_rejects.ts'
import { VectorStore } from '../src/vector_store.ts'
import { InMemoryStorage } from '../src/storage/in_memory_storage.ts'
import { FileSystemStorage } from '../src/storage/file_system_storage.ts'

Deno.test('VectorStore: basic insert and search', async () => {
  const store = VectorStore.create({
    hnsw: { dimensions: 3, seed: 42 },
  })

  await store.insert({ id: 'a', vector: [1, 0, 0] })
  await store.insert({ id: 'b', vector: [0, 1, 0] })
  await store.insert({ id: 'c', vector: [0, 0, 1] })

  assertEquals(store.size, 3)

  const results = await store.search([1, 0, 0], 2)
  assertEquals(results.length, 2)
  assertEquals(results[0].id, 'a')
  assertEquals(results[0].distance, 0)
})

Deno.test('VectorStore: batch insert', async () => {
  const store = VectorStore.create({
    hnsw: { dimensions: 2, seed: 42 },
  })

  await store.insert([
    { id: 'a', vector: [1, 0] },
    { id: 'b', vector: [0, 1] },
    { id: 'c', vector: [1, 1] },
  ])

  assertEquals(store.size, 3)
})

Deno.test('VectorStore: delete', async () => {
  const store = VectorStore.create({
    hnsw: { dimensions: 2, seed: 42 },
  })

  await store.insert([
    { id: 'a', vector: [1, 0] },
    { id: 'b', vector: [0, 1] },
  ])

  assertEquals(await store.delete('a'), true)
  assertEquals(store.size, 1)
  assertEquals(await store.delete('nonexistent'), false)
})

Deno.test('VectorStore: search with options', async () => {
  const store = VectorStore.create({
    hnsw: { dimensions: 2, seed: 42 },
  })

  await store.insert([
    { id: 'a', vector: [1, 0] },
    { id: 'b', vector: [0.9, 0.1] },
    { id: 'c', vector: [0, 1] },
  ])

  // Filter
  const filtered = await store.search([1, 0], 3, {
    filter: (id) => id !== 'a',
  })
  assertEquals(filtered[0].id, 'b')

  // efSearch override
  const withEf = await store.search([1, 0], 2, { efSearch: 200 })
  assertEquals(withEf.length, 2)
})

Deno.test('VectorStore: requires dimensions', () => {
  try {
    VectorStore.create({})
    assert(false, 'should have thrown')
  } catch (e) {
    assert((e as Error).message.includes('dimensions'))
  }
})

Deno.test('VectorStore: persistence with InMemoryStorage', async () => {
  const storage = new InMemoryStorage()

  // Create and populate
  const store1 = VectorStore.create({
    hnsw: { dimensions: 3, seed: 42 },
    storage,
  })

  await store1.insert([
    { id: 'a', vector: [1, 0, 0] },
    { id: 'b', vector: [0, 1, 0] },
    { id: 'c', vector: [0, 0, 1] },
  ])

  await store1.flush()
  await store1.close()

  // Reopen
  const store2 = await VectorStore.open({
    hnsw: { dimensions: 3, seed: 42 },
    storage,
  })

  assertEquals(store2.size, 3)

  const results = await store2.search([1, 0, 0], 2)
  assertEquals(results[0].id, 'a')
})

Deno.test('VectorStore: persistence with FileSystemStorage', async () => {
  const tmpDir = await Deno.makeTempDir()
  const storage = new FileSystemStorage(tmpDir)

  try {
    const store1 = VectorStore.create({
      hnsw: { dimensions: 4, seed: 42 },
      storage,
    })

    for (let i = 0; i < 50; i++) {
      await store1.insert({
        id: `v${i}`,
        vector: [
          Math.cos(i * 0.1),
          Math.sin(i * 0.1),
          Math.cos(i * 0.2),
          Math.sin(i * 0.2),
        ],
      })
    }

    const query = [1, 0, 1, 0]
    const resultsBefore = await store1.search(query, 5)
    await store1.flush()
    await store1.close()

    // Reopen
    const store2 = await VectorStore.open({
      hnsw: { dimensions: 4, seed: 42 },
      storage,
    })

    assertEquals(store2.size, 50)

    const resultsAfter = await store2.search(query, 5)
    assertEquals(resultsAfter.length, 5)
    // Same top results
    assertEquals(resultsAfter[0].id, resultsBefore[0].id)
  } finally {
    await Deno.remove(tmpDir, { recursive: true })
  }
})

Deno.test('VectorStore: sharding works', async () => {
  const store = VectorStore.create({
    hnsw: { dimensions: 2, seed: 42 },
    sharding: { maxVectorsPerShard: 10 },
  })

  for (let i = 0; i < 25; i++) {
    await store.insert({
      id: `v${i}`,
      vector: [Math.cos(i * 0.3), Math.sin(i * 0.3)],
    })
  }

  assertEquals(store.size, 25)

  const results = await store.search([1, 0], 5)
  assertEquals(results.length, 5)
  assert(results[0].distance <= results[1].distance)
})

Deno.test('VectorStore: compact and close', async () => {
  const store = VectorStore.create({
    hnsw: { dimensions: 2, seed: 42 },
  })

  await store.insert([
    { id: 'a', vector: [1, 0] },
    { id: 'b', vector: [0, 1] },
    { id: 'c', vector: [1, 1] },
  ])

  await store.delete('b')
  await store.compact()
  assertEquals(store.size, 2)

  await store.close()
})

Deno.test('VectorStore: open requires storage', async () => {
  await assertRejects(
    () => VectorStore.open({ hnsw: { dimensions: 3 } }),
    Error,
    'storage',
  )
})
