import type { StorageBackend } from '../types.ts'

/** Map-backed storage. Browser-safe, no filesystem required. */
export class InMemoryStorage implements StorageBackend {
  private store = new Map<string, Uint8Array>()

  async write(key: string, data: Uint8Array): Promise<void> {
    this.store.set(key, new Uint8Array(data))
  }

  async read(key: string): Promise<Uint8Array | null> {
    return this.store.get(key) ?? null
  }

  async delete(key: string): Promise<boolean> {
    return this.store.delete(key)
  }

  async list(): Promise<string[]> {
    return [...this.store.keys()]
  }

  async exists(key: string): Promise<boolean> {
    return this.store.has(key)
  }
}
