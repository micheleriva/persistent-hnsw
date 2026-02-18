import type { StorageBackend } from '../types.ts'

// Use node:fs/promises and node:path — supported by Deno, Node.js, and Bun
// dnt-shim-ignore
import * as fs from 'node:fs/promises'
// dnt-shim-ignore
import * as path from 'node:path'

function isNotFound(e: unknown): boolean {
  return (
    e instanceof Error &&
    'code' in e &&
    (e as { code: string }).code === 'ENOENT'
  )
}

/** Filesystem-backed storage. Works on Node.js, Deno, and Bun. */
export class FileSystemStorage implements StorageBackend {
  private basePath: string

  constructor(basePath: string) {
    this.basePath = basePath
  }

  async write(key: string, data: Uint8Array): Promise<void> {
    await fs.mkdir(this.basePath, { recursive: true })
    await fs.writeFile(this.filePath(key), data)
  }

  async read(key: string): Promise<Uint8Array | null> {
    try {
      const buf = await fs.readFile(this.filePath(key))
      return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength)
    } catch (e) {
      if (isNotFound(e)) return null
      throw e
    }
  }

  async delete(key: string): Promise<boolean> {
    try {
      await fs.unlink(this.filePath(key))
      return true
    } catch (e) {
      if (isNotFound(e)) return false
      throw e
    }
  }

  async list(): Promise<string[]> {
    const keys: string[] = []
    try {
      const entries = await fs.readdir(this.basePath, { withFileTypes: true })
      for (const entry of entries) {
        if (entry.isFile() && entry.name.endsWith('.hnsw')) {
          keys.push(entry.name.slice(0, -5))
        }
      }
    } catch (e) {
      if (isNotFound(e)) return []
      throw e
    }
    return keys
  }

  async exists(key: string): Promise<boolean> {
    try {
      await fs.stat(this.filePath(key))
      return true
    } catch (e) {
      if (isNotFound(e)) return false
      throw e
    }
  }

  private filePath(key: string): string {
    return path.join(this.basePath, `${key}.hnsw`)
  }
}
