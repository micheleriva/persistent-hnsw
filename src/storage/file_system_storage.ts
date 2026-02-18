import type { StorageBackend } from "../types.ts";

/** Deno filesystem-backed storage. Files stored as {basePath}/{key}.hnsw */
export class FileSystemStorage implements StorageBackend {
  private basePath: string;

  constructor(basePath: string) {
    this.basePath = basePath;
  }

  async write(key: string, data: Uint8Array): Promise<void> {
    await Deno.mkdir(this.basePath, { recursive: true });
    await Deno.writeFile(this.filePath(key), data);
  }

  async read(key: string): Promise<Uint8Array | null> {
    try {
      return await Deno.readFile(this.filePath(key));
    } catch (e) {
      if (e instanceof Deno.errors.NotFound) return null;
      throw e;
    }
  }

  async delete(key: string): Promise<boolean> {
    try {
      await Deno.remove(this.filePath(key));
      return true;
    } catch (e) {
      if (e instanceof Deno.errors.NotFound) return false;
      throw e;
    }
  }

  async list(): Promise<string[]> {
    const keys: string[] = [];
    try {
      for await (const entry of Deno.readDir(this.basePath)) {
        if (entry.isFile && entry.name.endsWith(".hnsw")) {
          keys.push(entry.name.slice(0, -5));
        }
      }
    } catch (e) {
      if (e instanceof Deno.errors.NotFound) return [];
      throw e;
    }
    return keys;
  }

  async exists(key: string): Promise<boolean> {
    try {
      await Deno.stat(this.filePath(key));
      return true;
    } catch (e) {
      if (e instanceof Deno.errors.NotFound) return false;
      throw e;
    }
  }

  private filePath(key: string): string {
    return `${this.basePath}/${key}.hnsw`;
  }
}
