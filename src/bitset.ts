/**
 * Compact visited set backed by Uint32Array.
 * Each bit represents one node ID. Much faster than Set<number> for dense IDs.
 */
export class Bitset {
  private bits: Uint32Array;

  constructor(capacity: number) {
    this.bits = new Uint32Array(Math.ceil(capacity / 32));
  }

  set(i: number): void {
    this.bits[i >> 5] |= 1 << (i & 31);
  }

  has(i: number): boolean {
    return (this.bits[i >> 5] & (1 << (i & 31))) !== 0;
  }

  clear(): void {
    this.bits.fill(0);
  }

  /** Grow to support at least `newCapacity` IDs. */
  grow(newCapacity: number): void {
    const needed = Math.ceil(newCapacity / 32);
    if (needed <= this.bits.length) return;
    const newBits = new Uint32Array(needed);
    newBits.set(this.bits);
    this.bits = newBits;
  }

  get capacity(): number {
    return this.bits.length * 32;
  }
}
