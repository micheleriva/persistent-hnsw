/**
 * Generic binary heap parameterized by a comparator.
 * comparator(a, b) < 0 means a has higher priority (should be on top).
 *
 * For a min-heap by distance: (a, b) => a.distance - b.distance
 * For a max-heap by distance: (a, b) => b.distance - a.distance
 */
export class BinaryHeap<T> {
  private data: T[];
  private compare: (a: T, b: T) => number;

  constructor(comparator: (a: T, b: T) => number) {
    this.data = [];
    this.compare = comparator;
  }

  get size(): number {
    return this.data.length;
  }

  peek(): T | undefined {
    return this.data[0];
  }

  push(value: T): void {
    this.data.push(value);
    this.siftUp(this.data.length - 1);
  }

  pop(): T | undefined {
    const { data } = this;
    if (data.length === 0) return undefined;
    const top = data[0];
    const last = data.pop()!;
    if (data.length > 0) {
      data[0] = last;
      this.siftDown(0);
    }
    return top;
  }

  /** Combined pop + push — more efficient than separate operations. */
  replaceTop(value: T): T {
    const top = this.data[0];
    this.data[0] = value;
    this.siftDown(0);
    return top;
  }

  clear(): void {
    this.data.length = 0;
  }

  /** Iterate over elements (not in priority order). */
  values(): T[] {
    return this.data;
  }

  private siftUp(i: number): void {
    const { data, compare } = this;
    const item = data[i];
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (compare(item, data[parent]) >= 0) break;
      data[i] = data[parent];
      i = parent;
    }
    data[i] = item;
  }

  private siftDown(i: number): void {
    const { data, compare } = this;
    const len = data.length;
    const halfLen = len >> 1;
    const item = data[i];

    while (i < halfLen) {
      let child = (i << 1) + 1;
      let childVal = data[child];
      const right = child + 1;

      if (right < len && compare(data[right], childVal) < 0) {
        child = right;
        childVal = data[right];
      }

      if (compare(childVal, item) >= 0) break;
      data[i] = childVal;
      i = child;
    }

    data[i] = item;
  }
}
