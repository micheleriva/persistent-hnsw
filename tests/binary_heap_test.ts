import { assertEquals } from "https://deno.land/std@0.224.0/assert/assert_equals.ts";
import { BinaryHeap } from "../src/binary_heap.ts";

Deno.test("BinaryHeap: min-heap basic operations", () => {
  const heap = new BinaryHeap<number>((a, b) => a - b);
  heap.push(5);
  heap.push(3);
  heap.push(7);
  heap.push(1);

  assertEquals(heap.size, 4);
  assertEquals(heap.peek(), 1);
  assertEquals(heap.pop(), 1);
  assertEquals(heap.pop(), 3);
  assertEquals(heap.pop(), 5);
  assertEquals(heap.pop(), 7);
  assertEquals(heap.size, 0);
});

Deno.test("BinaryHeap: max-heap basic operations", () => {
  const heap = new BinaryHeap<number>((a, b) => b - a);
  heap.push(5);
  heap.push(3);
  heap.push(7);
  heap.push(1);

  assertEquals(heap.peek(), 7);
  assertEquals(heap.pop(), 7);
  assertEquals(heap.pop(), 5);
});

Deno.test("BinaryHeap: pop from empty returns undefined", () => {
  const heap = new BinaryHeap<number>((a, b) => a - b);
  assertEquals(heap.pop(), undefined);
  assertEquals(heap.peek(), undefined);
});

Deno.test("BinaryHeap: replaceTop", () => {
  const heap = new BinaryHeap<number>((a, b) => a - b);
  heap.push(1);
  heap.push(5);
  heap.push(3);

  const old = heap.replaceTop(4);
  assertEquals(old, 1);
  assertEquals(heap.peek(), 3);
});

Deno.test("BinaryHeap: clear", () => {
  const heap = new BinaryHeap<number>((a, b) => a - b);
  heap.push(1);
  heap.push(2);
  heap.clear();
  assertEquals(heap.size, 0);
  assertEquals(heap.pop(), undefined);
});

Deno.test("BinaryHeap: with objects", () => {
  interface Item {
    id: string;
    distance: number;
  }
  const heap = new BinaryHeap<Item>((a, b) => a.distance - b.distance);
  heap.push({ id: "a", distance: 0.5 });
  heap.push({ id: "b", distance: 0.1 });
  heap.push({ id: "c", distance: 0.9 });

  assertEquals(heap.pop()!.id, "b");
  assertEquals(heap.pop()!.id, "a");
  assertEquals(heap.pop()!.id, "c");
});

Deno.test("BinaryHeap: single element", () => {
  const heap = new BinaryHeap<number>((a, b) => a - b);
  heap.push(42);
  assertEquals(heap.peek(), 42);
  assertEquals(heap.pop(), 42);
  assertEquals(heap.size, 0);
});

Deno.test("BinaryHeap: duplicate values", () => {
  const heap = new BinaryHeap<number>((a, b) => a - b);
  heap.push(3);
  heap.push(3);
  heap.push(3);
  assertEquals(heap.size, 3);
  assertEquals(heap.pop(), 3);
  assertEquals(heap.pop(), 3);
  assertEquals(heap.pop(), 3);
});

Deno.test("BinaryHeap: values returns all elements", () => {
  const heap = new BinaryHeap<number>((a, b) => a - b);
  heap.push(5);
  heap.push(3);
  heap.push(7);
  const values = heap.values();
  assertEquals(values.length, 3);
  assertEquals(values.sort(), [3, 5, 7]);
});
