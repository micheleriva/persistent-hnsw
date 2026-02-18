# Runtime Comparison

> Benchmarked on 2026-02-18 — aarch64-darwin

Runtimes tested: Node v22.12.0, Bun 1.3.9

## Insertion (128d, M=16, efConstruction=200)

| Vectors | Node v22.12.0 | Bun 1.3.9 |
| ------- | ------------- | --------- |
|    1.0K |     621 ops/s | 762 ops/s |
|    5.0K |     393 ops/s | 487 ops/s |
|   10.0K |     335 ops/s | 423 ops/s |

## Search (10K index, 128d, M=16, k=10)

| efSearch | Node v22.12.0 (µs/q) | Bun 1.3.9 (µs/q) | Node v22.12.0 (qps) | Bun 1.3.9 (qps) |
| -------- | -------------------- | ---------------- | ------------------- | --------------- |
|    ef=10 |                   78 |               64 |               12.8K |           15.6K |
|    ef=50 |                  278 |              177 |                3.6K |            5.6K |
|   ef=100 |                  474 |              306 |                2.1K |            3.3K |
|   ef=200 |                  822 |              508 |                1.2K |            2.0K |
|   ef=400 |                 1446 |              808 |                 692 |            1.2K |

### Search Throughput at ef=100

```
Node v22.12.0    ██████████████████████████ 2.1K qps
Bun 1.3.9        ████████████████████████████████████████ 3.3K qps
```

## Euclidean Distance Throughput

| Dimensions | Node v22.12.0 | Bun 1.3.9   |
| ---------- | ------------- | ----------- |
|       128d |   10.1M ops/s | 19.4M ops/s |
|       256d |    5.2M ops/s | 10.0M ops/s |
|       768d |    1.7M ops/s |  3.4M ops/s |
|      1536d |  870.7K ops/s |  1.7M ops/s |

### Euclidean 128d Throughput

```
Node v22.12.0    █████████████████████ 10.1M ops/s
Bun 1.3.9        ████████████████████████████████████████ 19.4M ops/s
```
