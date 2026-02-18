# HNSW Benchmark Results

> Benchmarked on 2026-02-18 — Deno 1.36.4, aarch64-darwin

## Scaling (128 dimensions, M=16, efConstruction=200)

| Vectors | Insert ops/s | Build Time | Search (ef=200) | Search p99 | QPS (ef=200) | Recall@ef200 | Recall@ef500 | Memory  | On Disk |
| ------- | ------------ | ---------- | --------------- | ---------- | ------------ | ------------ | ------------ | ------- | ------- |
| 1.0K    | 591          | 1.7s       | 313µs           | 321µs      | 3.2K         | 100.0%       | 100.0%       | 0.8 MB  | 0.8 MB  |
| 5.0K    | 371          | 13.5s      | 695µs           | 720µs      | 1.4K         | 99.0%        | 100.0%       | 4.0 MB  | 4.0 MB  |
| 10.0K   | 313          | 31.9s      | 877µs           | 956µs      | 1.1K         | 96.8%        | 100.0%       | 8.0 MB  | 8.1 MB  |
| 25.0K   | 267          | 93.8s      | 1095µs          | 1207µs     | 913          | 89.0%        | 97.7%        | 20.1 MB | 20.2 MB |
| 50.0K   | 242          | 206.4s     | 1376µs          | 1458µs     | 727          | 79.3%        | 92.7%        | 49.4 MB | 49.7 MB |

### Insertion Throughput

```
1.0K vectors  │████████████████████████████████████████ 591.2 ops/s
5.0K vectors  │█████████████████████████ 371.0 ops/s
10.0K vectors │█████████████████████ 313.0 ops/s
25.0K vectors │██████████████████ 266.6 ops/s
50.0K vectors │████████████████ 242.2 ops/s
```

### Search Latency (median)

```
1.0K vectors  │█████████ 313 µs
5.0K vectors  │████████████████████ 695 µs
10.0K vectors │█████████████████████████ 877 µs
25.0K vectors │████████████████████████████████ 1.1K µs
50.0K vectors │████████████████████████████████████████ 1.4K µs
```

### Memory Usage

```
1.0K vectors  │█ 0.8 MB
5.0K vectors  │███ 4.0 MB
10.0K vectors │██████ 8.0 MB
25.0K vectors │████████████████ 20.1 MB
50.0K vectors │████████████████████████████████████████ 49.4 MB
```

## Recall vs Latency Tradeoff (10K vectors, 128d, M=16)

| efSearch | Recall@10 | Median Latency | Throughput  |
| -------- | --------- | -------------- | ----------- |
| 10       | 34.2%     | 87µs           | 11.5K ops/s |
| 50       | 73.8%     | 322µs          | 3.1K ops/s  |
| 100      | 87.7%     | 523µs          | 1.9K ops/s  |
| 200      | 96.5%     | 950µs          | 1.1K ops/s  |
| 400      | 99.7%     | 1533µs         | 652 ops/s   |

### Recall–Latency Curve

```
Recall@10 │
 100% │                                                       ●
  95% │                                  ●                     
  90% │                   ●                                    
  85% │                                                        
  80% │                                                        
  75% │            ●                                           
  70% │                                                        
  65% │                                                        
  60% │                                                        
  55% │                                                        
  50% │                                                        
  45% │                                                        
  40% │                                                        
      └────────────────────────────────────────────────────────
       87µs                                            1533µs
                   Search Latency →
```

## Dimension Scaling (5K vectors, M=16, efConstruction=100, efSearch=100)

| Dimensions | Insert ops/s | Search ops/s | Search Median | Recall@10 | Memory  | On Disk |
| ---------- | ------------ | ------------ | ------------- | --------- | ------- | ------- |
| 32         | 759          | 4.6K         | 216µs         | 99.7%     | 2.2 MB  | 2.2 MB  |
| 64         | 569          | 3.4K         | 295µs         | 98.0%     | 2.8 MB  | 2.8 MB  |
| 128        | 391          | 2.3K         | 443µs         | 92.7%     | 4.0 MB  | 4.0 MB  |
| 256        | 246          | 1.2K         | 824µs         | 88.3%     | 6.5 MB  | 6.5 MB  |
| 384        | 170          | 963          | 1039µs        | 89.7%     | 8.9 MB  | 8.9 MB  |
| 512        | 140          | 723          | 1384µs        | 88.7%     | 11.3 MB | 11.4 MB |
| 768        | 97           | 538          | 1859µs        | 79.3%     | 16.2 MB | 16.2 MB |
| 1024       | 77           | 431          | 2319µs        | 83.0%     | 21.1 MB | 21.1 MB |
| 1536       | 52           | 292          | 3427µs        | 76.3%     | 30.9 MB | 30.9 MB |

### Search Throughput by Dimension

```
32d   │████████████████████████████████████████ 4.6K ops/s
64d   │█████████████████████████████ 3.4K ops/s
128d  │████████████████████ 2.3K ops/s
256d  │███████████ 1.2K ops/s
384d  │████████ 962.6 ops/s
512d  │██████ 722.6 ops/s
768d  │█████ 537.9 ops/s
1024d │████ 431.3 ops/s
1536d │███ 291.8 ops/s
```

### Memory by Dimension (after shrinkToFit)

```
32d   │███ 2.2 MB
64d   │████ 2.8 MB
128d  │█████ 4.0 MB
256d  │████████ 6.5 MB
384d  │████████████ 8.9 MB
512d  │███████████████ 11.3 MB
768d  │█████████████████████ 16.2 MB
1024d │███████████████████████████ 21.1 MB
1536d │████████████████████████████████████████ 30.9 MB
```

### Per-Dimension Details

<details>
<summary><b>32 dimensions</b> — 4.6K search/s, 2.2 MB</summary>

| Metric                     | Value      |
| -------------------------- | ---------- |
| Vectors                    | 5000       |
| Dimensions                 | 32         |
| Insert throughput          | 759 ops/s  |
| Build time                 | 6.6s       |
| Search median latency      | 216µs      |
| Search p99 latency         | 233µs      |
| Search throughput          | 4.6K ops/s |
| Recall@10 (ef=100)         | 99.7%      |
| Memory (live)              | 2.3 MB     |
| Memory (after shrinkToFit) | 2.2 MB     |
| Serialized size            | 2.2 MB     |
| Bytes per vector (memory)  | 457 B      |
| Bytes per vector (disk)    | 462 B      |

</details>

<details>
<summary><b>64 dimensions</b> — 3.4K search/s, 2.8 MB</summary>

| Metric                     | Value      |
| -------------------------- | ---------- |
| Vectors                    | 5000       |
| Dimensions                 | 64         |
| Insert throughput          | 569 ops/s  |
| Build time                 | 8.8s       |
| Search median latency      | 295µs      |
| Search p99 latency         | 310µs      |
| Search throughput          | 3.4K ops/s |
| Recall@10 (ef=100)         | 98.0%      |
| Memory (live)              | 2.9 MB     |
| Memory (after shrinkToFit) | 2.8 MB     |
| Serialized size            | 2.8 MB     |
| Bytes per vector (memory)  | 585 B      |
| Bytes per vector (disk)    | 590 B      |

</details>

<details>
<summary><b>128 dimensions</b> — 2.3K search/s, 4.0 MB</summary>

| Metric                     | Value      |
| -------------------------- | ---------- |
| Vectors                    | 5000       |
| Dimensions                 | 128        |
| Insert throughput          | 391 ops/s  |
| Build time                 | 12.8s      |
| Search median latency      | 443µs      |
| Search p99 latency         | 459µs      |
| Search throughput          | 2.3K ops/s |
| Recall@10 (ef=100)         | 92.7%      |
| Memory (live)              | 4.2 MB     |
| Memory (after shrinkToFit) | 4.0 MB     |
| Serialized size            | 4.0 MB     |
| Bytes per vector (memory)  | 841 B      |
| Bytes per vector (disk)    | 846 B      |

</details>

<details>
<summary><b>256 dimensions</b> — 1.2K search/s, 6.5 MB</summary>

| Metric                     | Value      |
| -------------------------- | ---------- |
| Vectors                    | 5000       |
| Dimensions                 | 256        |
| Insert throughput          | 246 ops/s  |
| Build time                 | 20.3s      |
| Search median latency      | 824µs      |
| Search p99 latency         | 1291µs     |
| Search throughput          | 1.2K ops/s |
| Recall@10 (ef=100)         | 88.3%      |
| Memory (live)              | 6.7 MB     |
| Memory (after shrinkToFit) | 6.5 MB     |
| Serialized size            | 6.5 MB     |
| Bytes per vector (memory)  | 1353 B     |
| Bytes per vector (disk)    | 1358 B     |

</details>

<details>
<summary><b>384 dimensions</b> — 963 search/s, 8.9 MB</summary>

| Metric                     | Value     |
| -------------------------- | --------- |
| Vectors                    | 5000      |
| Dimensions                 | 384       |
| Insert throughput          | 170 ops/s |
| Build time                 | 29.3s     |
| Search median latency      | 1039µs    |
| Search p99 latency         | 1134µs    |
| Search throughput          | 963 ops/s |
| Recall@10 (ef=100)         | 89.7%     |
| Memory (live)              | 9.2 MB    |
| Memory (after shrinkToFit) | 8.9 MB    |
| Serialized size            | 8.9 MB    |
| Bytes per vector (memory)  | 1865 B    |
| Bytes per vector (disk)    | 1870 B    |

</details>

<details>
<summary><b>512 dimensions</b> — 723 search/s, 11.3 MB</summary>

| Metric                     | Value     |
| -------------------------- | --------- |
| Vectors                    | 5000      |
| Dimensions                 | 512       |
| Insert throughput          | 140 ops/s |
| Build time                 | 35.7s     |
| Search median latency      | 1384µs    |
| Search p99 latency         | 1418µs    |
| Search throughput          | 723 ops/s |
| Recall@10 (ef=100)         | 88.7%     |
| Memory (live)              | 11.8 MB   |
| Memory (after shrinkToFit) | 11.3 MB   |
| Serialized size            | 11.4 MB   |
| Bytes per vector (memory)  | 2377 B    |
| Bytes per vector (disk)    | 2382 B    |

</details>

<details>
<summary><b>768 dimensions</b> — 538 search/s, 16.2 MB</summary>

| Metric                     | Value     |
| -------------------------- | --------- |
| Vectors                    | 5000      |
| Dimensions                 | 768       |
| Insert throughput          | 97 ops/s  |
| Build time                 | 51.4s     |
| Search median latency      | 1859µs    |
| Search p99 latency         | 1978µs    |
| Search throughput          | 538 ops/s |
| Recall@10 (ef=100)         | 79.3%     |
| Memory (live)              | 16.8 MB   |
| Memory (after shrinkToFit) | 16.2 MB   |
| Serialized size            | 16.2 MB   |
| Bytes per vector (memory)  | 3401 B    |
| Bytes per vector (disk)    | 3406 B    |

</details>

<details>
<summary><b>1024 dimensions</b> — 431 search/s, 21.1 MB</summary>

| Metric                     | Value     |
| -------------------------- | --------- |
| Vectors                    | 5000      |
| Dimensions                 | 1024      |
| Insert throughput          | 77 ops/s  |
| Build time                 | 64.7s     |
| Search median latency      | 2319µs    |
| Search p99 latency         | 2371µs    |
| Search throughput          | 431 ops/s |
| Recall@10 (ef=100)         | 83.0%     |
| Memory (live)              | 21.9 MB   |
| Memory (after shrinkToFit) | 21.1 MB   |
| Serialized size            | 21.1 MB   |
| Bytes per vector (memory)  | 4425 B    |
| Bytes per vector (disk)    | 4430 B    |

</details>

<details>
<summary><b>1536 dimensions</b> — 292 search/s, 30.9 MB</summary>

| Metric                     | Value     |
| -------------------------- | --------- |
| Vectors                    | 5000      |
| Dimensions                 | 1536      |
| Insert throughput          | 52 ops/s  |
| Build time                 | 96.9s     |
| Search median latency      | 3427µs    |
| Search p99 latency         | 5127µs    |
| Search throughput          | 292 ops/s |
| Recall@10 (ef=100)         | 76.3%     |
| Memory (live)              | 32.0 MB   |
| Memory (after shrinkToFit) | 30.9 MB   |
| Serialized size            | 30.9 MB   |
| Bytes per vector (memory)  | 6473 B    |
| Bytes per vector (disk)    | 6478 B    |

</details>

## Distance Function Throughput (Euclidean L2)

### Euclidean Distance ops/sec

```
128d  │████████████████████████████████████████ 10.5M ops/s
256d  │████████████████████ 5.3M ops/s
768d  │███████ 1.8M ops/s
1536d │███ 914.2K ops/s
```

## Serialization

| Vectors | Size    | Encode Time | Encode Speed | Decode Time | Decode Speed |
| ------- | ------- | ----------- | ------------ | ----------- | ------------ |
| 1.0K    | 0.8 MB  | 1ms         | 1592 MB/s    | 2ms         | 450 MB/s     |
| 10.0K   | 8.1 MB  | 10ms        | 801 MB/s     | 5ms         | 1570 MB/s    |
| 25.0K   | 20.2 MB | 10ms        | 2061 MB/s    | 12ms        | 1634 MB/s    |

## Impact of M Parameter (10K vectors, 128d, efConstruction=200, efSearch=200)

| M  | Insert ops/s | Search ops/s | Recall@10 | Memory |
| -- | ------------ | ------------ | --------- | ------ |
| 4  | 2.2K         | 2.9K         | 61.2%     | 6.2 MB |
| 8  | 1.1K         | 1.8K         | 84.4%     | 6.8 MB |
| 16 | 314          | 1.2K         | 96.8%     | 8.0 MB |
| 32 | 76           | 840          | 99.4%     | 9.8 MB |
