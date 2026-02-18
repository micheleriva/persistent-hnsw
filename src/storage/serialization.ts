import { HNSWIndex } from "../hnsw_index.ts";
import type { DistanceMetric, HNSWConfig } from "../types.ts";
import { defaultHNSWConfig } from "../types.ts";

const MAGIC = 0x574E5348; // "HNSW" in little-endian
const VERSION = 1;
const HEADER_SIZE = 64;

const METRIC_MAP: Record<DistanceMetric, number> = {
  euclidean: 0,
  cosine: 1,
  inner_product: 2,
};

const METRIC_REVERSE: DistanceMetric[] = [
  "euclidean",
  "cosine",
  "inner_product",
];

/** Encode an HNSWIndex into a binary Uint8Array. */
export function encodeShard(index: HNSWIndex): Uint8Array {
  const state = index.getInternalState();
  const config = index.config;
  const { count } = state;
  const dim = config.dimensions;

  // Calculate sizes
  const idBytes = encodeIdTable(state.internalToExternal, count);
  const idTableSize = alignTo8(idBytes.byteLength);

  const vectorsSize = count * dim * 4;
  const hasCosineNorms = config.metric === "cosine";
  const normsSize = hasCosineNorms ? count * 4 : 0;
  const levelsSize = alignTo8(count);

  // Adjacency: for each layer, store header + neighbor counts + flat neighbors
  let adjacencySize = 0;
  const numLayers = state.adjacency.length;
  for (let l = 0; l < numLayers; l++) {
    const maxN = l === 0 ? config.Mmax0 : config.M;
    // Header: 4 (layer index) + 4 (nodeCount) + 4 (maxNeighbors)
    adjacencySize += 12;
    // Neighbor counts: count bytes, aligned to 4
    adjacencySize += alignTo4(count);
    // Flat neighbors: count * maxN * 4
    adjacencySize += count * maxN * 4;
  }

  const totalSize = HEADER_SIZE + idTableSize + vectorsSize + normsSize +
    levelsSize + 4 + adjacencySize; // +4 for numLayers

  const buffer = new ArrayBuffer(totalSize);
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);
  let offset = 0;

  // HEADER (64 bytes)
  view.setUint32(offset, MAGIC, true);
  offset += 4;
  view.setUint32(offset, VERSION, true);
  offset += 4;
  view.setUint32(offset, dim, true);
  offset += 4;
  view.setUint32(offset, count, true);
  offset += 4;
  view.setInt32(offset, state.maxLevel, true);
  offset += 4;
  view.setInt32(offset, state.entryPointId, true);
  offset += 4;
  view.setUint32(offset, config.M, true);
  offset += 4;
  view.setUint32(offset, config.Mmax0, true);
  offset += 4;
  view.setUint8(offset, METRIC_MAP[config.metric]);
  offset += 1;
  view.setUint8(
    offset,
    (hasCosineNorms ? 1 : 0) | (config.useHeuristic ? 2 : 0) |
      (config.keepPrunedConnections ? 4 : 0),
  );
  offset += 1;
  view.setUint32(offset, config.efConstruction, true);
  offset += 4;
  view.setUint32(offset, config.efSearch, true);
  offset += 4;
  // Reserved — pad to 64 bytes
  offset = HEADER_SIZE;

  // ID TABLE
  bytes.set(idBytes, offset);
  offset += idTableSize;

  // VECTORS
  const vectorData = new Uint8Array(
    state.vectors.buffer,
    state.vectors.byteOffset,
    count * dim * 4,
  );
  bytes.set(vectorData, offset);
  offset += vectorsSize;

  // NORMS (cosine only)
  if (hasCosineNorms) {
    const normData = new Uint8Array(
      state.norms.buffer,
      state.norms.byteOffset,
      count * 4,
    );
    bytes.set(normData, offset);
    offset += normsSize;
  }

  // LEVELS
  bytes.set(state.levels.subarray(0, count), offset);
  offset += levelsSize;

  // ADJACENCY
  view.setUint32(offset, numLayers, true);
  offset += 4;

  for (let l = 0; l < numLayers; l++) {
    const maxN = l === 0 ? config.Mmax0 : config.M;

    view.setUint32(offset, l, true);
    offset += 4;
    view.setUint32(offset, count, true);
    offset += 4;
    view.setUint32(offset, maxN, true);
    offset += 4;

    // Neighbor counts
    bytes.set(state.neighborCounts[l].subarray(0, count), offset);
    offset += alignTo4(count);

    // Flat neighbor array (only the used portion)
    const adjData = new Uint8Array(
      state.adjacency[l].buffer,
      state.adjacency[l].byteOffset,
      count * maxN * 4,
    );
    bytes.set(adjData, offset);
    offset += count * maxN * 4;
  }

  return bytes;
}

/** Decode a binary Uint8Array back into an HNSWIndex. */
export function decodeShard(data: Uint8Array): HNSWIndex {
  const view = new DataView(
    data.buffer,
    data.byteOffset,
    data.byteLength,
  );
  let offset = 0;

  // HEADER
  const magic = view.getUint32(offset, true);
  offset += 4;
  if (magic !== MAGIC) throw new Error("Invalid HNSW file: bad magic number");

  const version = view.getUint32(offset, true);
  offset += 4;
  if (version !== VERSION) {
    throw new Error(`Unsupported HNSW version: ${version}`);
  }

  const dim = view.getUint32(offset, true);
  offset += 4;
  const count = view.getUint32(offset, true);
  offset += 4;
  const maxLevel = view.getInt32(offset, true);
  offset += 4;
  const entryPointId = view.getInt32(offset, true);
  offset += 4;
  const M = view.getUint32(offset, true);
  offset += 4;
  const Mmax0 = view.getUint32(offset, true);
  offset += 4;
  const metricByte = view.getUint8(offset);
  offset += 1;
  const flags = view.getUint8(offset);
  offset += 1;
  const hasCosineNorms = (flags & 1) !== 0;
  const useHeuristic = (flags & 2) !== 0;
  const keepPrunedConnections = (flags & 4) !== 0;
  const efConstruction = view.getUint32(offset, true);
  offset += 4;
  const efSearch = view.getUint32(offset, true);
  offset += 4;

  offset = HEADER_SIZE;

  const metric = METRIC_REVERSE[metricByte];
  const config: HNSWConfig = defaultHNSWConfig(dim, {
    M,
    Mmax0,
    efConstruction,
    efSearch,
    metric,
    useHeuristic,
    keepPrunedConnections,
  });

  // ID TABLE
  const { ids, bytesRead } = decodeIdTable(data, offset, count);
  offset += alignTo8(bytesRead);

  // VECTORS
  const capacity = count; // Tight allocation for deserialized data
  const vectors = new Float32Array(capacity * dim);
  const vectorBytes = new Uint8Array(
    data.buffer,
    data.byteOffset + offset,
    count * dim * 4,
  );
  new Float32Array(vectors.buffer).set(
    new Float32Array(
      vectorBytes.buffer,
      vectorBytes.byteOffset,
      count * dim,
    ),
  );
  offset += count * dim * 4;

  // NORMS
  const norms = new Float32Array(capacity);
  if (hasCosineNorms) {
    const normBytes = new Uint8Array(
      data.buffer,
      data.byteOffset + offset,
      count * 4,
    );
    new Float32Array(norms.buffer).set(
      new Float32Array(normBytes.buffer, normBytes.byteOffset, count),
    );
    offset += count * 4;
  }

  // LEVELS
  const levels = new Uint8Array(capacity);
  levels.set(data.subarray(offset, offset + count));
  offset += alignTo8(count);

  // ADJACENCY
  const numLayers = view.getUint32(offset, true);
  offset += 4;

  const adjacency: Uint32Array[] = [];
  const neighborCounts: Uint8Array[] = [];

  for (let i = 0; i < numLayers; i++) {
    const _layerIdx = view.getUint32(offset, true);
    offset += 4;
    const nodeCount = view.getUint32(offset, true);
    offset += 4;
    const maxN = view.getUint32(offset, true);
    offset += 4;

    // Neighbor counts
    const nc = new Uint8Array(capacity);
    nc.set(data.subarray(offset, offset + nodeCount));
    offset += alignTo4(nodeCount);

    // Flat neighbors
    const adj = new Uint32Array(capacity * maxN);
    adj.fill(0xFFFFFFFF);
    const adjBytes = new Uint8Array(
      data.buffer,
      data.byteOffset + offset,
      nodeCount * maxN * 4,
    );
    const adjSrc = new Uint32Array(
      adjBytes.buffer,
      adjBytes.byteOffset,
      nodeCount * maxN,
    );
    adj.set(adjSrc);
    offset += nodeCount * maxN * 4;

    adjacency.push(adj);
    neighborCounts.push(nc);
  }

  return HNSWIndex.fromInternalState(config, {
    vectors,
    norms,
    levels,
    adjacency,
    neighborCounts,
    internalToExternal: ids,
    count,
    capacity,
    entryPointId,
    maxLevel,
  });
}

/** Read only the header from serialized data (useful for metadata without full deserialization). */
export function readHeader(
  data: Uint8Array,
): { dimensions: number; count: number; metric: DistanceMetric; M: number } {
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  return {
    dimensions: view.getUint32(8, true),
    count: view.getUint32(12, true),
    metric: METRIC_REVERSE[view.getUint8(32)],
    M: view.getUint32(24, true),
  };
}

// --- Helpers ---

function encodeIdTable(ids: string[], count: number): Uint8Array {
  const encoder = new TextEncoder();
  const parts: Uint8Array[] = [];
  let totalLen = 0;

  // First pass: encode all strings
  for (let i = 0; i < count; i++) {
    const encoded = encoder.encode(ids[i]);
    parts.push(encoded);
    totalLen += 4 + encoded.byteLength; // 4 bytes for length prefix
  }

  const result = new Uint8Array(totalLen);
  const view = new DataView(result.buffer);
  let offset = 0;

  for (let i = 0; i < count; i++) {
    view.setUint32(offset, parts[i].byteLength, true);
    offset += 4;
    result.set(parts[i], offset);
    offset += parts[i].byteLength;
  }

  return result;
}

function decodeIdTable(
  data: Uint8Array,
  startOffset: number,
  count: number,
): { ids: string[]; bytesRead: number } {
  const decoder = new TextDecoder();
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const ids: string[] = [];
  let offset = startOffset;

  for (let i = 0; i < count; i++) {
    const len = view.getUint32(offset, true);
    offset += 4;
    const strBytes = data.subarray(offset, offset + len);
    ids.push(decoder.decode(strBytes));
    offset += len;
  }

  return { ids, bytesRead: offset - startOffset };
}

function alignTo4(n: number): number {
  return (n + 3) & ~3;
}

function alignTo8(n: number): number {
  return (n + 7) & ~7;
}
